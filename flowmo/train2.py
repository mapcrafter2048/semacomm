# filepath: /home/utka/proj/semacomm/flowmo/train.py
"""FlowMo train script with DeepSpeed."""

import contextlib
import glob
import os
import shutil
import time
import json
import sys
import fsspec
import lpips
import torch
import torch.distributed as dist
import torch.optim as optim
from mup import MuAdam, MuAdamW
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import wandb

import models, perceptual_loss, train_utils

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

BFLOAT16_IS_AVAILABLE = None


def _get_norm(model, getter):
    grad_norms = [
        (getter(p) ** 2).sum() for p in model.parameters() if p.grad is not None
    ]
    
    if not grad_norms:
        return torch.tensor(0.0, device='cuda')
    
    return torch.stack(grad_norms).sum().sqrt()


def get_args_and_config_deepspeed():
    """Custom config loader that handles DeepSpeed arguments."""
    args, unknown = train_utils.get_args_and_unknown()

    config = OmegaConf.load("flowmo/configs/base.yaml")
    
    # Filter out DeepSpeed-specific arguments
    deepspeed_args = ['--local_rank', '--local-rank', '--deepspeed', '--deepspeed_config']
    filtered_unknown = []
    
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        # Skip DeepSpeed arguments and their values
        if any(ds_arg in arg for ds_arg in deepspeed_args):
            if '=' not in arg and i + 1 < len(unknown):
                i += 1  # Skip the value as well
        else:
            filtered_unknown.append(arg)
        i += 1
    
    # Only merge non-DeepSpeed CLI arguments
    if filtered_unknown:
        OmegaConf.set_struct(config, True)
        cli = OmegaConf.from_dotlist(filtered_unknown)
        config = OmegaConf.merge(config, cli)
    else:
        OmegaConf.set_struct(config, True)
    
    return args, config


def log_gpu_memory(step, label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"Step {step} {label}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")



def train_step(config, model_engine, batch, aux_state):
    assert BFLOAT16_IS_AVAILABLE is not None
    dtype = torch.bfloat16 if BFLOAT16_IS_AVAILABLE else torch.float32

    aux = {"loss_dict": {}}
    
    b = batch["image"].shape[0]
    
    # With DeepSpeed, we don't need manual gradient accumulation
    # DeepSpeed handles it automatically based on ds_config
    
    with torch.autocast("cuda", dtype=dtype):
        loss, aux = models.rf_loss(config, model_engine, batch, aux_state)

    # DeepSpeed backward and step
    model_engine.backward(loss)

    if config.opt.log_norms:
        original_grad_norm = _get_norm(model_engine, getter=lambda p: p.grad)
        aux["loss_dict"]["debug/original_grad_norm"] = original_grad_norm
        aux["loss_dict"]["debug/param_norm"] = _get_norm(model_engine, getter=lambda p: p)
    
    model_engine.step()


    return loss.detach(), aux


def main(args, config):
    config = train_utils.restore_config(config)
    print(torch.__version__)
    models.MUP_ENABLED = config.model.enable_mup

    # Initialize DeepSpeed AFTER config parsing
    deepspeed.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    log_dir = os.path.join(args.results_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    global BFLOAT16_IS_AVAILABLE
    BFLOAT16_IS_AVAILABLE = (
        train_utils.bfloat16_is_available() and config.trainer.enable_bfloat16
    )
    print("Using bfloat16: ", BFLOAT16_IS_AVAILABLE)

    torch.manual_seed(0)

    # Build model without moving to GPU (DeepSpeed will handle it)
    model = train_utils.build_model(config)  # Use existing build_model function

    aux_state = {}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params: {n_params}")

    # Load DeepSpeed config
    with open("flowmo/ds_config.json", "r") as f:
        ds_config = json.load(f)
    
    # Update DeepSpeed config with our settings
    ds_config["train_batch_size"] = config.data.batch_size
    ds_config["gradient_accumulation_steps"] = config.opt.n_grad_acc
    ds_config["optimizer"]["params"]["lr"] = config.opt.lr
    ds_config["optimizer"]["params"]["betas"] = [config.opt.beta1, config.opt.beta2]
    ds_config["optimizer"]["params"]["weight_decay"] = config.opt.weight_decay

    # take the dataset and make it deepspeed compatible 
    train_dataloader = train_utils.load_dataset(config, split='train')
    train_dataset = train_dataloader.dataset


    # Initialize DeepSpeed engine
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        training_data=train_dataset
    )

    if rank == 0:
        writer = SummaryWriter(log_dir)
        wandb.init(project="flowmosemcom",
                   name=f"flowmo-{args.experiment_name}",
                   config=OmegaConf.to_container(config, resolve=True),
                   tags = ["deepspeed", "flowmo","CPU-OFFLOAD"])

    total_steps = 0

    # DeepSpeed checkpoint loading
    if args.resume_from_ckpt:
        _, client_state = model_engine.load_checkpoint(args.resume_from_ckpt)
        total_steps = client_state.get('total_steps', 0)

    model_ema = train_utils.SimpleEMA(model_engine.module, decay=config.model.ema_decay)

    tic = time.time()
    dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

    print("Training begins.")
    print(args)
    print(OmegaConf.to_yaml(config))
    if rank == 0:
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))

    # Initialize norms (same as before)
    with torch.no_grad():
        if config.model.fix_initial_norms:
            if config.model.fix_norm_mode == "channel":
                norm_kwargs = dict(axis=1, keepdims=True)
            elif config.model.fix_norm_mode == "l2":
                norm_kwargs = dict()
            else:
                raise NotImplementedError

            initial_norms = {
                k: torch.linalg.norm(v, **norm_kwargs)
                for (k, v) in models.get_weights_to_fix(model_engine)
            }
            print("Norms checksum", sum(v.sum() for v in initial_norms.values()))

    # LPIPS setup
    if config.opt.lpips_weight != 0.0:
        if config.opt.lpips_mode == "vgg":
            aux_state["lpips_model"] = (
                lpips.LPIPS(net="vgg").eval().requires_grad_(False).cuda()
            )
        elif config.opt.lpips_mode == "resnet":
            aux_state["lpips_model"] = (
                perceptual_loss.PerceptualLoss().eval().requires_grad_(False).cuda()
            )
        else:
            raise NotImplementedError

    running_losses = {}
    aux_state["dl_iter"] = dl_iter

    while total_steps <= config.trainer.max_steps:
        model_engine.train()
        
        dl_tic = time.time()
        batch = next(dl_iter)
        dl_toc = time.time()
        if dl_toc - dl_tic > 1.0:
            print(f"Dataloader took {dl_toc - dl_tic} seconds!")
        images = batch["image"]

        aux_state["total_steps"] = total_steps

        loss, aux = train_step(config, model_engine, batch, aux_state)
        loss_dict = aux["loss_dict"]

        for k, v in loss_dict.items():
            if k in running_losses:
                running_losses[k] += v
            else:
                running_losses[k] = v

        if config.model.fix_initial_norms:
            for name, weight in models.get_weights_to_fix(model_engine):
                weight.data = (
                    weight
                    / torch.linalg.norm(weight, **norm_kwargs)
                    * initial_norms[name]
                )

        model_ema.update(model_engine.module, step=total_steps)
        total_steps += 1

        if total_steps == 1:
            print("first step done!")
            print(images.min(), images.max(), images.mean())

        # Refresh dataloader
        if total_steps % 10_000 == 0:
            train_dataloader = train_utils.load_dataset(config, split='train')
            dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

        if total_steps % config.trainer.log_every == 0:
            toc = time.time()
            torch.cuda.synchronize()

            steps_per_sec = config.trainer.log_every / (toc - tic)
            running_losses = {
                k: (l / config.trainer.log_every).item()
                for (k, l) in running_losses.items()
            }
            reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            allocated_gb = torch.cuda.max_memory_allocated() / 1e9
            log_dict = dict(memory_usage=train_utils.memory_usage(),
                    total_steps=total_steps,
                    steps_per_sec=steps_per_sec,
                    reserved_gb=reserved_gb,
                    allocated_gb=allocated_gb,
                    **running_losses,)

            print(log_dict)
            if rank == 0:
                for k, v in running_losses.items():
                    writer.add_scalar(k, v, global_step=total_steps)
                writer.add_scalar(
                    "Steps per sec", steps_per_sec, global_step=total_steps
                )
                #log on wandB
                wandb.log(log_dict, step=total_steps)

            tic = time.time()
            running_losses = dict()

        # DeepSpeed checkpointing
        if rank == 0 and total_steps % config.trainer.checkpoint_every == 0:
            client_state = {'total_steps': total_steps}
            checkpoint_dir = os.path.join(log_dir, "checkpoints")
            model_engine.save_checkpoint(checkpoint_dir, tag=f"step_{total_steps}", client_state=client_state)
            print(f"Saved checkpoint at step {total_steps}")
            log_gpu_memory(total_steps, label="After Checkpoint")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log_gpu_memory(total_steps, label="After Cleanup")


if __name__ == "__main__":
    try:
        # Use custom config loader that handles DeepSpeed arguments
        args, config = get_args_and_config_deepspeed()
        main(args, config)
    finally:
        if dist.is_initialized():
            torch.distributed.destroy_process_group()
