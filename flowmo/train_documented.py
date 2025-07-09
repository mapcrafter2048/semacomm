"""FlowMo train script.

This script is the main entry point for training the FlowMo model. It handles
everything from setting up the distributed training environment to loading the data,
building the model, running the training loop, and saving checkpoints.

How to Run:
-----------
The script is designed to be run with `torch.distributed.run` for multi-GPU training.

Example command:
```bash
python -m torch.distributed.run --nproc_per_node=8 flowmo/train.py \
    --config_path=configs/your_config.yaml \
    --results_dir=./results \
    --experiment_name=my_flowmo_experiment
```

Arguments:
- `--config_path`: Path to the YAML configuration file. This is the most important
  argument, as it defines the dataset, model, and training parameters.
- `--results_dir`: Directory where training logs and checkpoints will be saved.
- `--experiment_name`: A specific name for this training run. A subdirectory will
  be created under `results_dir` with this name.
- `--resume_from_ckpt` (optional): Path to a checkpoint file to resume training from.

Configuration (`config.yaml`):
---------------------------------
To train on your own dataset, you will need to create a new YAML configuration file
or modify an existing one. Here are the key sections to pay attention to:

1. `dataset`:
   - `name`: The name of your dataset. This should correspond to a dataset loader
     defined in `flowmo/train_utils.py`. You may need to add a new loader for
     your custom dataset.
   - `path`: The absolute path to the root directory of your dataset.
   - `resolution`: The resolution (height and width) to which your images/videos
     will be resized.
   - `n_frames`: The number of frames to be used in each training clip.

2. `model`:
   - This section defines the architecture of the FlowMo model. You can adjust
     hyperparameters like `c_hidden`, `n_stages`, etc., based on your requirements.
   - `ema_decay`: The decay rate for the Exponential Moving Average (EMA) of the
     model weights. EMA helps in stabilizing the training and often leads to
     better performance.

3. `opt` (Optimizer):
   - `lr`: The learning rate.
   - `weight_decay`: The weight decay for regularization.
   - `n_grad_acc`: Number of gradient accumulation steps. This is useful when your
     batch size is limited by GPU memory. The effective batch size will be
     `batch_size * n_grad_acc`.
   - `lpips_weight`: The weight for the LPIPS perceptual loss. A non-zero value
     adds a perceptual loss term, which can improve image quality.

4. `trainer`:
   - `max_steps`: The total number of training steps to run.
   - `log_every`: How often (in steps) to print logs to the console.
   - `checkpoint_every`: How often (in steps) to save a model checkpoint.
   - `gs_checkpoint_bucket`: If you want to save checkpoints to Google Cloud
     Storage, specify the bucket name here. Otherwise, checkpoints are saved locally.

Code Structure:
---------------
- `main(args, config)`: The main function that orchestrates the entire training
  process. It initializes distributed training, sets up logging, builds the model
  and optimizer, loads the dataset, and runs the main training loop.
- `train_step(config, model, batch, optimizer, aux_state)`: Performs a single
  training step. This includes the forward pass, loss calculation (using
  `models.rf_loss`), backpropagation with gradient accumulation, and an optimizer step.
- `_get_norm(model, getter)`: A utility function to calculate the norm of model
  parameters or their gradients.
"""

import contextlib
import glob
import os
import shutil
import time

import fsspec
import lpips
import torch
import torch.distributed as dist
import torch.optim as optim
from mup import MuAdam, MuAdamW
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import models, perceptual_loss, train_utils

# Performance optimizations for PyTorch 2.0+
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

BFLOAT16_IS_AVAILABLE = None


def _get_norm(model, getter):
    """Computes the L2 norm of model parameters or their gradients."""
    return sum(
        (getter(p) ** 2).sum() for p in model.parameters() if p.grad is not None
    ).sqrt()


def train_step(config, model, batch, optimizer, aux_state):
    """
    Performs a single training step, including forward pass, loss calculation,
    backward pass, and optimizer step.

    Args:
        config: The training configuration.
        model: The model to be trained.
        batch: A dictionary containing the input data for the current step.
        optimizer: The optimizer to use for updating the model's weights.
        aux_state: A dictionary for auxiliary state information.

    Returns:
        A tuple containing the total loss and a dictionary with auxiliary data.
    """
    assert BFLOAT16_IS_AVAILABLE is not None
    dtype = torch.bfloat16 if BFLOAT16_IS_AVAILABLE else torch.float32

    aux = {"loss_dict": {}}
    total_loss = 0

    optimizer.zero_grad()
    b = batch["image"].shape[0]
    # Gradient accumulation: split the batch into smaller chunks.
    chunksize = b // config.opt.n_grad_acc
    batch_chunks = [
        {k: v[i * chunksize : (i + 1) * chunksize] for (k, v) in batch.items()}
        for i in range(config.opt.n_grad_acc)
    ]

    total_loss = 0.0
    assert len(batch_chunks) == config.opt.n_grad_acc
    for i, batch_chunk in enumerate(batch_chunks):
        # The `no_sync()` context manager is used to disable gradient synchronization
        # for all but the last gradient accumulation step. This is a performance
        # optimization for distributed training.
        with (
            contextlib.nullcontext()
            if i == config.opt.n_grad_acc - 1
            else model.no_sync()
        ):
            # Automatic mixed-precision training.
            with torch.autocast(
                "cuda",
                dtype=dtype,
            ):
                loss, aux = models.rf_loss(config, model, batch_chunk, aux_state)
                # Normalize the loss by the number of accumulation steps.
                loss = loss / config.opt.n_grad_acc

            loss.backward()
            total_loss += loss.detach()

    if config.opt.log_norms:
        original_grad_norm = _get_norm(model, getter=lambda p: p.grad)
        aux["loss_dict"]["debug/original_grad_norm"] = original_grad_norm
        aux["loss_dict"]["debug/param_norm"] = _get_norm(model, getter=lambda p: p)

    optimizer.step()
    return total_loss, aux


def main(args, config):
    """
    The main function for the training script.
    """
    config = train_utils.restore_config(config)
    print(f"PyTorch version: {torch.__version__}")
    models.MUP_ENABLED = config.model.enable_mup

    # Initialize distributed training environment.
    train_utils.soft_init()

    rank = dist.get_rank()
    print(f"Process rank: {rank}")
    dist.barrier()

    # Set up logging directory.
    log_dir = os.path.join(args.results_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Set up cache directory for TorchInductor (a PyTorch compiler).
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        log_dir, f"torchinductor_cache_{str(rank)}"
    )

    device = rank % torch.cuda.device_count()
    print(f"Using device: {device}, Total CUDA devices: {torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    global BFLOAT16_IS_AVAILABLE
    BFLOAT16_IS_AVAILABLE = (
        train_utils.bfloat16_is_available() and config.trainer.enable_bfloat16
    )
    print(f"Using bfloat16: {BFLOAT16_IS_AVAILABLE}")

    torch.manual_seed(0)

    # Build the model from the configuration.
    model = train_utils.build_model(config)

    aux_state = {}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params / 1e6:.2f}M")

    # Set a unique seed for each process.
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    # Wrap the model with DistributedDataParallel.
    model = DistributedDataParallel(model, find_unused_parameters=True)

    # Select optimizer based on whether mup is enabled.
    if config.model.enable_mup:
        opt_cls = MuAdamW if config.opt.weight_decay else MuAdam
    else:
        opt_cls = optim.AdamW if config.opt.weight_decay else optim.Adam

    # Create parameter groups for the optimizer.
    encoder_pg = {
        "params": [p for (n, p) in model.named_parameters() if "encoder" in n]
    }
    decoder_pg = {
        "params": [p for (n, p) in model.named_parameters() if "decoder" in n]
    }
    assert set(encoder_pg["params"]).union(set(decoder_pg["params"])) == set(
        model.parameters()
    )

    def build_optimizer(pgs):
        optimizer = opt_cls(
            pgs,
            lr=config.opt.lr,
            weight_decay=config.opt.weight_decay,
            betas=(config.opt.beta1, config.opt.beta2),
        )
        return optimizer

    optimizer = build_optimizer([encoder_pg, decoder_pg])
    rebuilt_optimizer = False

    # Load the training dataset.
    train_dataloader = train_utils.load_dataset(config, split='train')

    if rank == 0:
        writer = SummaryWriter(log_dir)

    total_steps = 0

    # Restore from a checkpoint if available.
    latest_ckpt = train_utils.get_last_checkpoint(config, log_dir)
    if latest_ckpt:
        print(f"Resuming from latest checkpoint: {latest_ckpt}")
        total_steps = train_utils.restore_from_ckpt(model.module, optimizer, path=latest_ckpt)
    elif args.resume_from_ckpt:
        print(f"Resuming from user-provided checkpoint: {args.resume_from_ckpt}")
        total_steps = train_utils.restore_from_ckpt(
            model.module, optimizer, path=args.resume_from_ckpt
        )

    # Set up Exponential Moving Average (EMA) for the model.
    model_ema = train_utils.SimpleEMA(model.module, decay=config.model.ema_decay)

    tic = time.time()
    dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

    print("Training begins.")
    print("Arguments:", args)
    print("Configuration:\n", OmegaConf.to_yaml(config))
    if rank == 0:
        # Save the configuration to the log directory.
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))

    # Fix initial norms of weights if specified.
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
                for (k, v) in models.get_weights_to_fix(model)
            }
            print("Norms checksum", sum(v.sum() for v in initial_norms.values()))

    # Initialize LPIPS model for perceptual loss.
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

    # Main training loop.
    while total_steps <= config.trainer.max_steps:
        model.train()
        # Optionally freeze the encoder after a certain number of steps.
        if config.opt.freeze_encoder or total_steps >= config.opt.freeze_encoder_after:
            if not rebuilt_optimizer:
                print(f"Freezing encoder and rebuilding optimizer at step {total_steps}")
                optimizer = build_optimizer([decoder_pg])
                rebuilt_optimizer = True
                model.module.encoder.requires_grad_(False)
                model_ema.decay = config.model.ema_decay

        dl_tic = time.time()
        batch = next(dl_iter)
        dl_toc = time.time()
        if dl_toc - dl_tic > 1.0:
            print(f"Dataloader took {dl_toc - dl_tic:.2f} seconds!")
        images = batch["image"]

        aux_state["total_steps"] = total_steps

        loss, aux = train_step(config, model, batch, optimizer, aux_state)
        loss_dict = aux["loss_dict"]

        # Accumulate running losses for logging.
        for k, v in loss_dict.items():
            if k in running_losses:
                running_losses[k] += v
            else:
                running_losses[k] = v

        # Re-normalize weights if `fix_initial_norms` is enabled.
        if config.model.fix_initial_norms:
            for name, weight in models.get_weights_to_fix(model):
                weight.data = (
                    weight
                    / torch.linalg.norm(weight, **norm_kwargs)
                    * initial_norms[name]
                )

        # Update the EMA model.
        model_ema.update(model.module, step=total_steps)

        total_steps += 1

        if total_steps == 1:
            print("First step completed!")
            print(f"Image stats: min={images.min()}, max={images.max()}, mean={images.mean()}")

        # Refresh dataloader periodically.
        if total_steps % 10_000 == 0:
            train_dataloader = train_utils.load_dataset(config, split='train')
            dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

        # Logging.
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

            with torch.no_grad():
                encoder_checksum = sum(
                    p.mean() for p in model.module.encoder.parameters()
                ).item()
                running_losses["encoder_checksum"] = encoder_checksum

            log_data = dict(
                memory_usage=train_utils.memory_usage(),
                total_steps=total_steps,
                steps_per_sec=f"{steps_per_sec:.2f}",
                reserved_gb=f"{reserved_gb:.2f}",
                allocated_gb=f"{allocated_gb:.2f}",
                **running_losses,
            )
            print(log_data)

            if rank == 0:
                for k, v in running_losses.items():
                    writer.add_scalar(k, v, global_step=total_steps)
                writer.add_scalar(
                    "Steps per sec", steps_per_sec, global_step=total_steps
                )

            tic = time.time()
            running_losses = dict()

        # Checkpointing.
        if rank == 0 and total_steps % config.trainer.checkpoint_every == 0:
            # If a GCS bucket is specified, save checkpoints there.
            if config.trainer.gs_checkpoint_bucket:
                local_checkpoint_dir = os.path.join(log_dir, "checkpoints")
                os.makedirs(local_checkpoint_dir, exist_ok=True)
                local_checkpoint_path = os.path.join(
                    local_checkpoint_dir, f"{total_steps:08d}.pth"
                )

                if not os.path.exists(local_checkpoint_path):
                    torch.save(
                        {
                            "total_steps": total_steps,
                            "model_ema_state_dict": train_utils.cpu_state_dict(
                                model_ema.model
                            ),
                            "model_state_dict": train_utils.cpu_state_dict(
                                model.module
                            ),
                        },
                        local_checkpoint_path,
                    )

                gcs_checkpoint_dir = os.path.join(
                    config.trainer.gs_checkpoint_bucket, f"{log_dir}/checkpoints"
                )
                fs = fsspec.filesystem("gs")

                gcs_checkpoint_path = (
                    f"{gcs_checkpoint_dir}/{os.path.basename(local_checkpoint_path)}"
                )
                if not fs.exists(gcs_checkpoint_path):
                    with (
                        fs.open(gcs_checkpoint_path, "wb") as gcs_file,
                        open(local_checkpoint_path, "rb") as local_file,
                    ):
                        shutil.copyfileobj(local_file, gcs_file)
                os.remove(local_checkpoint_path)

                # Clean up old checkpoints in GCS.
                gcs_checkpoints = sorted(fs.glob(f"{gcs_checkpoint_dir}/*.pth"))
                for ckpt in gcs_checkpoints[:-2]:
                    ckpt_step = os.path.splitext(os.path.basename(ckpt))[0]
                    if (int(ckpt_step) % config.trainer.keep_every) != 0:
                        fs.rm(ckpt)
            else:
                # Save checkpoints locally.
                checkpoint_dir = os.path.join(log_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "%.8d.pth" % total_steps)

                if not os.path.exists(checkpoint_path):
                    torch.save(
                        {
                            "total_steps": total_steps,
                            "model_ema_state_dict": train_utils.cpu_state_dict(
                                model_ema.model
                            ),
                            "model_state_dict": train_utils.cpu_state_dict(
                                model.module
                            ),
                        },
                        checkpoint_path,
                    )

                # Clean up old local checkpoints.
                for checkpoint in sorted(
                    glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                )[:-2]:
                    ckpt_step, _ = os.path.basename(checkpoint).split(".")
                    if (int(ckpt_step) % config.trainer.keep_every) != 0:
                        os.remove(checkpoint)

            print(f"Saved checkpoint at step {total_steps}")
            print(
                dict(
                    reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                    allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
            )


if __name__ == "__main__":
    try:
        args, config = train_utils.get_args_and_config()
        main(args, config)
    finally:
        # Ensure the distributed process group is always destroyed.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
