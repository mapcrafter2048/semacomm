# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/nz/cnz6sodu47e5mqrswyjz6y7cx62vuydwyeebqkndjolroizhugqd.py
# Topologically Sorted Source Nodes: [codebook_value, gt, neg, quantized, gt_1, int_1, int_2, mul, indices], Original ATen: [aten._to_copy, aten.gt, aten.neg, aten.where, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   codebook_value => full_default
#   gt => gt
#   gt_1 => gt_1
#   indices => sum_1
#   int_1 => convert_element_type_1
#   int_2 => convert_element_type_2
#   mul => mul
#   neg => full_default_1
#   quantized => where
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_3, 0), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], -1.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %full_default, %full_default_1), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_1, torch.int32), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.int32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %convert_element_type_2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [3]), kwargs = {})
triton_per_fused__to_copy_gt_mul_neg_sum_where_0 = async_compile.triton('triton_per_fused__to_copy_gt_mul_neg_sum_where_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_gt_mul_neg_sum_where_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_gt_mul_neg_sum_where_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 9
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*r0_1 + 18*(x0 // 2) + ((x0 % 2))), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = -1.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5 > tmp1
    tmp7 = tmp6.to(tl.int32)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/pg/cpgquwkwgwdhesbvpgwzysvm7ziyuehplzjva3uohpts7afwrsqv.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   einsum => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/io/cioxfswbedcqoklpnb7bgicyr34cvhiwdfan5t27onhk53ovrknv.py
# Topologically Sorted Source Nodes: [code, codebook_value, gt, neg, quantized, commit_loss, commit_loss_1], Original ATen: [aten.clone, aten._to_copy, aten.gt, aten.neg, aten.where, aten.mse_loss, aten.mean, aten.mse_loss_backward]
# Source node to ATen node mapping:
#   code => clone
#   codebook_value => full_default
#   commit_loss => convert_element_type_8, convert_element_type_9, pow_1, sub_4
#   commit_loss_1 => mean_2
#   gt => gt
#   neg => full_default_1
#   quantized => where
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_3, 0), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], -1.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %full_default, %full_default_1), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_9, %convert_element_type_8), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, 2.0), kwargs = {})
triton_red_fused__to_copy_clone_gt_mean_mse_loss_mse_loss_backward_neg_where_2 = async_compile.triton('triton_red_fused__to_copy_clone_gt_mean_mse_loss_mse_loss_backward_neg_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_gt_mean_mse_loss_mse_loss_backward_neg_where_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_clone_gt_mean_mse_loss_mse_loss_backward_neg_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 4608
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = (r0_index % 2)
        r0_1 = ((r0_index // 2) % 256)
        r0_2 = r0_index // 512
        r0_4 = r0_index
        r0_3 = (r0_index % 512)
        tmp0 = tl.load(in_ptr0 + (r0_0 + 2*r0_2 + 18*r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (2*r0_2 + 18*(r0_3 // 2) + ((r0_3 % 2))), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = 0.0
        tmp4 = tmp2 > tmp3
        tmp5 = 1.0
        tmp6 = -1.0
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 - tmp8
        tmp10 = 2.0
        tmp11 = tmp9 * tmp10
        tmp12 = tmp9 * tmp9
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tl.store(out_ptr0 + (tl.broadcast_to(r0_4, [XBLOCK, R0_BLOCK])), tmp0, r0_mask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_4, [XBLOCK, R0_BLOCK])), tmp11, r0_mask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)
''', device_str='cuda')


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/wb/cwbsgnvxuflvlxf4z65xl2s2poupxyik56pl6cxslg3m7xfgs7dc.py
# Topologically Sorted Source Nodes: [quantized_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   quantized_5 => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 9)
    x2 = xindex // 18
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2*x2 + 512*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp4 = 1.0
    tmp5 = -1.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp6 - tmp0
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/hv/chvra5xrmj346f2kqbwyajknblbu7p5yebpakaw2vguezcn6jswn.py
# Topologically Sorted Source Nodes: [logits, probs, truediv_1, add, log_probs, mul_3, sum_2], Original ATen: [aten.mul, aten._softmax, aten.div, aten.add, aten._to_copy, aten._log_softmax, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   log_probs => convert_element_type_7, log, sub_2
#   logits => mul_1
#   mul_3 => mul_3
#   probs => div_1, exp, sum_2
#   sum_2 => sum_5
#   truediv_1 => div_2
# Graph fragment:
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 2), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.float32), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, 1), kwargs = {})
#   %amax_default : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 0.1), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, 0.1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, 1e-05), kwargs = {})
#   %convert_element_type_7 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_7, -1), kwargs = {})
#   %sub_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %getitem), kwargs = {})
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_1, %log), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [-1]), kwargs = {dtype: torch.float32})
triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4 = async_compile.triton('triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 7, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 10.0
    tmp4 = tmp2 * tmp3
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp10, 0))
    tmp13 = tmp8 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl_math.log(tmp17)
    tmp19 = tmp2.to(tl.float32)
    tmp20 = 1.0
    tmp21 = tmp19 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp22, 0))
    tmp25 = tmp21 - tmp24
    tmp26 = tmp25 * tmp3
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tl.broadcast_to(tmp27, [R0_BLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = (tmp27 / tmp30)
    tmp32 = tmp7 - tmp12
    tmp33 = tmp32 - tmp18
    tmp34 = tmp31 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [R0_BLOCK])
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
    tl.store(out_ptr2 + (x0), tmp30, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/4u/c4u3qn25xsqgeyfnua2bo73unnpp4tdzibbvnujonupx5n24hsoq.py
# Topologically Sorted Source Nodes: [avg_probs], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   avg_probs => mean
# Graph fragment:
#   %mean : [num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%permute_7, [1, 2, 3]), kwargs = {})
triton_per_fused_mean_5 = async_compile.triton('triton_per_fused_mean_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp8 = 10.0
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = (tmp15 / tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: results/memory_optimized_experiment/torchinductor_cache_0/ew/cewo67cbls3o6pndqngfbei3khtheqshhif4elju3nw5ygcunc4x.py
# Topologically Sorted Source Nodes: [codebook_value, gt, neg, quantized, add_1, log, mul_2, sum_1, avg_entropy, sample_entropy, sample_entropy_1, mul_4, mul_5, loss, commit_loss, commit_loss_1, mul_6, mul_7, quantizer_loss], Original ATen: [aten._to_copy, aten.gt, aten.neg, aten.where, aten.add, aten.log, aten.mul, aten.sum, aten.mean, aten.sub, aten.mse_loss]
# Source node to ATen node mapping:
#   add_1 => add_1
#   avg_entropy => neg_1
#   codebook_value => full_default
#   commit_loss => convert_element_type_8, convert_element_type_9, pow_1, sub_4
#   commit_loss_1 => mean_2
#   gt => gt
#   log => log_1
#   loss => sub_3
#   mul_2 => mul_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   neg => full_default_1
#   quantized => where
#   quantizer_loss => add_3
#   sample_entropy => neg_2
#   sample_entropy_1 => mean_1
#   sum_1 => sum_4
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_3, 0), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], -1.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt, %full_default, %full_default_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, %log_1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {dtype: torch.float32})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_4,), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_5,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%neg_2,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 1.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_9, %convert_element_type_8), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, 0.0025), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_2, 0.000625), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
triton_per_fused__to_copy_add_gt_log_mean_mse_loss_mul_neg_sub_sum_where_6 = async_compile.triton('triton_per_fused__to_copy_add_gt_log_mean_mse_loss_mul_neg_sub_sum_where_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_gt_log_mean_mse_loss_mul_neg_sub_sum_where_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_gt_log_mean_mse_loss_mul_neg_sub_sum_where_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp8 = tl.load(in_ptr1 + (r0_0), None)
    tmp22 = tl.load(in_ptr2 + (0))
    tmp23 = tl.broadcast_to(tmp22, [1])
    tmp1 = 1e-05
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.log(tmp2)
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = -tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = -tmp7
    tmp18 = tmp17 * tmp15
    tmp19 = tmp16 - tmp18
    tmp20 = 0.0025
    tmp21 = tmp19 * tmp20
    tmp24 = 4608.0
    tmp25 = (tmp23 / tmp24)
    tmp26 = 0.000625
    tmp27 = tmp25 * tmp26
    tmp28 = tmp21 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp28, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 256, 18), (4608, 18, 1))
    assert_size_stride(primals_2, (9, ), (1, ))
    assert_size_stride(primals_3, (512, 9), (9, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 512, 1), (512, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [codebook_value, gt, neg, quantized, gt_1, int_1, int_2, mul, indices], Original ATen: [aten._to_copy, aten.gt, aten.neg, aten.where, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_gt_mul_neg_sum_where_0.run(primals_1, primals_2, buf0, 512, 9, stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((512, 9), (9, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_3, buf2, 4608, stream=stream0)
        del primals_3
        buf1 = empty_strided_cuda((1, 9, 256, 2), (4608, 512, 2, 1), torch.bfloat16)
        buf16 = empty_strided_cuda((1, 512, 1, 9), (4608, 1, 4608, 512), torch.float32)
        buf14 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [code, codebook_value, gt, neg, quantized, commit_loss, commit_loss_1], Original ATen: [aten.clone, aten._to_copy, aten.gt, aten.neg, aten.where, aten.mse_loss, aten.mean, aten.mse_loss_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_clone_gt_mean_mse_loss_mse_loss_backward_neg_where_2.run(primals_1, buf1, buf16, buf14, 1, 4608, stream=stream0)
        buf15 = empty_strided_cuda((1, 256, 9, 2), (4608, 18, 2, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [quantized_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf1, primals_1, buf15, 4608, stream=stream0)
        del primals_1
        buf3 = empty_strided_cuda((1, 512, 512), (262144, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (1, 512, 9), (0, 1, 512), 0), reinterpret_tensor(buf2, (1, 9, 512), (0, 1, 9), 0), out=buf3)
        del buf1
        buf7 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf8 = reinterpret_tensor(buf7, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf7  # reuse
        buf6 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 512, 1), (512, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [logits, probs, truediv_1, add, log_probs, mul_3, sum_2], Original ATen: [aten.mul, aten._softmax, aten.div, aten.add, aten._to_copy, aten._log_softmax, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4.run(buf8, buf3, buf6, buf4, buf5, buf12, 512, 512, stream=stream0)
        buf9 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [avg_probs], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_5.run(buf10, buf3, buf4, buf5, 512, 512, stream=stream0)
        buf13 = empty_strided_cuda((), (), torch.float32)
        buf17 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [codebook_value, gt, neg, quantized, add_1, log, mul_2, sum_1, avg_entropy, sample_entropy, sample_entropy_1, mul_4, mul_5, loss, commit_loss, commit_loss_1, mul_6, mul_7, quantizer_loss], Original ATen: [aten._to_copy, aten.gt, aten.neg, aten.where, aten.add, aten.log, aten.mul, aten.sum, aten.mean, aten.sub, aten.mse_loss]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_gt_log_mean_mse_loss_mul_neg_sub_sum_where_6.run(buf17, buf10, buf12, buf14, 1, 512, stream=stream0)
        del buf12
        del buf14
    return (reinterpret_tensor(buf15, (1, 256, 18), (4608, 18, 1), 0), reinterpret_tensor(buf0, (512, ), (1, ), 0), buf17, buf3, buf4, buf5, buf6, buf8, buf10, buf16, reinterpret_tensor(buf2, (1, 512, 9), (9, 9, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 256, 18), (4608, 18, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((512, 9), (9, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
