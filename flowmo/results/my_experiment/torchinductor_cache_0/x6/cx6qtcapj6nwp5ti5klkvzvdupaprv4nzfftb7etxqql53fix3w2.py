# AOT ID: ['14_forward']
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


# kernel path: results/my_experiment/torchinductor_cache_0/jl/cjlhngggtqqarqxkrmha3igvgbpfwi3pvqbkzzz3aytuj3naxvaa.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/ho/cho75mvd43aknkqbq3srrmck7d7m4iyskzreqm37n5yxw6qbhocc.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/6m/c6m3m5v3ft7wyugpnsk4vhg4rnonapuctkv7u2vaf3pmwcagflij.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/gf/cgfjm42laeaoj4cpqfxnca3ef7j3z3hblhuvtvnxmkgslsgra3ma.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_6
#   mean => mean
#   pow_1 => pow_1
# Graph fragment:
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_3 = async_compile.triton('triton_per_fused__to_copy_mean_pow_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6144
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 12)
    x1 = xindex // 12
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x0 + 2304*x1), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/b7/cb73xzbufpgrele2bwga74squ26he4mrjlv2bx5v4oar5mfpdjzf.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean, add, rsqrt], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   add => add
#   float_1 => convert_element_type_6
#   mean => mean
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_poi_fused__to_copy_add_mean_pow_rsqrt_4 = async_compile.triton('triton_poi_fused__to_copy_add_mean_pow_rsqrt_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_pow_rsqrt_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_pow_rsqrt_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 12)
    y1 = yindex // 12
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 12*x2 + 3072*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp5, ymask & xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/js/cjshipzil5uhemv56hfqvcotini7ejzwb2teusbgrm5syzbzepye.py
# Topologically Sorted Source Nodes: [float_2, pow_2, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_8
#   mean_1 => mean_1
#   pow_2 => pow_2
# Graph fragment:
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_5 = async_compile.triton('triton_per_fused__to_copy_mean_pow_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6144
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 12)
    x1 = xindex // 12
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + r0_2 + 64*x0 + 2304*x1), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/qo/cqoukfsb6blnyjyq4bjvpmqy7pdonia6xgav6vqp2rn5lbqaz4xt.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%select_2, %primals_8], 2), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6684672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 4352)
    x0 = (xindex % 64)
    x2 = ((xindex // 278528) % 12)
    x3 = xindex // 3342336
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1536 + x0 + 64*x2 + 2304*(x1) + 589824*x3), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4352, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*x2 + 2304*((-256) + x1) + 9437184*x3), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/iy/ciyaadeonvhviga2vickvuvz6b3qr2jc3k2giby7sqqkdsuwpbmz.py
# Topologically Sorted Source Nodes: [mul_4, mul_5, add_2, mul_6, mul_7, add_3, type_as, type_as_1], Original ATen: [aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   type_as => convert_element_type_14
#   type_as_1 => convert_element_type_15
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %select_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %select_6), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %select_8), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %select_10), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.bfloat16), kwargs = {})
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_mul_7 = async_compile.triton('triton_poi_fused__to_copy_add_mul_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6684672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex // 3342336
    x6 = (xindex % 278528)
    x2 = ((xindex // 64) % 4352)
    x1 = ((xindex // 2) % 32)
    x3 = ((xindex // 278528) % 12)
    x8 = xindex // 278528
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x6 + 557056*x4), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (1 + 2*x6 + 557056*x4), None, eviction_policy='evict_last')
    tmp1 = x2
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (256*x8 + (x2)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tmp1 >= tmp4
    tmp18 = tl.full([1], 4352, tl.int64)
    tmp19 = tmp1 < tmp18
    tmp20 = tl.load(in_ptr4 + (2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp5, tmp16, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp0 * tmp22
    tmp25 = tl.load(in_ptr1 + (1 + 2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp8
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (1 + 2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp5, tmp32, tmp33)
    tmp35 = tl.load(in_ptr4 + (1 + 2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp5, tmp34, tmp35)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp24 * tmp37
    tmp39 = tmp23 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.load(in_ptr1 + (768 + 2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.load(in_ptr5 + (256*x8 + (x2)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl.load(in_ptr6 + (2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp5, tmp49, tmp50)
    tmp52 = tl.load(in_ptr7 + (2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp53 = tl.where(tmp5, tmp51, tmp52)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp0 * tmp54
    tmp56 = tl.load(in_ptr1 + (769 + 2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp57 * tmp43
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tl.load(in_ptr6 + (1 + 2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp60 * tmp61
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp5, tmp63, tmp64)
    tmp66 = tl.load(in_ptr7 + (1 + 2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tl.where(tmp5, tmp65, tmp66)
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp24 * tmp68
    tmp70 = tmp55 + tmp69
    tmp71 = tmp70.to(tl.float32)
    tl.store(out_ptr1 + (x5), tmp40, None)
    tl.store(out_ptr3 + (x5), tmp71, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/o6/co6iymcneotln65kdnxcxv72o3jvrqyzvhofhc7t3un63ra2orlh.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_17, permute_3
# Graph fragment:
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.bfloat16), kwargs = {})
#   %permute_3 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_17, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_8 = async_compile.triton('triton_poi_fused__to_copy_t_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/l6/cl6v55a6pyfrtgegwz23jzmuyuu5ehzzeilahoohuoav5avnxcqm.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 4096)
    x2 = xindex // 3145728
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (16384 + 64*x1 + 278528*(x0 // 64) + 3342336*x2 + ((x0 % 64))), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/wf/cwffkicwj7bae7k2sbxqwovvmbq3yy3ekvksswkbbp5ohbvstaqe.py
# Topologically Sorted Source Nodes: [linear_1, mul_8, add_4, layer_norm, mul_9, add_5, linear_2], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_4 => add_5
#   add_5 => add_7
#   layer_norm => add_6, convert_element_type_20, mul_9, rsqrt_2, sub, var_mean
#   linear_1 => add_4, convert_element_type_16
#   linear_2 => convert_element_type_23
#   mul_8 => mul_8
#   mul_9 => mul_10
# Graph fragment:
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_11, torch.bfloat16), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %convert_element_type_16), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_12, %mul_8), kwargs = {})
#   %convert_element_type_20 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_20, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_20, %getitem_10), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 1.0), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, 0), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_10 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = 768.0
    tmp26 = (tmp24 / tmp25)
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp8 - tmp18
    tmp31 = tmp30 * tmp29
    tmp32 = tmp31 * tmp5
    tmp33 = 0.0
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp7, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask)
    tl.store(out_ptr0 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/of/cofh3eo7ig3opvr3lcyblfj45c7ferfsmkzk5owu733qqkwm3gzf.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_22
# Graph fragment:
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_13, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_11 = async_compile.triton('triton_poi_fused__to_copy_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/mj/cmj2umgtif5vbwcp4ehdyuedr6ophzlnywcd5trn7v5q57pdli3u.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_21
# Graph fragment:
#   %convert_element_type_21 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/ci/cciybxbpf5jjkynwiv4r3ttdckvunbumvrc562loiiuelzjuriwb.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_8, add_9, convert_element_type_27, convert_element_type_28, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, tanh
# Graph fragment:
#   %convert_element_type_27 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_11, torch.float32), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_27, %convert_element_type_27), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %convert_element_type_27), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, 0.044715), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_27, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.7978845608028654), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_27, 0.5), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_14,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %add_9), kwargs = {})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_13 = async_compile.triton('triton_poi_fused_gelu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = 0.044715
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/wp/cwpi7kxufq3nrwziy4vnvman3rkfvwl2kznsmuvkkfx26hvb2nfv.py
# Topologically Sorted Source Nodes: [mul_10, add_6], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_10
#   mul_10 => mul_17
# Graph fragment:
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_17), kwargs = {})
triton_poi_fused_add_mul_14 = async_compile.triton('triton_poi_fused_add_mul_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_14(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tl.store(in_out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/yb/cyb2yahvfl6c44hweyawksxv444ozpddetjb4zchmnm2ugugjuo7.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear_4 => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 256)
    x2 = xindex // 196608
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (64*x1 + 278528*(x0 // 64) + 3342336*x2 + ((x0 % 64))), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/hw/chwxbni6qrrljpuozgq6wpar43j7y4lfkxyk4o4hghsolo2gz6ff.py
# Topologically Sorted Source Nodes: [linear_4, mul_11, add_7, layer_norm_1, mul_12, add_8], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_7 => add_12
#   add_8 => add_14
#   layer_norm_1 => add_13, convert_element_type_38, mul_19, rsqrt_3, sub_1, var_mean_1
#   linear_4 => add_11, convert_element_type_34
#   mul_11 => mul_18
#   mul_12 => mul_20
# Graph fragment:
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_18, torch.bfloat16), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %convert_element_type_34), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 1), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_19, %mul_18), kwargs = {})
#   %convert_element_type_38 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_38, %getitem_12), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_3), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, 1.0), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, 0), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_16 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = 768.0
    tmp26 = (tmp24 / tmp25)
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp8 - tmp18
    tmp31 = tmp30 * tmp29
    tmp32 = tmp31 * tmp5
    tmp33 = 0.0
    tmp34 = tmp32 + tmp33
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp7, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp34, r0_mask)
    tl.store(out_ptr0 + (x0), tmp18, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19 = args
    args.clear()
    assert_size_stride(primals_1, (2, 256, 768), (196608, 768, 1))
    assert_size_stride(primals_2, (2304, 768), (768, 1))
    assert_size_stride(primals_3, (2304, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (2, 12, 4096, 64), (3145728, 64, 768, 1))
    assert_size_stride(primals_7, (2, 12, 4096, 64), (3145728, 64, 768, 1))
    assert_size_stride(primals_8, (2, 12, 4096, 64), (9437184, 64, 2304, 1))
    assert_size_stride(primals_9, (2, 1, 4352, 32, 2, 2), (557056, 557056, 128, 4, 2, 1))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (2, 4096, 768), (3145728, 768, 1))
    assert_size_stride(primals_13, (3072, 768), (768, 1))
    assert_size_stride(primals_14, (3072, ), (1, ))
    assert_size_stride(primals_15, (768, 3072), (3072, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768), (768, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (2, 256, 768), (196608, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_1, buf0, 393216, stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_2, buf1, 1769472, stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((2304, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_3, buf2, 2304, stream=stream0)
        del primals_3
        buf3 = empty_strided_cuda((512, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf2, reinterpret_tensor(buf0, (512, 768), (768, 1), 0), reinterpret_tensor(buf1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf3)
        del buf2
        buf4 = empty_strided_cuda((2, 12, 256, 1), (3072, 1, 12, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_3.run(buf3, buf4, 6144, 64, stream=stream0)
        buf5 = empty_strided_cuda((2, 12, 256, 1), (3072, 256, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean, add, rsqrt], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_4.run(buf4, buf5, 24, 256, stream=stream0)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [float_2, pow_2, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_5.run(buf3, buf6, 6144, 64, stream=stream0)
        buf7 = empty_strided_cuda((2, 12, 256, 1), (3072, 256, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_2, pow_2, mean_1, add_1, rsqrt_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_4.run(buf6, buf7, 24, 256, stream=stream0)
        del buf6
        buf8 = empty_strided_cuda((2, 12, 4352, 64), (3342336, 278528, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf3, primals_8, buf8, 6684672, stream=stream0)
        del primals_8
        buf11 = empty_strided_cuda((2, 12, 4352, 64), (3342336, 278528, 64, 1), torch.bfloat16)
        buf12 = empty_strided_cuda((2, 12, 4352, 64), (3342336, 278528, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul_4, mul_5, add_2, mul_6, mul_7, add_3, type_as, type_as_1], Original ATen: [aten.mul, aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_7.run(primals_9, buf3, buf5, primals_4, primals_6, buf7, primals_5, primals_7, buf11, buf12, 6684672, stream=stream0)
        del primals_6
        del primals_7
        # Topologically Sorted Source Nodes: [scaled_dot_product_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf13 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf11, buf12, buf8, scale=0.125)
        buf14 = buf13[0]
        assert_size_stride(buf14, (2, 12, 4352, 64), (3342336, 278528, 64, 1))
        buf15 = buf13[1]
        assert_size_stride(buf15, (2, 12, 4352), (52224, 4352, 1))
        buf16 = buf13[6]
        assert_size_stride(buf16, (2, ), (1, ))
        buf17 = buf13[7]
        assert_size_stride(buf17, (), ())
        del buf13
        buf19 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_8.run(primals_10, buf19, 589824, stream=stream0)
        del primals_10
        buf20 = empty_strided_cuda((2, 4096, 768), (3145728, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf14, buf20, 6291456, stream=stream0)
        buf21 = empty_strided_cuda((8192, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (8192, 768), (768, 1), 0), buf19, out=buf21)
        buf22 = reinterpret_tensor(buf21, (2, 4096, 768), (3145728, 768, 1), 0); del buf21  # reuse
        buf23 = empty_strided_cuda((2, 4096, 1), (4096, 1, 1), torch.float32)
        buf24 = empty_strided_cuda((2, 4096, 1), (4096, 1, 8192), torch.float32)
        buf26 = reinterpret_tensor(buf24, (2, 4096, 1), (4096, 1, 1), 0); del buf24  # reuse
        buf27 = empty_strided_cuda((2, 4096, 768), (3145728, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1, mul_8, add_4, layer_norm, mul_9, add_5, linear_2], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_10.run(buf22, buf26, primals_12, primals_11, buf23, buf27, 8192, 768, stream=stream0)
        del primals_11
        del primals_12
        buf28 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_13, buf28, 2359296, stream=stream0)
        del primals_13
        buf29 = empty_strided_cuda((3072, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_14, buf29, 3072, stream=stream0)
        del primals_14
        buf30 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf29, reinterpret_tensor(buf27, (8192, 768), (768, 1), 0), reinterpret_tensor(buf28, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf30)
        del buf29
        buf31 = empty_strided_cuda((2, 4096, 3072), (12582912, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_13.run(buf30, buf31, 25165824, stream=stream0)
        buf32 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_15, buf32, 2359296, stream=stream0)
        del primals_15
        buf33 = empty_strided_cuda((8192, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf31, (8192, 3072), (3072, 1), 0), buf32, out=buf33)
        buf34 = reinterpret_tensor(buf33, (2, 4096, 768), (3145728, 768, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [mul_10, add_6], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_14.run(buf34, buf22, primals_16, 6291456, stream=stream0)
        del primals_16
        buf35 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_8.run(primals_17, buf35, 589824, stream=stream0)
        del primals_17
        buf36 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf14, buf36, 393216, stream=stream0)
        buf37 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 768), (768, 1), 0), buf35, out=buf37)
        buf38 = reinterpret_tensor(buf37, (2, 256, 768), (196608, 768, 1), 0); del buf37  # reuse
        buf39 = empty_strided_cuda((2, 256, 1), (256, 1, 1), torch.float32)
        buf40 = empty_strided_cuda((2, 256, 1), (256, 1, 512), torch.float32)
        buf42 = reinterpret_tensor(buf40, (2, 256, 1), (256, 1, 1), 0); del buf40  # reuse
        buf43 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4, mul_11, add_7, layer_norm_1, mul_12, add_8], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_16.run(buf38, buf42, primals_19, primals_18, buf39, buf43, 512, 768, stream=stream0)
        del primals_18
        del primals_19
    return (buf43, buf38, buf34, primals_4, primals_5, primals_9, reinterpret_tensor(buf0, (512, 768), (768, 1), 0), reinterpret_tensor(buf3, (2, 12, 256, 64), (589824, 64, 2304, 1), 0), reinterpret_tensor(buf3, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), buf5, buf7, buf8, buf11, buf12, buf14, buf15, buf16, buf17, reinterpret_tensor(buf20, (8192, 768), (768, 1), 0), buf22, buf23, buf26, reinterpret_tensor(buf27, (8192, 768), (768, 1), 0), buf30, reinterpret_tensor(buf31, (8192, 3072), (3072, 1), 0), reinterpret_tensor(buf36, (512, 768), (768, 1), 0), buf38, buf39, buf42, reinterpret_tensor(buf35, (768, 768), (768, 1), 0), reinterpret_tensor(buf32, (768, 3072), (3072, 1), 0), buf28, reinterpret_tensor(buf19, (768, 768), (768, 1), 0), buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 256, 768), (196608, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, 12, 4096, 64), (3145728, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((2, 12, 4096, 64), (3145728, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((2, 12, 4096, 64), (9437184, 64, 2304, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((2, 1, 4352, 32, 2, 2), (557056, 557056, 128, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2, 4096, 768), (3145728, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, 256, 768), (196608, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
