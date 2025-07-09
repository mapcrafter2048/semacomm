# AOT ID: ['1_forward']
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


# kernel path: results/my_experiment/torchinductor_cache_0/bk/cbkptwgyiqiydurf4dhpnvo26gr3bcrd2pzslcmugjnollyfe2sa.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/m4/cm4vqyj56kbzejvqn7j3cw6unhzavubrcgk6byja52ou62qmriih.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_31
# Graph fragment:
#   %convert_element_type_31 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/v4/cv4nj3sjgzkh23fmx7mnccp47zgb4jp76anhmezhsbq37i3nayay.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   linear_1 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([512, 18], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_view_2 = async_compile.triton('triton_poi_fused_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/rf/crf5jn3wxmnc4wp5hzessiptp2pihpqh22os7holrnxtd5twidlb.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/lk/clktnesrueopqya6qxw2db77jiqxbm4mogjx5xw5cp632jwrsxv3.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/yy/cyypxint2bbcmytj3zhf2boktamkgkjbjxdrdktu2kgyjpttyovg.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 48)
    x1 = ((xindex // 48) % 4096)
    x2 = xindex // 196608
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (12*((x1 % 64)) + 768*(x0 // 12) + 3072*(x1 // 64) + 196608*x2 + ((x0 % 12))), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/5k/c5ksnsykpaqph6lyxedicxsxkle34pt6isggvpwy7b2k5oy4jegz.py
# Topologically Sorted Source Nodes: [stack_3], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_3 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_25, %unsqueeze_26, %unsqueeze_27, %unsqueeze_25], -1), kwargs = {})
triton_poi_fused_stack_6 = async_compile.triton('triton_poi_fused_stack_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 139264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 4352)
    x3 = xindex // 69632
    x1 = ((xindex // 4) % 4)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x2
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (3*(x2) + 768*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 4352, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (3*((-256) + x2)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float64)
    tmp20 = 2*x1
    tmp21 = tmp20.to(tl.float64)
    tmp22 = tl.full([1], 0.125, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full([1], 10000.0, tl.float64)
    tmp25 = libdevice.pow(tmp24, tmp23)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = (tmp26 / tmp25)
    tmp28 = tl.full([1], 1.0, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp19 * tmp29
    tmp31 = libdevice.cos(tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 2, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = x2
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp38 >= tmp39
    tmp41 = tl.full([1], 256, tl.int64)
    tmp42 = tmp38 < tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tl.load(in_ptr0 + (3*(x2) + 768*x3), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp38 >= tmp41
    tmp46 = tl.full([1], 4352, tl.int64)
    tmp47 = tmp38 < tmp46
    tmp48 = tmp45 & tmp37
    tmp49 = tl.load(in_ptr1 + (3*((-256) + x2)), tmp48, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp42, tmp44, tmp49)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51.to(tl.float64)
    tmp53 = 2*x1
    tmp54 = tmp53.to(tl.float64)
    tmp55 = tl.full([1], 0.125, tl.float64)
    tmp56 = tmp54 * tmp55
    tmp57 = tl.full([1], 10000.0, tl.float64)
    tmp58 = libdevice.pow(tmp57, tmp56)
    tmp59 = tl.full([1], 1, tl.int32)
    tmp60 = (tmp59 / tmp58)
    tmp61 = tl.full([1], 1.0, tl.float64)
    tmp62 = tmp60 * tmp61
    tmp63 = tmp52 * tmp62
    tmp64 = libdevice.sin(tmp63)
    tmp65 = -tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp37, tmp65, tmp66)
    tmp68 = tmp0 >= tmp35
    tmp69 = tl.full([1], 3, tl.int64)
    tmp70 = tmp0 < tmp69
    tmp71 = tmp68 & tmp70
    tmp72 = x2
    tmp73 = tl.full([1], 0, tl.int64)
    tmp74 = tmp72 >= tmp73
    tmp75 = tl.full([1], 256, tl.int64)
    tmp76 = tmp72 < tmp75
    tmp77 = tmp76 & tmp71
    tmp78 = tl.load(in_ptr0 + (3*(x2) + 768*x3), tmp77, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp72 >= tmp75
    tmp80 = tl.full([1], 4352, tl.int64)
    tmp81 = tmp72 < tmp80
    tmp82 = tmp79 & tmp71
    tmp83 = tl.load(in_ptr1 + (3*((-256) + x2)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.where(tmp76, tmp78, tmp83)
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp85.to(tl.float64)
    tmp87 = 2*x1
    tmp88 = tmp87.to(tl.float64)
    tmp89 = tl.full([1], 0.125, tl.float64)
    tmp90 = tmp88 * tmp89
    tmp91 = tl.full([1], 10000.0, tl.float64)
    tmp92 = libdevice.pow(tmp91, tmp90)
    tmp93 = tl.full([1], 1, tl.int32)
    tmp94 = (tmp93 / tmp92)
    tmp95 = tl.full([1], 1.0, tl.float64)
    tmp96 = tmp94 * tmp95
    tmp97 = tmp86 * tmp96
    tmp98 = libdevice.sin(tmp97)
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp71, tmp98, tmp99)
    tmp101 = tmp0 >= tmp69
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp0 < tmp102
    tmp104 = x2
    tmp105 = tl.full([1], 0, tl.int64)
    tmp106 = tmp104 >= tmp105
    tmp107 = tl.full([1], 256, tl.int64)
    tmp108 = tmp104 < tmp107
    tmp109 = tmp108 & tmp101
    tmp110 = tl.load(in_ptr0 + (3*(x2) + 768*x3), tmp109, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp104 >= tmp107
    tmp112 = tl.full([1], 4352, tl.int64)
    tmp113 = tmp104 < tmp112
    tmp114 = tmp111 & tmp101
    tmp115 = tl.load(in_ptr1 + (3*((-256) + x2)), tmp114, eviction_policy='evict_last', other=0.0)
    tmp116 = tl.where(tmp108, tmp110, tmp115)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tmp117.to(tl.float64)
    tmp119 = 2*x1
    tmp120 = tmp119.to(tl.float64)
    tmp121 = tl.full([1], 0.125, tl.float64)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.full([1], 10000.0, tl.float64)
    tmp124 = libdevice.pow(tmp123, tmp122)
    tmp125 = tl.full([1], 1, tl.int32)
    tmp126 = (tmp125 / tmp124)
    tmp127 = tl.full([1], 1.0, tl.float64)
    tmp128 = tmp126 * tmp127
    tmp129 = tmp118 * tmp128
    tmp130 = libdevice.cos(tmp129)
    tmp131 = tl.full(tmp130.shape, 0.0, tmp130.dtype)
    tmp132 = tl.where(tmp101, tmp130, tmp131)
    tmp133 = tl.where(tmp71, tmp100, tmp132)
    tmp134 = tl.where(tmp37, tmp67, tmp133)
    tmp135 = tl.where(tmp4, tmp33, tmp134)
    tl.store(out_ptr0 + (x6), tmp135, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/nt/cntn7dontywatle3qdac5xbvrzxspphrtzdxeykxipim5ocfyhyc.py
# Topologically Sorted Source Nodes: [layer_norm_1, mul_5, add_3], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.mul, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_3 => add_11
#   layer_norm_1 => add_10, convert_element_type_43, mul_24, rsqrt_3, sub_1, var_mean_1
#   mul_5 => mul_25
# Graph fragment:
#   %convert_element_type_43 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_43, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_43, %getitem_3), kwargs = {})
#   %mul_24 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_3), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, 1.0), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, 0), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 768), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_7(in_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel):
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
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.full([1], 768, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp1 - tmp11
    tmp19 = 768.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = 0.0
    tmp28 = tmp26 + tmp27
    tmp29 = 0.0013020833333333333
    tmp30 = tmp23 * tmp29
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp24, r0_mask)
    tl.store(out_ptr3 + (r0_1 + 768*x0), tmp28, r0_mask)
    tl.store(out_ptr4 + (x0), tmp30, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/uz/cuzl2bb7im7wvy7j7f3336vmxgmcs6t2hqx3vumqi4niy5woxkud.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_32
# Graph fragment:
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/ev/cev24pnmyjqjgajuijbwtg67izyimqb23w4ehv7bg5wnopo7r4rk.py
# Topologically Sorted Source Nodes: [stack_4], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_4 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_32, %unsqueeze_33, %unsqueeze_34, %unsqueeze_32], -1), kwargs = {})
triton_poi_fused_stack_9 = async_compile.triton('triton_poi_fused_stack_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 487424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x2 = ((xindex // 56) % 4352)
    x3 = xindex // 243712
    x1 = ((xindex // 4) % 14)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x2
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (1 + 3*(x2) + 768*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 4352, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (1 + 3*((-256) + x2)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float64)
    tmp20 = 2*x1
    tmp21 = tmp20.to(tl.float64)
    tmp22 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full([1], 10000.0, tl.float64)
    tmp25 = libdevice.pow(tmp24, tmp23)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = (tmp26 / tmp25)
    tmp28 = tl.full([1], 1.0, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp19 * tmp29
    tmp31 = libdevice.cos(tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 2, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = x2
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp38 >= tmp39
    tmp41 = tl.full([1], 256, tl.int64)
    tmp42 = tmp38 < tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tl.load(in_ptr0 + (1 + 3*(x2) + 768*x3), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp38 >= tmp41
    tmp46 = tl.full([1], 4352, tl.int64)
    tmp47 = tmp38 < tmp46
    tmp48 = tmp45 & tmp37
    tmp49 = tl.load(in_ptr1 + (1 + 3*((-256) + x2)), tmp48, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp42, tmp44, tmp49)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51.to(tl.float64)
    tmp53 = 2*x1
    tmp54 = tmp53.to(tl.float64)
    tmp55 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp56 = tmp54 * tmp55
    tmp57 = tl.full([1], 10000.0, tl.float64)
    tmp58 = libdevice.pow(tmp57, tmp56)
    tmp59 = tl.full([1], 1, tl.int32)
    tmp60 = (tmp59 / tmp58)
    tmp61 = tl.full([1], 1.0, tl.float64)
    tmp62 = tmp60 * tmp61
    tmp63 = tmp52 * tmp62
    tmp64 = libdevice.sin(tmp63)
    tmp65 = -tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp37, tmp65, tmp66)
    tmp68 = tmp0 >= tmp35
    tmp69 = tl.full([1], 3, tl.int64)
    tmp70 = tmp0 < tmp69
    tmp71 = tmp68 & tmp70
    tmp72 = x2
    tmp73 = tl.full([1], 0, tl.int64)
    tmp74 = tmp72 >= tmp73
    tmp75 = tl.full([1], 256, tl.int64)
    tmp76 = tmp72 < tmp75
    tmp77 = tmp76 & tmp71
    tmp78 = tl.load(in_ptr0 + (1 + 3*(x2) + 768*x3), tmp77, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp72 >= tmp75
    tmp80 = tl.full([1], 4352, tl.int64)
    tmp81 = tmp72 < tmp80
    tmp82 = tmp79 & tmp71
    tmp83 = tl.load(in_ptr1 + (1 + 3*((-256) + x2)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.where(tmp76, tmp78, tmp83)
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp85.to(tl.float64)
    tmp87 = 2*x1
    tmp88 = tmp87.to(tl.float64)
    tmp89 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp90 = tmp88 * tmp89
    tmp91 = tl.full([1], 10000.0, tl.float64)
    tmp92 = libdevice.pow(tmp91, tmp90)
    tmp93 = tl.full([1], 1, tl.int32)
    tmp94 = (tmp93 / tmp92)
    tmp95 = tl.full([1], 1.0, tl.float64)
    tmp96 = tmp94 * tmp95
    tmp97 = tmp86 * tmp96
    tmp98 = libdevice.sin(tmp97)
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp71, tmp98, tmp99)
    tmp101 = tmp0 >= tmp69
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp0 < tmp102
    tmp104 = x2
    tmp105 = tl.full([1], 0, tl.int64)
    tmp106 = tmp104 >= tmp105
    tmp107 = tl.full([1], 256, tl.int64)
    tmp108 = tmp104 < tmp107
    tmp109 = tmp108 & tmp101
    tmp110 = tl.load(in_ptr0 + (1 + 3*(x2) + 768*x3), tmp109, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp104 >= tmp107
    tmp112 = tl.full([1], 4352, tl.int64)
    tmp113 = tmp104 < tmp112
    tmp114 = tmp111 & tmp101
    tmp115 = tl.load(in_ptr1 + (1 + 3*((-256) + x2)), tmp114, eviction_policy='evict_last', other=0.0)
    tmp116 = tl.where(tmp108, tmp110, tmp115)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tmp117.to(tl.float64)
    tmp119 = 2*x1
    tmp120 = tmp119.to(tl.float64)
    tmp121 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.full([1], 10000.0, tl.float64)
    tmp124 = libdevice.pow(tmp123, tmp122)
    tmp125 = tl.full([1], 1, tl.int32)
    tmp126 = (tmp125 / tmp124)
    tmp127 = tl.full([1], 1.0, tl.float64)
    tmp128 = tmp126 * tmp127
    tmp129 = tmp118 * tmp128
    tmp130 = libdevice.cos(tmp129)
    tmp131 = tl.full(tmp130.shape, 0.0, tmp130.dtype)
    tmp132 = tl.where(tmp101, tmp130, tmp131)
    tmp133 = tl.where(tmp71, tmp100, tmp132)
    tmp134 = tl.where(tmp37, tmp67, tmp133)
    tmp135 = tl.where(tmp4, tmp33, tmp134)
    tl.store(out_ptr0 + (x6), tmp135, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/5k/c5kemjeec3y2bkzlehgukx5rnkgsswsgdb6gz4v7yuosladlyk2x.py
# Topologically Sorted Source Nodes: [stack_5], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_5 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_39, %unsqueeze_40, %unsqueeze_41, %unsqueeze_39], -1), kwargs = {})
triton_poi_fused_stack_10 = async_compile.triton('triton_poi_fused_stack_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 487424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x2 = ((xindex // 56) % 4352)
    x3 = xindex // 243712
    x1 = ((xindex // 4) % 14)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x2
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (2 + 3*(x2) + 768*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 4352, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (2 + 3*((-256) + x2)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float64)
    tmp20 = 2*x1
    tmp21 = tmp20.to(tl.float64)
    tmp22 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full([1], 10000.0, tl.float64)
    tmp25 = libdevice.pow(tmp24, tmp23)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = (tmp26 / tmp25)
    tmp28 = tl.full([1], 1.0, tl.float64)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp19 * tmp29
    tmp31 = libdevice.cos(tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 2, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = x2
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp38 >= tmp39
    tmp41 = tl.full([1], 256, tl.int64)
    tmp42 = tmp38 < tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tl.load(in_ptr0 + (2 + 3*(x2) + 768*x3), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp38 >= tmp41
    tmp46 = tl.full([1], 4352, tl.int64)
    tmp47 = tmp38 < tmp46
    tmp48 = tmp45 & tmp37
    tmp49 = tl.load(in_ptr1 + (2 + 3*((-256) + x2)), tmp48, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp42, tmp44, tmp49)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51.to(tl.float64)
    tmp53 = 2*x1
    tmp54 = tmp53.to(tl.float64)
    tmp55 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp56 = tmp54 * tmp55
    tmp57 = tl.full([1], 10000.0, tl.float64)
    tmp58 = libdevice.pow(tmp57, tmp56)
    tmp59 = tl.full([1], 1, tl.int32)
    tmp60 = (tmp59 / tmp58)
    tmp61 = tl.full([1], 1.0, tl.float64)
    tmp62 = tmp60 * tmp61
    tmp63 = tmp52 * tmp62
    tmp64 = libdevice.sin(tmp63)
    tmp65 = -tmp64
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp37, tmp65, tmp66)
    tmp68 = tmp0 >= tmp35
    tmp69 = tl.full([1], 3, tl.int64)
    tmp70 = tmp0 < tmp69
    tmp71 = tmp68 & tmp70
    tmp72 = x2
    tmp73 = tl.full([1], 0, tl.int64)
    tmp74 = tmp72 >= tmp73
    tmp75 = tl.full([1], 256, tl.int64)
    tmp76 = tmp72 < tmp75
    tmp77 = tmp76 & tmp71
    tmp78 = tl.load(in_ptr0 + (2 + 3*(x2) + 768*x3), tmp77, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp72 >= tmp75
    tmp80 = tl.full([1], 4352, tl.int64)
    tmp81 = tmp72 < tmp80
    tmp82 = tmp79 & tmp71
    tmp83 = tl.load(in_ptr1 + (2 + 3*((-256) + x2)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.where(tmp76, tmp78, tmp83)
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp85.to(tl.float64)
    tmp87 = 2*x1
    tmp88 = tmp87.to(tl.float64)
    tmp89 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp90 = tmp88 * tmp89
    tmp91 = tl.full([1], 10000.0, tl.float64)
    tmp92 = libdevice.pow(tmp91, tmp90)
    tmp93 = tl.full([1], 1, tl.int32)
    tmp94 = (tmp93 / tmp92)
    tmp95 = tl.full([1], 1.0, tl.float64)
    tmp96 = tmp94 * tmp95
    tmp97 = tmp86 * tmp96
    tmp98 = libdevice.sin(tmp97)
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp71, tmp98, tmp99)
    tmp101 = tmp0 >= tmp69
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp0 < tmp102
    tmp104 = x2
    tmp105 = tl.full([1], 0, tl.int64)
    tmp106 = tmp104 >= tmp105
    tmp107 = tl.full([1], 256, tl.int64)
    tmp108 = tmp104 < tmp107
    tmp109 = tmp108 & tmp101
    tmp110 = tl.load(in_ptr0 + (2 + 3*(x2) + 768*x3), tmp109, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp104 >= tmp107
    tmp112 = tl.full([1], 4352, tl.int64)
    tmp113 = tmp104 < tmp112
    tmp114 = tmp111 & tmp101
    tmp115 = tl.load(in_ptr1 + (2 + 3*((-256) + x2)), tmp114, eviction_policy='evict_last', other=0.0)
    tmp116 = tl.where(tmp108, tmp110, tmp115)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tmp117.to(tl.float64)
    tmp119 = 2*x1
    tmp120 = tmp119.to(tl.float64)
    tmp121 = tl.full([1], 0.03571428571428571, tl.float64)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.full([1], 10000.0, tl.float64)
    tmp124 = libdevice.pow(tmp123, tmp122)
    tmp125 = tl.full([1], 1, tl.int32)
    tmp126 = (tmp125 / tmp124)
    tmp127 = tl.full([1], 1.0, tl.float64)
    tmp128 = tmp126 * tmp127
    tmp129 = tmp118 * tmp128
    tmp130 = libdevice.cos(tmp129)
    tmp131 = tl.full(tmp130.shape, 0.0, tmp130.dtype)
    tmp132 = tl.where(tmp101, tmp130, tmp131)
    tmp133 = tl.where(tmp71, tmp100, tmp132)
    tmp134 = tl.where(tmp37, tmp67, tmp133)
    tmp135 = tl.where(tmp4, tmp33, tmp134)
    tl.store(out_ptr0 + (x6), tmp135, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/qx/cqxo43dcqhooszq2srvoezajlsfbp44flpwu5x2w3q5xrs3r3sfm.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_23, %convert_element_type_26, %convert_element_type_29], -3), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1114112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 32)
    x0 = (xindex % 4)
    x2 = xindex // 128
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 16*x2), tmp4, other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 18, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr1 + (x0 + 4*((-4) + x1) + 56*x2), tmp12, other=0.0)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tmp0 >= tmp10
    tmp18 = tl.full([1], 32, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr2 + (x0 + 4*((-18) + x1) + 56*x2), tmp17, other=0.0)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp17, tmp21, tmp22)
    tmp24 = tl.where(tmp12, tmp16, tmp23)
    tmp25 = tl.where(tmp4, tmp8, tmp24)
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/lo/clo6dvttlumzhd2g6jdbv7djlktt6qkjuzp6cgxvv2xctcbohrpz.py
# Topologically Sorted Source Nodes: [layer_norm, mul, add, linear_2], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add_7
#   layer_norm => add_6, convert_element_type_30, mul_18, rsqrt, sub, var_mean
#   linear_2 => convert_element_type_33
#   mul => mul_19
# Graph fragment:
#   %convert_element_type_30 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_30, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_30, %getitem_1), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, 1.0), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, 0), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_12 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_12(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel):
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
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.full([1], 768, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = 768.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp1 - tmp11
    tmp24 = tmp23 * tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = 0.0
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp29, r0_mask)
    tl.store(out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/ty/ctya6aczap63mkq4wstp77w6iuuxcymxwvp6jrvsc2uibqsvbmrf.py
# Topologically Sorted Source Nodes: [float_7, pow_7, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_7 => convert_element_type_37
#   mean => mean
#   pow_7 => pow_7
# Graph fragment:
#   %convert_element_type_37 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_6, torch.float32), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_37, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_13 = async_compile.triton('triton_per_fused__to_copy_mean_pow_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_13(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 98304
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 12)
    x1 = xindex // 12
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x0 + 2304*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/mw/cmwm4kicj7tjyyjbnw7cxp3ihh5ufsdkf6cllsq4da6nxolxyxqn.py
# Topologically Sorted Source Nodes: [float_7, pow_7, mean, add_1, rsqrt], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   add_1 => add_8
#   float_7 => convert_element_type_37
#   mean => mean
#   pow_7 => pow_7
#   rsqrt => rsqrt_1
# Graph fragment:
#   %convert_element_type_37 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_6, torch.float32), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_37, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
triton_poi_fused__to_copy_add_mean_pow_rsqrt_14 = async_compile.triton('triton_poi_fused__to_copy_add_mean_pow_rsqrt_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_pow_rsqrt_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_pow_rsqrt_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 12)
    y1 = yindex // 12
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 12*x2 + 49152*y1), ymask, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp5, ymask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/gp/cgp6xd44d736trxjtcfgc76gpmvjqqhsbcousni2zfzaxzdgru26.py
# Topologically Sorted Source Nodes: [float_8, pow_8, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_8 => convert_element_type_39
#   mean_1 => mean_1
#   pow_8 => pow_8
# Graph fragment:
#   %convert_element_type_39 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_7, torch.float32), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_39, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_15 = async_compile.triton('triton_per_fused__to_copy_mean_pow_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_15(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 98304
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 12)
    x1 = xindex // 12
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + r0_2 + 64*x0 + 2304*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/wc/cwcpqsq3xnuvzukp2vh3hzo65c72ltoaqmkrhmjy2ylvt2rh7g3g.py
# Topologically Sorted Source Nodes: [float_7, mul_1, to, mul_2, to_2], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   float_7 => convert_element_type_37
#   mul_1 => mul_20
#   mul_2 => mul_21
#   to => convert_element_type_38
#   to_2 => convert_element_type_41
# Graph fragment:
#   %convert_element_type_37 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_6, torch.float32), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_37, %rsqrt_1), kwargs = {})
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_38, %primals_10), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_mul_16 = async_compile.triton('triton_poi_fused__to_copy_mul_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = (xindex % 768)
    x5 = xindex // 768
    x1 = ((xindex // 64) % 12)
    x2 = ((xindex // 768) % 4096)
    x3 = xindex // 3145728
    x0 = (xindex % 64)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 2304*x5), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 4096*x1 + 49152*x3), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x6), tmp8, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/hy/chywlf67jg6oathfpxfsthnbj4wrhvcxfgyq74ragxousvwknzmy.py
# Topologically Sorted Source Nodes: [float_8, mul_3, to_1, mul_4, to_3], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   float_8 => convert_element_type_39
#   mul_3 => mul_22
#   mul_4 => mul_23
#   to_1 => convert_element_type_40
#   to_3 => convert_element_type_42
# Graph fragment:
#   %convert_element_type_39 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_7, torch.float32), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_39, %rsqrt_2), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_40, %primals_11), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_mul_17 = async_compile.triton('triton_poi_fused__to_copy_mul_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = (xindex % 768)
    x5 = xindex // 768
    x1 = ((xindex // 64) % 12)
    x2 = ((xindex // 768) % 4096)
    x3 = xindex // 3145728
    x0 = (xindex % 64)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x4 + 2304*x5), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + 4096*x1 + 49152*x3), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x6), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (2, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(primals_2, (768, 48), (48, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, 18), (18, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (2, 256, 3), (768, 3, 1))
    assert_size_stride(primals_7, (2, 4096, 3), (0, 3, 1))
    assert_size_stride(primals_8, (2304, 768), (768, 1))
    assert_size_stride(primals_9, (2304, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((768, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_3, buf2, 768, stream=stream0)
        del primals_3
        buf6 = empty_strided_cuda((768, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_5, buf6, 768, stream=stream0)
        del primals_5
        buf18 = empty_strided_cuda((2304, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_9, buf18, 2304, stream=stream0)
        del primals_9
        buf4 = empty_strided_cuda((512, 18), (18, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf4, 9216, stream=stream0)
        buf5 = empty_strided_cuda((768, 18), (18, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_4, buf5, 13824, stream=stream0)
        del primals_4
        buf1 = empty_strided_cuda((768, 48), (48, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(primals_2, buf1, 36864, stream=stream0)
        del primals_2
        buf7 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf6, buf4, reinterpret_tensor(buf5, (18, 768), (1, 18), 0), alpha=1, beta=1, out=buf7)
        del buf5
        del buf6
        buf0 = empty_strided_cuda((2, 4096, 48), (196608, 48, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(primals_1, buf0, 393216, stream=stream0)
        del primals_1
        buf8 = empty_strided_cuda((2, 4352, 4, 4), (69632, 16, 4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [stack_3], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_6.run(primals_6, primals_7, buf8, 139264, stream=stream0)
        buf29 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.float32)
        buf30 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.float32)
        buf31 = empty_strided_cuda((2, 256, 1), (256, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_1, mul_5, add_3], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.mul, aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_7.run(buf7, buf29, buf30, buf31, 512, 768, stream=stream0)
        buf17 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_8, buf17, 1769472, stream=stream0)
        del primals_8
        buf9 = empty_strided_cuda((2, 4352, 14, 4), (243712, 56, 4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [stack_4], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_9.run(primals_6, primals_7, buf9, 487424, stream=stream0)
        buf10 = empty_strided_cuda((2, 4352, 14, 4), (243712, 56, 4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [stack_5], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_10.run(primals_6, primals_7, buf10, 487424, stream=stream0)
        del primals_6
        del primals_7
        buf11 = empty_strided_cuda((2, 4352, 32, 2, 2), (557056, 128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf8, buf9, buf10, buf11, 1114112, stream=stream0)
        del buf10
        del buf8
        del buf9
        buf3 = empty_strided_cuda((8192, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf2, reinterpret_tensor(buf0, (8192, 48), (48, 1), 0), reinterpret_tensor(buf1, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf3)
        del buf1
        del buf2
        buf12 = empty_strided_cuda((2, 4096, 1), (4096, 1, 1), torch.float32)
        buf13 = empty_strided_cuda((2, 4096, 1), (4096, 1, 8192), torch.float32)
        buf15 = reinterpret_tensor(buf13, (2, 4096, 1), (4096, 1, 1), 0); del buf13  # reuse
        buf16 = empty_strided_cuda((2, 4096, 768), (3145728, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm, mul, add, linear_2], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_12.run(buf15, buf3, buf12, buf16, 8192, 768, stream=stream0)
        buf19 = empty_strided_cuda((8192, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf18, reinterpret_tensor(buf16, (8192, 768), (768, 1), 0), reinterpret_tensor(buf17, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf19)
        del buf18
        buf20 = empty_strided_cuda((2, 12, 4096, 1), (49152, 1, 12, 98304), torch.float32)
        # Topologically Sorted Source Nodes: [float_7, pow_7, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_13.run(buf19, buf20, 98304, 64, stream=stream0)
        buf21 = empty_strided_cuda((2, 12, 4096, 1), (49152, 4096, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_7, pow_7, mean, add_1, rsqrt], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_14.run(buf20, buf21, 24, 4096, stream=stream0)
        buf22 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [float_8, pow_8, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_15.run(buf19, buf22, 98304, 64, stream=stream0)
        buf23 = empty_strided_cuda((2, 12, 4096, 1), (49152, 4096, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_8, pow_8, mean_1, add_2, rsqrt_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_14.run(buf22, buf23, 24, 4096, stream=stream0)
        del buf22
        buf24 = empty_strided_cuda((2, 12, 4096, 64), (3145728, 64, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_7, mul_1, to, mul_2, to_2], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_16.run(buf19, buf21, primals_10, buf24, 6291456, stream=stream0)
        buf25 = empty_strided_cuda((2, 12, 4096, 64), (3145728, 64, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_8, mul_3, to_1, mul_4, to_3], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_17.run(buf19, buf23, primals_11, buf25, 6291456, stream=stream0)
    return (buf30, buf24, buf25, reinterpret_tensor(buf19, (2, 12, 4096, 64), (9437184, 64, 2304, 1), 1536), reinterpret_tensor(buf11, (2, 1, 4352, 32, 2, 2), (557056, 557056, 128, 4, 2, 1), 0), reinterpret_tensor(buf3, (2, 4096, 768), (3145728, 768, 1), 0), reinterpret_tensor(buf7, (2, 256, 768), (196608, 768, 1), 0), primals_10, primals_11, reinterpret_tensor(buf0, (8192, 48), (48, 1), 0), reinterpret_tensor(buf3, (2, 4096, 768), (3145728, 768, 1), 0), buf4, buf12, buf15, reinterpret_tensor(buf16, (8192, 768), (768, 1), 0), reinterpret_tensor(buf19, (2, 12, 4096, 64), (9437184, 64, 2304, 1), 0), reinterpret_tensor(buf19, (2, 12, 4096, 64), (9437184, 64, 2304, 1), 768), buf21, buf23, buf29, buf31, buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, 256, 3), (768, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, 4096, 3), (0, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
