# AOT ID: ['18_forward']
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


# kernel path: results/my_experiment/torchinductor_cache_0/bh/cbhvte5qpughjeidz7iqe6un2q7u2fd3uf2rqamngjarcbgfjrya.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.bfloat16), kwargs = {})
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
    x0 = (xindex % 48)
    x1 = ((xindex // 48) % 4096)
    x2 = xindex // 196608
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (12*((x1 % 64)) + 768*(x0 // 12) + 3072*(x1 // 64) + 196608*x2 + ((x0 % 12))), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/y5/cy5qdznkxoei7hu3l7r4f5ikkehod3527i2hc5uicnu22h5sjkyv.py
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
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/ea/ceatsj6qh5nx36637edbggxf25lbrwcuglvxx4jj2rubpapqy5wy.py
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
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/hg/chgrqwwlwiejbwemfwyinzgv63kwvlj3bbn643v2ea5qk6lu54m5.py
# Topologically Sorted Source Nodes: [cat, linear_1], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat => cat
#   linear_1 => convert_element_type_10
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cos, %sin], -1), kwargs = {})
#   %convert_element_type_10 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_3 = async_compile.triton('triton_poi_fused__to_copy_cat_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), xmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 1000.0
    tmp7 = tmp5 * tmp6
    tmp8 = x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = -9.210340371976184
    tmp11 = tmp9 * tmp10
    tmp12 = 0.0078125
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp7 * tmp14
    tmp16 = tl_math.cos(tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 256, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (x1), xmask & tmp19, eviction_policy='evict_last', other=0.0)
    tmp23 = 1000.0
    tmp24 = tmp22 * tmp23
    tmp25 = (-128) + x0
    tmp26 = tmp25.to(tl.float32)
    tmp27 = -9.210340371976184
    tmp28 = tmp26 * tmp27
    tmp29 = 0.0078125
    tmp30 = tmp28 * tmp29
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp24 * tmp31
    tmp33 = tl_math.sin(tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp18, tmp35)
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp37, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/jk/cjkagm2nhtq337jmewwp7vvy6qjtacw32bqg3whoiidrfuprwqx7.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_9
# Graph fragment:
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/o5/co5oawkg7l655q3fnrbgo5x2qxivyymsl4vgzxclljtigixyirvq.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_17
# Graph fragment:
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/kl/cklhrqha5jwnp3ttryzam5acy6lyk66zmmghd2il5x4nruuncqly.py
# Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_3 => convert_element_type_22
# Graph fragment:
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/se/csevvt6wbldg64sa4554jei5gvtp44qmoscoyewditof6aklad35.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %view_10], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 4352)
    x0 = (xindex % 3)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 4352, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = x0
    tmp13 = tl.full([1], 2, tl.int32)
    tmp14 = tmp12 == tmp13
    tmp15 = tmp13 == tmp13
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp13 == tmp16
    tmp18 = tmp16 == tmp16
    tmp19 = ((((-256) + x1) // 64) % 64)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp17, tmp20, tmp21)
    tmp24 = tl.where(tmp17, tmp22, tmp23)
    tmp25 = (((-256) + x1) % 64)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 + tmp26
    tmp28 = tl.where(tmp15, tmp27, tmp24)
    tmp29 = tmp12 == tmp16
    tmp30 = tl.where(tmp29, tmp20, tmp21)
    tmp31 = tl.where(tmp29, tmp22, tmp30)
    tmp32 = tl.where(tmp14, tmp27, tmp31)
    tmp33 = tl.where(tmp14, tmp28, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp9, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp8, tmp35)
    tl.store(out_ptr0 + (x3), tmp36, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/jp/cjpsropnzh35cmoaziaa45ndvilsfhdkpcsyg32fzlznweyn77c2.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_37, %convert_element_type_40, %convert_element_type_43], -3), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1671168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 48)
    x0 = (xindex % 4)
    x2 = xindex // 192
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (3*x2), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.float64)
    tmp14 = 2*(x1)
    tmp15 = tmp14.to(tl.float64)
    tmp16 = tl.full([1], 0.08333333333333333, tl.float64)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1], 10000.0, tl.float64)
    tmp19 = libdevice.pow(tmp18, tmp17)
    tmp20 = tl.full([1], 1, tl.int32)
    tmp21 = (tmp20 / tmp19)
    tmp22 = tl.full([1], 1.0, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 * tmp23
    tmp25 = libdevice.cos(tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tmp5 >= tmp8
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp5 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tmp31 & tmp4
    tmp33 = tl.load(in_ptr0 + (3*x2), tmp32, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34.to(tl.float64)
    tmp36 = 2*(x1)
    tmp37 = tmp36.to(tl.float64)
    tmp38 = tl.full([1], 0.08333333333333333, tl.float64)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.full([1], 10000.0, tl.float64)
    tmp41 = libdevice.pow(tmp40, tmp39)
    tmp42 = tl.full([1], 1, tl.int32)
    tmp43 = (tmp42 / tmp41)
    tmp44 = tl.full([1], 1.0, tl.float64)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp35 * tmp45
    tmp47 = libdevice.sin(tmp46)
    tmp48 = -tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp32, tmp48, tmp49)
    tmp51 = tmp5 >= tmp29
    tmp52 = tl.full([1], 3, tl.int64)
    tmp53 = tmp5 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tmp54 & tmp4
    tmp56 = tl.load(in_ptr0 + (3*x2), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp57.to(tl.float64)
    tmp59 = 2*(x1)
    tmp60 = tmp59.to(tl.float64)
    tmp61 = tl.full([1], 0.08333333333333333, tl.float64)
    tmp62 = tmp60 * tmp61
    tmp63 = tl.full([1], 10000.0, tl.float64)
    tmp64 = libdevice.pow(tmp63, tmp62)
    tmp65 = tl.full([1], 1, tl.int32)
    tmp66 = (tmp65 / tmp64)
    tmp67 = tl.full([1], 1.0, tl.float64)
    tmp68 = tmp66 * tmp67
    tmp69 = tmp58 * tmp68
    tmp70 = libdevice.sin(tmp69)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp55, tmp70, tmp71)
    tmp73 = tmp5 >= tmp52
    tmp74 = tl.full([1], 4, tl.int64)
    tmp75 = tmp5 < tmp74
    tmp76 = tmp73 & tmp4
    tmp77 = tl.load(in_ptr0 + (3*x2), tmp76, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tmp78.to(tl.float64)
    tmp80 = 2*(x1)
    tmp81 = tmp80.to(tl.float64)
    tmp82 = tl.full([1], 0.08333333333333333, tl.float64)
    tmp83 = tmp81 * tmp82
    tmp84 = tl.full([1], 10000.0, tl.float64)
    tmp85 = libdevice.pow(tmp84, tmp83)
    tmp86 = tl.full([1], 1, tl.int32)
    tmp87 = (tmp86 / tmp85)
    tmp88 = tl.full([1], 1.0, tl.float64)
    tmp89 = tmp87 * tmp88
    tmp90 = tmp79 * tmp89
    tmp91 = libdevice.cos(tmp90)
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp76, tmp91, tmp92)
    tmp94 = tl.where(tmp54, tmp72, tmp93)
    tmp95 = tl.where(tmp31, tmp50, tmp94)
    tmp96 = tl.where(tmp9, tmp27, tmp95)
    tmp97 = tmp96.to(tl.float32)
    tmp98 = tl.full(tmp97.shape, 0.0, tmp97.dtype)
    tmp99 = tl.where(tmp4, tmp97, tmp98)
    tmp100 = tmp0 >= tmp3
    tmp101 = tl.full([1], 27, tl.int64)
    tmp102 = tmp0 < tmp101
    tmp103 = tmp100 & tmp102
    tmp104 = x0
    tmp105 = tl.full([1], 0, tl.int64)
    tmp106 = tmp104 >= tmp105
    tmp107 = tl.full([1], 1, tl.int64)
    tmp108 = tmp104 < tmp107
    tmp109 = tmp108 & tmp103
    tmp110 = tl.load(in_ptr0 + (1 + 3*x2), tmp109, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp111.to(tl.float64)
    tmp113 = 2*((-6) + x1)
    tmp114 = tmp113.to(tl.float64)
    tmp115 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp116 = tmp114 * tmp115
    tmp117 = tl.full([1], 10000.0, tl.float64)
    tmp118 = libdevice.pow(tmp117, tmp116)
    tmp119 = tl.full([1], 1, tl.int32)
    tmp120 = (tmp119 / tmp118)
    tmp121 = tl.full([1], 1.0, tl.float64)
    tmp122 = tmp120 * tmp121
    tmp123 = tmp112 * tmp122
    tmp124 = libdevice.cos(tmp123)
    tmp125 = tl.full(tmp124.shape, 0.0, tmp124.dtype)
    tmp126 = tl.where(tmp109, tmp124, tmp125)
    tmp127 = tmp104 >= tmp107
    tmp128 = tl.full([1], 2, tl.int64)
    tmp129 = tmp104 < tmp128
    tmp130 = tmp127 & tmp129
    tmp131 = tmp130 & tmp103
    tmp132 = tl.load(in_ptr0 + (1 + 3*x2), tmp131, eviction_policy='evict_last', other=0.0)
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tmp133.to(tl.float64)
    tmp135 = 2*((-6) + x1)
    tmp136 = tmp135.to(tl.float64)
    tmp137 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp138 = tmp136 * tmp137
    tmp139 = tl.full([1], 10000.0, tl.float64)
    tmp140 = libdevice.pow(tmp139, tmp138)
    tmp141 = tl.full([1], 1, tl.int32)
    tmp142 = (tmp141 / tmp140)
    tmp143 = tl.full([1], 1.0, tl.float64)
    tmp144 = tmp142 * tmp143
    tmp145 = tmp134 * tmp144
    tmp146 = libdevice.sin(tmp145)
    tmp147 = -tmp146
    tmp148 = tl.full(tmp147.shape, 0.0, tmp147.dtype)
    tmp149 = tl.where(tmp131, tmp147, tmp148)
    tmp150 = tmp104 >= tmp128
    tmp151 = tl.full([1], 3, tl.int64)
    tmp152 = tmp104 < tmp151
    tmp153 = tmp150 & tmp152
    tmp154 = tmp153 & tmp103
    tmp155 = tl.load(in_ptr0 + (1 + 3*x2), tmp154, eviction_policy='evict_last', other=0.0)
    tmp156 = tmp155.to(tl.float32)
    tmp157 = tmp156.to(tl.float64)
    tmp158 = 2*((-6) + x1)
    tmp159 = tmp158.to(tl.float64)
    tmp160 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp161 = tmp159 * tmp160
    tmp162 = tl.full([1], 10000.0, tl.float64)
    tmp163 = libdevice.pow(tmp162, tmp161)
    tmp164 = tl.full([1], 1, tl.int32)
    tmp165 = (tmp164 / tmp163)
    tmp166 = tl.full([1], 1.0, tl.float64)
    tmp167 = tmp165 * tmp166
    tmp168 = tmp157 * tmp167
    tmp169 = libdevice.sin(tmp168)
    tmp170 = tl.full(tmp169.shape, 0.0, tmp169.dtype)
    tmp171 = tl.where(tmp154, tmp169, tmp170)
    tmp172 = tmp104 >= tmp151
    tmp173 = tl.full([1], 4, tl.int64)
    tmp174 = tmp104 < tmp173
    tmp175 = tmp172 & tmp103
    tmp176 = tl.load(in_ptr0 + (1 + 3*x2), tmp175, eviction_policy='evict_last', other=0.0)
    tmp177 = tmp176.to(tl.float32)
    tmp178 = tmp177.to(tl.float64)
    tmp179 = 2*((-6) + x1)
    tmp180 = tmp179.to(tl.float64)
    tmp181 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp182 = tmp180 * tmp181
    tmp183 = tl.full([1], 10000.0, tl.float64)
    tmp184 = libdevice.pow(tmp183, tmp182)
    tmp185 = tl.full([1], 1, tl.int32)
    tmp186 = (tmp185 / tmp184)
    tmp187 = tl.full([1], 1.0, tl.float64)
    tmp188 = tmp186 * tmp187
    tmp189 = tmp178 * tmp188
    tmp190 = libdevice.cos(tmp189)
    tmp191 = tl.full(tmp190.shape, 0.0, tmp190.dtype)
    tmp192 = tl.where(tmp175, tmp190, tmp191)
    tmp193 = tl.where(tmp153, tmp171, tmp192)
    tmp194 = tl.where(tmp130, tmp149, tmp193)
    tmp195 = tl.where(tmp108, tmp126, tmp194)
    tmp196 = tmp195.to(tl.float32)
    tmp197 = tl.full(tmp196.shape, 0.0, tmp196.dtype)
    tmp198 = tl.where(tmp103, tmp196, tmp197)
    tmp199 = tmp0 >= tmp101
    tmp200 = tl.full([1], 48, tl.int64)
    tmp201 = tmp0 < tmp200
    tmp202 = x0
    tmp203 = tl.full([1], 0, tl.int64)
    tmp204 = tmp202 >= tmp203
    tmp205 = tl.full([1], 1, tl.int64)
    tmp206 = tmp202 < tmp205
    tmp207 = tmp206 & tmp199
    tmp208 = tl.load(in_ptr0 + (2 + 3*x2), tmp207, eviction_policy='evict_last', other=0.0)
    tmp209 = tmp208.to(tl.float32)
    tmp210 = tmp209.to(tl.float64)
    tmp211 = 2*((-27) + x1)
    tmp212 = tmp211.to(tl.float64)
    tmp213 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp214 = tmp212 * tmp213
    tmp215 = tl.full([1], 10000.0, tl.float64)
    tmp216 = libdevice.pow(tmp215, tmp214)
    tmp217 = tl.full([1], 1, tl.int32)
    tmp218 = (tmp217 / tmp216)
    tmp219 = tl.full([1], 1.0, tl.float64)
    tmp220 = tmp218 * tmp219
    tmp221 = tmp210 * tmp220
    tmp222 = libdevice.cos(tmp221)
    tmp223 = tl.full(tmp222.shape, 0.0, tmp222.dtype)
    tmp224 = tl.where(tmp207, tmp222, tmp223)
    tmp225 = tmp202 >= tmp205
    tmp226 = tl.full([1], 2, tl.int64)
    tmp227 = tmp202 < tmp226
    tmp228 = tmp225 & tmp227
    tmp229 = tmp228 & tmp199
    tmp230 = tl.load(in_ptr0 + (2 + 3*x2), tmp229, eviction_policy='evict_last', other=0.0)
    tmp231 = tmp230.to(tl.float32)
    tmp232 = tmp231.to(tl.float64)
    tmp233 = 2*((-27) + x1)
    tmp234 = tmp233.to(tl.float64)
    tmp235 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp236 = tmp234 * tmp235
    tmp237 = tl.full([1], 10000.0, tl.float64)
    tmp238 = libdevice.pow(tmp237, tmp236)
    tmp239 = tl.full([1], 1, tl.int32)
    tmp240 = (tmp239 / tmp238)
    tmp241 = tl.full([1], 1.0, tl.float64)
    tmp242 = tmp240 * tmp241
    tmp243 = tmp232 * tmp242
    tmp244 = libdevice.sin(tmp243)
    tmp245 = -tmp244
    tmp246 = tl.full(tmp245.shape, 0.0, tmp245.dtype)
    tmp247 = tl.where(tmp229, tmp245, tmp246)
    tmp248 = tmp202 >= tmp226
    tmp249 = tl.full([1], 3, tl.int64)
    tmp250 = tmp202 < tmp249
    tmp251 = tmp248 & tmp250
    tmp252 = tmp251 & tmp199
    tmp253 = tl.load(in_ptr0 + (2 + 3*x2), tmp252, eviction_policy='evict_last', other=0.0)
    tmp254 = tmp253.to(tl.float32)
    tmp255 = tmp254.to(tl.float64)
    tmp256 = 2*((-27) + x1)
    tmp257 = tmp256.to(tl.float64)
    tmp258 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp259 = tmp257 * tmp258
    tmp260 = tl.full([1], 10000.0, tl.float64)
    tmp261 = libdevice.pow(tmp260, tmp259)
    tmp262 = tl.full([1], 1, tl.int32)
    tmp263 = (tmp262 / tmp261)
    tmp264 = tl.full([1], 1.0, tl.float64)
    tmp265 = tmp263 * tmp264
    tmp266 = tmp255 * tmp265
    tmp267 = libdevice.sin(tmp266)
    tmp268 = tl.full(tmp267.shape, 0.0, tmp267.dtype)
    tmp269 = tl.where(tmp252, tmp267, tmp268)
    tmp270 = tmp202 >= tmp249
    tmp271 = tl.full([1], 4, tl.int64)
    tmp272 = tmp202 < tmp271
    tmp273 = tmp270 & tmp199
    tmp274 = tl.load(in_ptr0 + (2 + 3*x2), tmp273, eviction_policy='evict_last', other=0.0)
    tmp275 = tmp274.to(tl.float32)
    tmp276 = tmp275.to(tl.float64)
    tmp277 = 2*((-27) + x1)
    tmp278 = tmp277.to(tl.float64)
    tmp279 = tl.full([1], 0.023809523809523808, tl.float64)
    tmp280 = tmp278 * tmp279
    tmp281 = tl.full([1], 10000.0, tl.float64)
    tmp282 = libdevice.pow(tmp281, tmp280)
    tmp283 = tl.full([1], 1, tl.int32)
    tmp284 = (tmp283 / tmp282)
    tmp285 = tl.full([1], 1.0, tl.float64)
    tmp286 = tmp284 * tmp285
    tmp287 = tmp276 * tmp286
    tmp288 = libdevice.cos(tmp287)
    tmp289 = tl.full(tmp288.shape, 0.0, tmp288.dtype)
    tmp290 = tl.where(tmp273, tmp288, tmp289)
    tmp291 = tl.where(tmp251, tmp269, tmp290)
    tmp292 = tl.where(tmp228, tmp247, tmp291)
    tmp293 = tl.where(tmp206, tmp224, tmp292)
    tmp294 = tmp293.to(tl.float32)
    tmp295 = tl.full(tmp294.shape, 0.0, tmp294.dtype)
    tmp296 = tl.where(tmp199, tmp294, tmp295)
    tmp297 = tl.where(tmp103, tmp198, tmp296)
    tmp298 = tl.where(tmp4, tmp99, tmp297)
    tl.store(out_ptr0 + (x4), tmp298, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/v2/cv2qixo57wkajjase3wgdlu2umleslny4tzegmspmbrw2crbi733.py
# Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu => convert_element_type_14, convert_element_type_15, mul_4, sigmoid
# Graph fragment:
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_1, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_14,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %sigmoid), kwargs = {})
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
triton_poi_fused_silu_9 = async_compile.triton('triton_poi_fused_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/4a/c4ajcwaau4zikq7d44pyv73mtpd3vl4ioo3umjdggywdax3cus6z.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_2 => constant_pad_nd_default_3
# Graph fragment:
#   %constant_pad_nd_default_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convert_element_type_15, [0, 0, 0, 6]), kwargs = {})
triton_poi_fused_addmm_10 = async_compile.triton('triton_poi_fused_addmm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1152
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), xmask & tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (2, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(primals_2, (1152, 48), (48, 1))
    assert_size_stride(primals_3, (1152, ), (1, ))
    assert_size_stride(primals_4, (2, ), (1, ))
    assert_size_stride(primals_5, (1152, 256), (256, 1))
    assert_size_stride(primals_6, (1152, ), (1, ))
    assert_size_stride(primals_7, (1152, 1152), (1152, 1))
    assert_size_stride(primals_8, (1152, ), (1, ))
    assert_size_stride(primals_9, (2, 256, 19), (4864, 19, 1))
    assert_size_stride(primals_10, (1152, 19), (19, 1))
    assert_size_stride(primals_11, (1152, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 4096, 48), (196608, 48, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_1, buf0, 393216, stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((1152, 48), (48, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_2, buf1, 55296, stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((1152, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_3, buf2, 1152, stream=stream0)
        del primals_3
        buf4 = empty_strided_cuda((2, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat, linear_1], Original ATen: [aten.cat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_3.run(primals_4, buf4, 512, stream=stream0)
        del primals_4
        buf5 = empty_strided_cuda((1152, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(primals_5, buf5, 294912, stream=stream0)
        del primals_5
        buf6 = empty_strided_cuda((1152, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_6, buf6, 1152, stream=stream0)
        del primals_6
        buf10 = empty_strided_cuda((1152, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(primals_7, buf10, 1327104, stream=stream0)
        del primals_7
        buf11 = empty_strided_cuda((1152, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_8, buf11, 1152, stream=stream0)
        del primals_8
        buf13 = empty_strided_cuda((1152, 19), (19, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(primals_10, buf13, 21888, stream=stream0)
        del primals_10
        buf14 = empty_strided_cuda((1152, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_11, buf14, 1152, stream=stream0)
        del primals_11
        buf16 = empty_strided_cuda((2, 4352, 3), (13056, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf16, 26112, stream=stream0)
        buf3 = empty_strided_cuda((8192, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf2, reinterpret_tensor(buf0, (8192, 48), (48, 1), 0), reinterpret_tensor(buf1, (48, 1152), (1, 48), 0), alpha=1, beta=1, out=buf3)
        del buf1
        del buf2
        buf7 = empty_strided_cuda((2, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf6, buf4, reinterpret_tensor(buf5, (256, 1152), (1, 256), 0), alpha=1, beta=1, out=buf7)
        del buf5
        del buf6
        buf15 = empty_strided_cuda((512, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf14, reinterpret_tensor(primals_9, (512, 19), (19, 1), 0), reinterpret_tensor(buf13, (19, 1152), (1, 19), 0), alpha=1, beta=1, out=buf15)
        del buf14
        buf17 = empty_strided_cuda((2, 4352, 48, 2, 2), (835584, 192, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf16, buf17, 1671168, stream=stream0)
        del buf16
        buf8 = empty_strided_cuda((2, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_silu_9.run(buf7, buf8, 2304, stream=stream0)
        buf9 = empty_strided_cuda((8, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_10.run(buf8, buf9, 9216, stream=stream0)
        buf12 = empty_strided_cuda((8, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf11, buf9, reinterpret_tensor(buf10, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf12)
        del buf11
        del buf9
        buf18 = empty_strided_cuda((2, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_1], Original ATen: [aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_silu_9.run(buf12, buf18, 2304, stream=stream0)
    return (buf18, reinterpret_tensor(buf12, (2, 1152), (1152, 1), 0), reinterpret_tensor(buf3, (2, 4096, 1152), (4718592, 1152, 1), 0), reinterpret_tensor(buf15, (2, 256, 1152), (294912, 1152, 1), 0), reinterpret_tensor(buf17, (2, 1, 4352, 48, 2, 2), (835584, 835584, 192, 4, 2, 1), 0), reinterpret_tensor(buf0, (8192, 48), (48, 1), 0), buf4, buf7, buf8, reinterpret_tensor(buf12, (2, 1152), (1152, 1), 0), reinterpret_tensor(primals_9, (512, 19), (19, 1), 0), buf13, buf10, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1152, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1152, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, 256, 19), (4864, 19, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((1152, 19), (19, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
