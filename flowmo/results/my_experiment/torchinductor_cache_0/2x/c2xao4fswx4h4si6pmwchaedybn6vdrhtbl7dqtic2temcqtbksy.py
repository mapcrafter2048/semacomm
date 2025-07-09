# AOT ID: ['3_forward']
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


# kernel path: results/my_experiment/torchinductor_cache_0/fa/cfavui4kn4kgeak27nfmmchxvtxtcge3o4ooxy6hkrqbmbzvtdlm.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_1, permute
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_0 = async_compile.triton('triton_poi_fused__to_copy_t_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/5b/c5bdwym3qsv6cf75pu22w37enoemdruyony4joisaimognmxtqsk.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%primals_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 196608)
    x1 = xindex // 196608
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3342336*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/6g/c6gt4qdx6tcmrrq6ria5j3oqwophhfqi43i76tem3nvkfhuws73u.py
# Topologically Sorted Source Nodes: [linear, mul, add, layer_norm, mul_1, add_1, linear_1], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_3
#   layer_norm => add_2, convert_element_type_4, mul_1, rsqrt, sub, var_mean
#   linear => add, convert_element_type
#   linear_1 => convert_element_type_7
#   mul => mul
#   mul_1 => mul_2
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %convert_element_type), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, %mul), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 1.0), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_2 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel):
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
    tmp35 = tmp34.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp7, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp35, r0_mask)
    tl.store(out_ptr0 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/yg/cyg7vpbog4rsbfhnk62rh2367ggj7pdn3rocvxsykoftd3xs2wkn.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_6
# Graph fragment:
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/az/cazso4zzeufbmwivjd4kuttwrgobfkdrdfgoovegrhluaz24olie.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_5
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_6, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/so/csoi7shjq3tgk65ghexssqxqh735fxjd2kfuvoonpj4wc33yeeg3.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_4, add_5, convert_element_type_11, convert_element_type_12, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, tanh
# Graph fragment:
#   %convert_element_type_11 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, %convert_element_type_11), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %convert_element_type_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 0.044715), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_11, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.7978845608028654), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, 0.5), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_5), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_5 = async_compile.triton('triton_poi_fused_gelu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: results/my_experiment/torchinductor_cache_0/m4/cm4f5regva6hclnyu4xqs5gfjczqv43eazaqlru4wykjwkn3aofw.py
# Topologically Sorted Source Nodes: [mul_2, add_2, layer_norm_2, mul_8, add_6], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_2 => add_6
#   add_6 => add_12
#   layer_norm_2 => add_11, convert_element_type_31, mul_16, rsqrt_4, sub_2, var_mean_2
#   mul_2 => mul_9
#   mul_8 => mul_17
# Graph fragment:
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, 1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_9), kwargs = {})
#   %convert_element_type_31 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_31, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_31, %getitem_5), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_4), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 1.0), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, 0), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_6 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel):
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


# kernel path: results/my_experiment/torchinductor_cache_0/ou/coujrnwfbe6v56qyyics4xsog6exr4iotdm2kfjtolrxvbwljnhg.py
# Topologically Sorted Source Nodes: [layer_norm_1, mul_3, add_3, linear_3], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_8
#   layer_norm_1 => add_7, convert_element_type_18, mul_10, rsqrt_1, sub_1, var_mean_1
#   linear_3 => convert_element_type_21
#   mul_3 => mul_11
# Graph fragment:
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_18, %getitem_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 1.0), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, 0), kwargs = {})
#   %convert_element_type_21 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mul_native_layer_norm_7 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_7(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel):
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


# kernel path: results/my_experiment/torchinductor_cache_0/uz/cuzl2bb7im7wvy7j7f3336vmxgmcs6t2hqx3vumqi4niy5woxkud.py
# Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_3 => convert_element_type_20
# Graph fragment:
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.bfloat16), kwargs = {})
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


# kernel path: results/my_experiment/torchinductor_cache_0/nc/cncrihntpdfb3ijhima4bmo52xt7kq4wj7x5lizye3ar4vxpork5.py
# Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_3 => convert_element_type_19
# Graph fragment:
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_11, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/t3/ct3gvq4i4iddbehh5po6fr37u7t7hjiar27ajccchlvsbve5elxh.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_25
#   mean => mean
#   pow_1 => pow_1
# Graph fragment:
#   %convert_element_type_25 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_25, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_10 = async_compile.triton('triton_per_fused__to_copy_mean_pow_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: results/my_experiment/torchinductor_cache_0/fw/cfwzeq2bdcsabrxlsabh4rdofuib5h5cx6xxqy7nm6op7xv66v7p.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean, add_4, rsqrt], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   add_4 => add_9
#   float_1 => convert_element_type_25
#   mean => mean
#   pow_1 => pow_1
#   rsqrt => rsqrt_2
# Graph fragment:
#   %convert_element_type_25 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_25, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
triton_poi_fused__to_copy_add_mean_pow_rsqrt_11 = async_compile.triton('triton_poi_fused__to_copy_add_mean_pow_rsqrt_11', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_pow_rsqrt_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_pow_rsqrt_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: results/my_experiment/torchinductor_cache_0/rl/crlz7on7lq5ttbtaiulwx7edwrsljlehy2kdz3ixmclqdmupibe7.py
# Topologically Sorted Source Nodes: [float_2, pow_2, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_27
#   mean_1 => mean_1
#   pow_2 => pow_2
# Graph fragment:
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_1, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_27, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_12 = async_compile.triton('triton_per_fused__to_copy_mean_pow_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_12(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: results/my_experiment/torchinductor_cache_0/5c/c5cv7t6dhky5crhipklaoojjbbnmlzp5wv3td7rm53q2qpulpbug.py
# Topologically Sorted Source Nodes: [float_1, mul_4, to, mul_5, to_2], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_25
#   mul_4 => mul_12
#   mul_5 => mul_13
#   to => convert_element_type_26
#   to_2 => convert_element_type_29
# Graph fragment:
#   %convert_element_type_25 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.float32), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_25, %rsqrt_2), kwargs = {})
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %primals_12), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_mul_13 = async_compile.triton('triton_poi_fused__to_copy_mul_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: results/my_experiment/torchinductor_cache_0/mq/cmqbxi6q3nfv4fhsv3txnmrmspc7otu7zrujdkmnhdkqlzgchc24.py
# Topologically Sorted Source Nodes: [float_2, mul_6, to_1, mul_7, to_3], Original ATen: [aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   float_2 => convert_element_type_27
#   mul_6 => mul_14
#   mul_7 => mul_15
#   to_1 => convert_element_type_28
#   to_3 => convert_element_type_30
# Graph fragment:
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_1, torch.float32), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_27, %rsqrt_3), kwargs = {})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_28, %primals_13), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_mul_14 = async_compile.triton('triton_poi_fused__to_copy_mul_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (2, 256, 768), (3342336, 768, 1))
    assert_size_stride(primals_2, (768, 768), (768, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (2, 256, 768), (196608, 768, 1))
    assert_size_stride(primals_5, (3072, 768), (768, 1))
    assert_size_stride(primals_6, (3072, ), (1, ))
    assert_size_stride(primals_7, (768, 3072), (3072, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (2, 4096, 768), (3145728, 768, 1))
    assert_size_stride(primals_10, (2304, 768), (768, 1))
    assert_size_stride(primals_11, (2304, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_0.run(primals_2, buf0, 589824, stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(primals_1, buf1, 393216, stream=stream0)
        del primals_1
        buf2 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (512, 768), (768, 1), 0), buf0, out=buf2)
        buf3 = reinterpret_tensor(buf2, (2, 256, 768), (196608, 768, 1), 0); del buf2  # reuse
        buf4 = empty_strided_cuda((2, 256, 1), (256, 1, 1), torch.float32)
        buf5 = empty_strided_cuda((2, 256, 1), (256, 1, 512), torch.float32)
        buf7 = reinterpret_tensor(buf5, (2, 256, 1), (256, 1, 1), 0); del buf5  # reuse
        buf8 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear, mul, add, layer_norm, mul_1, add_1, linear_1], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_2.run(buf3, buf7, primals_4, primals_3, buf4, buf8, 512, 768, stream=stream0)
        del primals_3
        del primals_4
        buf9 = empty_strided_cuda((3072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_5, buf9, 2359296, stream=stream0)
        del primals_5
        buf10 = empty_strided_cuda((3072, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(primals_6, buf10, 3072, stream=stream0)
        del primals_6
        buf11 = empty_strided_cuda((512, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf10, reinterpret_tensor(buf8, (512, 768), (768, 1), 0), reinterpret_tensor(buf9, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf11)
        del buf10
        buf12 = empty_strided_cuda((2, 256, 3072), (786432, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_5.run(buf11, buf12, 1572864, stream=stream0)
        buf13 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_7, buf13, 2359296, stream=stream0)
        del primals_7
        buf14 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 3072), (3072, 1), 0), buf13, out=buf14)
        buf15 = reinterpret_tensor(buf14, (2, 256, 768), (196608, 768, 1), 0); del buf14  # reuse
        buf30 = empty_strided_cuda((2, 256, 1), (256, 1, 1), torch.float32)
        buf31 = empty_strided_cuda((2, 256, 1), (256, 1, 512), torch.float32)
        buf33 = reinterpret_tensor(buf31, (2, 256, 1), (256, 1, 1), 0); del buf31  # reuse
        buf34 = empty_strided_cuda((2, 256, 768), (196608, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, add_2, layer_norm_2, mul_8, add_6], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_6.run(buf15, buf33, buf3, primals_8, buf30, buf34, 512, 768, stream=stream0)
        del primals_8
        buf16 = empty_strided_cuda((2, 4096, 1), (4096, 1, 1), torch.float32)
        buf17 = empty_strided_cuda((2, 4096, 1), (4096, 1, 8192), torch.float32)
        buf19 = reinterpret_tensor(buf17, (2, 4096, 1), (4096, 1, 1), 0); del buf17  # reuse
        buf20 = empty_strided_cuda((2, 4096, 768), (3145728, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm_1, mul_3, add_3, linear_3], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_7.run(buf19, primals_9, buf16, buf20, 8192, 768, stream=stream0)
        buf21 = empty_strided_cuda((2304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_10, buf21, 1769472, stream=stream0)
        del primals_10
        buf22 = empty_strided_cuda((2304, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(primals_11, buf22, 2304, stream=stream0)
        del primals_11
        buf23 = empty_strided_cuda((8192, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf22, reinterpret_tensor(buf20, (8192, 768), (768, 1), 0), reinterpret_tensor(buf21, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf23)
        del buf22
        buf24 = empty_strided_cuda((2, 12, 4096, 1), (49152, 1, 12, 98304), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_10.run(buf23, buf24, 98304, 64, stream=stream0)
        buf25 = empty_strided_cuda((2, 12, 4096, 1), (49152, 4096, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean, add_4, rsqrt], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_11.run(buf24, buf25, 24, 4096, stream=stream0)
        buf26 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [float_2, pow_2, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_12.run(buf23, buf26, 98304, 64, stream=stream0)
        buf27 = empty_strided_cuda((2, 12, 4096, 1), (49152, 4096, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_2, pow_2, mean_1, add_5, rsqrt_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_11.run(buf26, buf27, 24, 4096, stream=stream0)
        del buf26
        buf28 = empty_strided_cuda((2, 12, 4096, 64), (3145728, 64, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_1, mul_4, to, mul_5, to_2], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_13.run(buf23, buf25, primals_12, buf28, 6291456, stream=stream0)
        buf29 = empty_strided_cuda((2, 12, 4096, 64), (3145728, 64, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_2, mul_6, to_1, mul_7, to_3], Original ATen: [aten._to_copy, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_14.run(buf23, buf27, primals_13, buf29, 6291456, stream=stream0)
    return (buf34, buf28, buf29, reinterpret_tensor(buf23, (2, 12, 4096, 64), (9437184, 64, 2304, 1), 1536), buf15, primals_9, primals_12, primals_13, reinterpret_tensor(buf1, (512, 768), (768, 1), 0), buf3, buf4, buf7, reinterpret_tensor(buf8, (512, 768), (768, 1), 0), buf11, reinterpret_tensor(buf12, (512, 3072), (3072, 1), 0), buf15, buf16, buf19, reinterpret_tensor(buf20, (8192, 768), (768, 1), 0), reinterpret_tensor(buf23, (2, 12, 4096, 64), (9437184, 64, 2304, 1), 0), reinterpret_tensor(buf23, (2, 12, 4096, 64), (9437184, 64, 2304, 1), 768), buf25, buf27, buf30, buf33, buf21, reinterpret_tensor(buf13, (768, 3072), (3072, 1), 0), buf9, reinterpret_tensor(buf0, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 256, 768), (3342336, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, 256, 768), (196608, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, 4096, 768), (3145728, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
