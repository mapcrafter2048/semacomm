# AOT ID: ['0_inference']
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


# kernel path: results/my_experiment/torchinductor_cache_0/ac/cacragqxcl5aqoz5le3j32thnnh7ins5pndi7vkuoynos5vxdpf3.py
# Topologically Sorted Source Nodes: [img_ids, iadd, iadd_1], Original ATen: [aten.zeros, aten.add]
# Source node to ATen node mapping:
#   iadd => add
#   iadd_1 => add_1
#   img_ids => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([64, 64, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select, %unsqueeze), kwargs = {})
#   %select_scatter_default : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %add, 2, 1), kwargs = {})
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %select_1, 2, 1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_6, %unsqueeze_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %add_1, 2, 2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %select_7, 2, 2), kwargs = {})
triton_poi_fused_add_zeros_0 = async_compile.triton('triton_poi_fused_add_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_zeros_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3)
    x2 = xindex // 192
    x1 = ((xindex // 3) % 64)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp1 == tmp4
    tmp6 = tmp4 == tmp4
    tmp7 = x2
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 0.0
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = tl.where(tmp5, tmp8, tmp9)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = x1
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = tl.where(tmp3, tmp15, tmp12)
    tmp17 = tmp0 == tmp4
    tmp18 = tl.where(tmp17, tmp8, tmp9)
    tmp19 = tl.where(tmp17, tmp10, tmp18)
    tmp20 = tl.where(tmp2, tmp15, tmp19)
    tmp21 = tl.where(tmp2, tmp16, tmp20)
    tl.store(out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/2g/c2giwn4pcqsvr4il64ulm7duwrp2k4nqymoua6vnxfa4kz7artsa.py
# Topologically Sorted Source Nodes: [txt_ids, txt_ids_1], Original ATen: [aten.zeros, aten.add]
# Source node to ATen node mapping:
#   txt_ids => full_default_1
#   txt_ids_1 => add_2
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2, 256, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_1, %unsqueeze_4), kwargs = {})
triton_poi_fused_add_zeros_1 = async_compile.triton('triton_poi_fused_add_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 256)
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 64, 3), (192, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [img_ids, iadd, iadd_1], Original ATen: [aten.zeros, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_zeros_0.run(buf0, 12288, stream=stream0)
        buf1 = empty_strided_cuda((2, 256, 3), (768, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [txt_ids, txt_ids_1], Original ATen: [aten.zeros, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_zeros_1.run(buf1, 1536, stream=stream0)
    return (reinterpret_tensor(buf0, (2, 4096, 3), (0, 3, 1), 0), buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
