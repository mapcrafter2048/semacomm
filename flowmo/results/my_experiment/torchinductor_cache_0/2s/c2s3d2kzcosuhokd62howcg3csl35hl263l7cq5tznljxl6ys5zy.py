# AOT ID: ['0_backward']
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


# kernel path: results/my_experiment/torchinductor_cache_0/bd/cbd22e6jyf67sf2xa2gb4eufruwchlvjvoije34wcugshfc6yxtm.py
# Topologically Sorted Source Nodes: [logits, probs, truediv_1, add, log_probs], Original ATen: [aten.mul, aten._softmax, aten.div, aten.add, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   add => add
#   log_probs => convert_element_type_7, sub_1, sub_2
#   logits => mul_1
#   probs => div_1, exp
#   truediv_1 => div_2
# Graph fragment:
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 2), kwargs = {})
#   %convert_element_type_default : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.float32), kwargs = {})
#   %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, 1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 0.1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %div_1 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_2), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %div_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, 0.1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_2, 1e-05), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %amax_1), kwargs = {})
#   %sub_2 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %sub_2), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %permute_12), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [-1], True), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_2, %sum_6), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_14, %mul_18), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_7, torch.bfloat16), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_11, 0.1), kwargs = {})
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %div_1), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [-1], True), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %fma : [num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg_6, %sum_7, %mul_19), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%fma, torch.bfloat16), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_12, 0.1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %div_8), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 2), kwargs = {})
triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0 = async_compile.triton('triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp9 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.0025
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.001953125
    tmp7 = tmp5 * tmp6
    tmp8 = -tmp7
    tmp10 = 2.0
    tmp11 = tmp9 * tmp10
    tmp12 = 10.0
    tmp13 = tmp11 * tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 - tmp19
    tmp21 = tmp8 * tmp20
    tmp22 = -tmp3
    tmp23 = tmp22 * tmp4
    tmp24 = -tmp23
    tmp26 = tmp25 + tmp14
    tmp27 = tl_math.log(tmp26)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp24 * tmp25
    tmp30 = (tmp29 / tmp26)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31 * tmp6
    tmp33 = tmp21 + tmp32
    tmp34 = tmp11.to(tl.float32)
    tmp35 = tmp34 * tmp4
    tmp37 = tmp35 - tmp36
    tmp38 = tmp37 * tmp12
    tmp39 = tl_math.exp(tmp38)
    tmp41 = (tmp39 / tmp40)
    tmp42 = tmp8 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [R0_BLOCK])
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp46 = tmp33 * tmp41
    tmp47 = tl.broadcast_to(tmp46, [R0_BLOCK])
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp47, 0))
    tmp50 = tl_math.exp(tmp20)
    tmp51 = tmp50 * tmp45
    tmp52 = tmp42 - tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 * tmp12
    tmp55 = -tmp41
    tmp56 = libdevice.fma(tmp55, tmp49, tmp46)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp57 * tmp12
    tmp59 = tmp54 + tmp58
    tmp60 = tmp59 * tmp10
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp60, None)
''', device_str='cuda')


# kernel path: results/my_experiment/torchinductor_cache_0/mz/cmzxghy63jb5j566fxllkxexikiebhkoqfeedbxuoykgvrx6ns4v.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 2}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 9)
    y1 = yindex // 9
    tmp0 = tl.load(in_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 2*y1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, YBLOCK])
    tmp11 = tl.load(in_ptr3 + (y0 + 9*x2 + 18*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = 0.000625
    tmp5 = tmp3 * tmp4
    tmp6 = 0.00021701388888888888
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp0 + tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x2 + 2*y3), tmp12, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    bmm, amax_default, sum_2, amax_1, log, mean, mul_10, permute_14, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(bmm, (1, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_default, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(sum_2, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(amax_1, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(log, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mean, (512, ), (1, ))
    assert_size_stride(mul_10, (1, 512, 1, 9), (4608, 1, 4608, 512))
    assert_size_stride(permute_14, (1, 512, 9), (9, 9, 1))
    assert_size_stride(tangents_1, (1, 256, 18), (4608, 18, 1))
    assert_size_stride(tangents_2, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = reinterpret_tensor(bmm, (1, 512, 1, 512), (262144, 512, 512, 1), 0); del bmm  # reuse
        # Topologically Sorted Source Nodes: [logits, probs, truediv_1, add, log_probs], Original ATen: [aten.mul, aten._softmax, aten.div, aten.add, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten._softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0.run(buf4, tangents_2, amax_1, log, mean, amax_default, sum_2, 512, 512, stream=stream0)
        del amax_1
        del amax_default
        del log
        del mean
        del sum_2
        buf5 = empty_strided_cuda((1, 512, 9), (4608, 9, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (1, 512, 512), (0, 512, 1), 0), permute_14, out=buf5)
        del buf4
        del permute_14
        buf6 = empty_strided_cuda((1, 256, 9, 2), (4608, 18, 2, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(tangents_1, mul_10, tangents_2, buf5, buf6, 2304, 2, stream=stream0)
        del buf5
        del mul_10
        del tangents_1
        del tangents_2
    return (reinterpret_tensor(buf6, (1, 256, 18), (4608, 18, 1), 0), None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    bmm = rand_strided((1, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((1, 512, 1, 9), (4608, 1, 4608, 512), device='cuda:0', dtype=torch.float32)
    permute_14 = rand_strided((1, 512, 9), (9, 9, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 256, 18), (4608, 18, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([bmm, amax_default, sum_2, amax_1, log, mean, mul_10, permute_14, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
