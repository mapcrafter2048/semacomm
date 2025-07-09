
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
