
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
