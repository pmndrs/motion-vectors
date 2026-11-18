import { MotionFlag } from '../../flags.js';
import { RESIDUAL_FIXED_POINT } from './motionResidual.js';

/** u32 words of the warping error counters: sum (lo, hi), scored, over, occluded, offscreen, noPrevious, pixels. */
export const WARPING_WORDS = 8;
/** Byte size of `WarpConstants`. */
export const WARPING_CONSTANTS_BYTES = 48;
/** Warping pass flags. */
export const WARPING_FLAG_DEPTH_TEST = 1;
export const WARPING_FLAG_HEAT = 2;

/**
 * Warping error: a reference-free temporal-stability metric. Each pixel of the
 * current image is compared with the previous image sampled (bilinear) where
 * one contract motion field says it came from, in conditioned luma
 * (`l / (1 + l)`, firefly-safe for HDR). Low means the image moves with its
 * motion and doesn't flicker; it needs no ground truth.
 *
 * Occlusion mask (when a previous linear depth is bound): a pixel is scored
 * only where the previous linear depth at its previous position matches the
 * motion's own `.w` (CONTRACT.md: the surface's previous linear depth) within
 * a relative tolerance, so disoccluded pixels don't count as error. Pixels
 * flagged "no previous position" (`.z` = 4), coming from off-screen or failing
 * the mask are counted, not scored. The same per-pixel test and fixed-point
 * rounding as the warp residual (`motionResidual.ts`), so with camera motion
 * from depth as the field its counts equal that candidate's, pixel for pixel.
 *
 * The motion and depth can be at render resolution and the images at display
 * resolution: both are read at the nearest texel to the pixel's UV. The error
 * sum is a 64-bit fixed-point pair (×1024), so 8K images can't overflow it.
 */
export const WARPING_ERROR_SHADER = /* wgsl */ `
struct WarpConstants {
    size : vec2f,
    invSize : vec2f,
    motionSize : vec2f,
    depthSize : vec2f,
    depthTolerance : f32,
    threshold : f32,
    flags : u32,
    pad : u32,
};

const FIXED : f32 = ${RESIDUAL_FIXED_POINT}.0;

@group(0) @binding(0) var<uniform> W : WarpConstants;
@group(0) @binding(1) var current : texture_2d<f32>;
@group(0) @binding(2) var previous : texture_2d<f32>;
@group(0) @binding(3) var linearSampler : sampler;
@group(0) @binding(4) var motion : texture_2d<f32>;
@group(0) @binding(5) var previousDepth : texture_2d<f32>;
@group(0) @binding(6) var heatOut : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<storage, read_write> counters : array<atomic<u32>, ${WARPING_WORDS}>;

// sum, scored, over, occluded, offscreen, noPrevious, pixels
var<workgroup> partial : array<atomic<u32>, 7>;

fn conditionedLuma(c : vec3f) -> f32 {
    let l = dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    return l / (1.0 + l);
}

fn texel(uv : vec2f, size : vec2f) -> vec2i {
    return vec2i(clamp(floor(uv * size), vec2f(0.0), size - 1.0));
}

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid : vec3u,
    @builtin(local_invocation_index) lid : u32,
) {
    if (lid < 7u) { atomicStore(&partial[lid], 0u); }
    workgroupBarrier();

    // No early return: every invocation must reach the barriers below.
    if (all(vec2f(gid.xy) < W.size)) {
        let uv = (vec2f(gid.xy) + 0.5) * W.invSize;
        let m = textureLoad(motion, texel(uv, W.motionSize), 0);
        let prevUV = uv - m.xy * vec2f(0.5, -0.5);
        // 0 scored, 1 occluded, 2 off-screen, 3 no previous position
        var state = 0u;
        if (round(m.z) == ${MotionFlag.NoPrevious}.0) {
            state = 3u;
        } else if (any(prevUV < vec2f(0.0)) || any(prevUV > vec2f(1.0))) {
            state = 2u;
        } else if ((W.flags & ${WARPING_FLAG_DEPTH_TEST}u) != 0u) {
            let prevDepth = textureLoad(previousDepth, texel(prevUV, W.depthSize), 0).r;
            if (!(m.w > 0.0 && abs(prevDepth - m.w) <= W.depthTolerance * m.w)) { state = 1u; }
        }
        var error = 0.0;
        if (state == 0u) {
            let lc = conditionedLuma(textureLoad(current, vec2i(gid.xy), 0).rgb);
            let lp = conditionedLuma(textureSampleLevel(previous, linearSampler, prevUV, 0.0).rgb);
            error = abs(lc - lp);
            atomicAdd(&partial[0], u32(round(error * FIXED)));
            atomicAdd(&partial[1], 1u);
            if (error > W.threshold) { atomicAdd(&partial[2], 1u); }
        } else {
            atomicAdd(&partial[2u + state], 1u);
        }
        atomicAdd(&partial[6], 1u);

        if ((W.flags & ${WARPING_FLAG_HEAT}u) != 0u) {
            // Scored: error ×8 as grey. Not scored: blue (mask), dark blue (off-screen / no previous).
            var color = vec4f(vec3f(saturate(error * 8.0)), 1.0);
            if (state == 1u) { color = vec4f(0.1, 0.25, 0.85, 1.0); }
            if (state >= 2u) { color = vec4f(0.05, 0.1, 0.35, 1.0); }
            textureStore(heatOut, vec2i(gid.xy), color);
        }
    }

    workgroupBarrier();
    if (lid == 0u) {
        // 64-bit sum: carry into the high word when the low word wraps.
        let sum = atomicLoad(&partial[0]);
        if (sum != 0u) {
            let before = atomicAdd(&counters[0], sum);
            if (before + sum < before) { atomicAdd(&counters[1], 1u); }
        }
        for (var i = 1u; i < 7u; i++) {
            let value = atomicLoad(&partial[i]);
            if (value != 0u) { atomicAdd(&counters[i + 1u], value); }
        }
    }
}
`;
