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
 * How the warping error resamples the previous image at a pixel's previous
 * position (docs/MOTION.md §9, "Resampling bias"):
 * - `'bilinear'`: one hardware-filtered tap. It low-passes the previous image
 *   on any motion that isn't a whole number of pixels, so sharp images read
 *   less stable than soft ones. Needs a filterable format (`rgba32float` only
 *   with `float32-filterable`).
 * - `'catmull-rom'`: exact 16-tap (4×4) bicubic Catmull-Rom, deringed to the
 *   nearest 2×2 texels. Interpolating and exact on quadratics, so it keeps the
 *   detail a bilinear tap blurs away. It reads texels with `textureLoad`, so it
 *   takes any float-sampled color format, filterable or not.
 */
export type WarpingResample = 'bilinear' | 'catmull-rom';
/** Every {@link WarpingResample}. */
export const WARPING_RESAMPLE_FILTERS: readonly WarpingResample[] = ['bilinear', 'catmull-rom'];

/**
 * Checks a resample option (JavaScript callers get no type check).
 * @param value - The option as given; `undefined` selects `'bilinear'`
 * @returns The filter
 */
export function resolveWarpingResample(value: unknown): WarpingResample {
    if (value === undefined) return 'bilinear';
    if (typeof value === 'string' && (WARPING_RESAMPLE_FILTERS as readonly string[]).includes(value)) return value as WarpingResample;
    throw new Error(`WarpingError: resample must be one of ${WARPING_RESAMPLE_FILTERS.map((name) => `'${name}'`).join(', ')}; got ${JSON.stringify(value)}.`);
}

/**
 * The 4×4 Catmull-Rom resample (B = 0, C = 1/2; per-axis weights of the taps
 * at −1, 0, 1, 2 from the texel left of / above the sample sum to 1), deringed:
 * the result is clamped to the range of the 2×2 texels a bilinear tap would
 * blend, so the negative lobes can't overshoot a sharp edge (FSR2's history
 * resample does the same). 16 loads rather than the 5- or 9-tap
 * bilinear-optimized forms: those need hardware filtering (a filterable
 * format) and inherit its fixed-point weights, and the 5-tap one drops the
 * corner taps, an approximation a meter shouldn't carry. Edges clamp like the
 * bilinear sampler's default addressing.
 */
const SAMPLE_PREVIOUS = /* wgsl */ `
fn tapWeights(t : f32) -> vec4f {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4f(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

fn samplePrevious(uv : vec2f) -> vec3f {
    let p = uv * W.size - 0.5;
    let base = floor(p);
    let wx = tapWeights(p.x - base.x);
    let wy = tapWeights(p.y - base.y);
    let origin = vec2i(base) - 1;
    let last = vec2i(W.size) - 1;
    var sum = vec3f(0.0);
    var lo = vec3f(3.0e38);
    var hi = vec3f(-3.0e38);
    for (var j = 0; j < 4; j++) {
        let y = clamp(origin.y + j, 0, last.y);
        var row = vec3f(0.0);
        for (var i = 0; i < 4; i++) {
            let c = textureLoad(previous, vec2i(clamp(origin.x + i, 0, last.x), y), 0).rgb;
            row += wx[i] * c;
            if (i == 1 || i == 2) {
                if (j == 1 || j == 2) {
                    lo = min(lo, c);
                    hi = max(hi, c);
                }
            }
        }
        sum += wy[j] * row;
    }
    return select(sum, clamp(sum, lo, hi), DERING);
}
`;

/**
 * Warping error: a reference-free temporal-stability metric. Each pixel of the
 * current image is compared with the previous image resampled (bilinear by
 * default, see {@link WarpingResample}) where one contract motion field says
 * it came from, in conditioned luma (`l / (1 + l)`, firefly-safe for HDR). Low means the image moves with its
 * motion and doesn't flicker; it needs no ground truth.
 *
 * Occlusion mask (when a previous linear depth is bound): a pixel is scored
 * only where the previous linear depth at its previous position matches the
 * motion's own `.w` (CONTRACT.md: the surface's previous linear depth) within
 * a relative tolerance, so disoccluded pixels don't count as error. Pixels
 * flagged "no previous position" (`.z` = 4), coming from off-screen or failing
 * the mask are counted, not scored. The same per-pixel test and fixed-point
 * rounding as the warp residual (`motionResidual.ts`), so with camera motion
 * from depth as the field its counts equal that candidate's, pixel for pixel
 * (with `'bilinear'`, the audit's resample, its error does too).
 *
 * The motion and depth can be at render resolution and the images at display
 * resolution: both are read at the nearest texel to the pixel's UV. The error
 * sum is a 64-bit fixed-point pair (×1024), so 8K images can't overflow it.
 * @param resample - How the previous image is resampled (default `'bilinear'`)
 * @param dering - Catmull-Rom's 2×2 deringing clamp (default on; off only for the lab's A/B)
 * @returns The WGSL
 */
export function warpingErrorShader(resample: WarpingResample = 'bilinear', dering = true): string {
    const bilinear = resample === 'bilinear';
    // Bilinear keeps its sampler binding; the 16-tap filters only load, so binding 3 is absent.
    const sampler = bilinear ? '@group(0) @binding(3) var linearSampler : sampler;\n' : '';
    const resampler = bilinear ? '' : `\nconst DERING : bool = ${dering};\n${SAMPLE_PREVIOUS}`;
    const previousSample = bilinear ? 'textureSampleLevel(previous, linearSampler, prevUV, 0.0).rgb' : 'samplePrevious(prevUV)';
    return /* wgsl */ `
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
${sampler}@group(0) @binding(4) var motion : texture_2d<f32>;
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
${resampler}
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
            let lp = conditionedLuma(${previousSample});
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
}

/** The default (bilinear) warping error shader. */
export const WARPING_ERROR_SHADER = warpingErrorShader();
