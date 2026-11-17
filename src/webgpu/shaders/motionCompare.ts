/**
 * Motion-vector endpoint error against an oracle (EXPERIMENTAL, motion lab).
 *
 * The oracle is a motion field rendered in the same pass as the candidates, by
 * per-object materials that know the scene's exact previous positions. Both are
 * written for the same rasterized surface point, so the per-pixel difference is
 * the candidate's error and nothing else: no resampling, no aliasing, no
 * occlusion test. That makes it exact where the warp residual is only a proxy.
 *
 * Oracle texel (rgba16float): `.xy` motion in three's `velocity` convention
 * (NDC delta, current − previous), `.z` = `marker · 8 + object id` with the
 * marker {@link ORACLE_VALID} or {@link ORACLE_NO_PREVIOUS} (anything else,
 * such as a background clear colour, is uncovered) and the id in
 * `0 … COMPARE_OBJECTS − 1`, `.w` the previous linear depth (the contract's `.w`).
 *
 * Candidates are full contract texels (CONTRACT.md), so besides `.xy` the pass
 * scores their `.z` flag and `.w` depth. A flag is wrong when it claims no
 * previous position ({@link MotionFlag.NoPrevious}) or no surface where the
 * oracle has a previous position, or anything but `NoPrevious` where it has
 * none. The depth error is relative, `|w − oracle w| / oracle w`, over the
 * pixels where both claim a previous position.
 *
 * Per object, statistics words (stride {@link COMPARE_OBJECT_STRIDE}):
 * `[covered, noPrevious, Σ|oracle| px, reserved]`, then per candidate
 * (zero motion first, then up to three fields) `[Σ error px, error > 0.25 px,
 * error > 1 px, Σ depth error, depth pixels, wrong flags]`. Sums are fixed
 * point: endpoint errors ×{@link COMPARE_FIXED_POINT} capped at
 * {@link COMPARE_CAP_PIXELS} px, depth errors ×{@link COMPARE_DEPTH_FIXED_POINT}
 * capped at {@link COMPARE_DEPTH_CAP}, so a full frame up to 1920×1080 cannot
 * overflow a u32 (larger frames need a coarser fixed point).
 */

import { MotionFlag } from '../../flags.js';

/** Oracle marker for a covered pixel whose surface point had a previous position. */
export const ORACLE_VALID = 3;
/** Oracle marker for a covered pixel with no previous position (behind the previous camera, or no previous frame). */
export const ORACLE_NO_PREVIOUS = 2;
/** The oracle's `.z` for `marker` and object `id`. */
export const oracleCode = (marker: number, id: number): number => marker * 8 + id;
/** Object ids the comparison separates. */
export const COMPARE_OBJECTS = 8;
/** Zero motion plus up to three candidate fields. */
export const COMPARE_CANDIDATES = 4;
export const COMPARE_HEADER_WORDS = 4;
export const COMPARE_CANDIDATE_WORDS = 6;
export const COMPARE_OBJECT_STRIDE = COMPARE_HEADER_WORDS + COMPARE_CANDIDATES * COMPARE_CANDIDATE_WORDS;
export const COMPARE_WORDS = COMPARE_OBJECTS * COMPARE_OBJECT_STRIDE;
export const COMPARE_FIXED_POINT = 32;
export const COMPARE_CAP_PIXELS = 64;
export const COMPARE_SMALL_PIXELS = 0.25;
export const COMPARE_LARGE_PIXELS = 1;
export const COMPARE_DEPTH_FIXED_POINT = 4096;
export const COMPARE_DEPTH_CAP = 0.5;
export const COMPARE_CONSTANTS_BYTES = 16;

export const MOTION_COMPARE_SHADER = /* wgsl */ `
struct CompareConstants {
    size: vec2f,
    candidateCount: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> C: CompareConstants;
@group(0) @binding(1) var oracle: texture_2d<f32>;
@group(0) @binding(2) var motion1: texture_2d<f32>;
@group(0) @binding(3) var motion2: texture_2d<f32>;
@group(0) @binding(4) var motion3: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

const OBJECTS: u32 = ${COMPARE_OBJECTS}u;
const STRIDE: u32 = ${COMPARE_OBJECT_STRIDE}u;
const HEADER: u32 = ${COMPARE_HEADER_WORDS}u;
const PER_CANDIDATE: u32 = ${COMPARE_CANDIDATE_WORDS}u;
const WORDS: u32 = ${COMPARE_WORDS}u;
const FIXED: f32 = ${COMPARE_FIXED_POINT}.0;
const CAP: f32 = ${COMPARE_CAP_PIXELS}.0;

var<workgroup> local: array<atomic<u32>, WORDS>;

fn fixedPixels(value: f32) -> u32 {
    return u32(round(min(value, CAP) * FIXED));
}

fn candidate(index: u32, pixel: vec2i) -> vec4f {
    switch index {
        case 1u: { return textureLoad(motion1, pixel, 0); }
        case 2u: { return textureLoad(motion2, pixel, 0); }
        case 3u: { return textureLoad(motion3, pixel, 0); }
        default: { return vec4f(0.0); }
    }
}

fn fixedDepth(value: f32) -> u32 {
    return u32(round(min(value, ${COMPARE_DEPTH_CAP}) * ${COMPARE_DEPTH_FIXED_POINT}.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
    for (var i = lid; i < WORDS; i += 64u) { atomicStore(&local[i], 0u); }
    workgroupBarrier();

    // No early return: every invocation must reach the barriers.
    let inside = all(vec2f(gid.xy) < C.size);
    let pixel = vec2i(gid.xy);
    let texel = select(vec4f(0.0), textureLoad(oracle, pixel, 0), inside);
    let code = u32(max(round(texel.z), 0.0));
    let marker = f32(code / 8u);
    let id = code % 8u;
    let base = id * STRIDE;
    // NDC delta → render pixels.
    let toPixels = 0.5 * C.size;
    if (inside && marker == ${ORACLE_NO_PREVIOUS}.0) {
        atomicAdd(&local[base + 1u], 1u);
        for (var k = 0u; k <= C.candidateCount; k++) {
            let flag = round(candidate(k, pixel).z);
            if (flag != ${MotionFlag.NoPrevious}.0) { atomicAdd(&local[base + HEADER + k * PER_CANDIDATE + 5u], 1u); }
        }
    }
    if (inside && marker == ${ORACLE_VALID}.0) {
        atomicAdd(&local[base], 1u);
        atomicAdd(&local[base + 2u], fixedPixels(length(texel.xy * toPixels)));
        for (var k = 0u; k <= C.candidateCount; k++) {
            let value = candidate(k, pixel);
            let error = length((value.xy - texel.xy) * toPixels);
            let offset = base + HEADER + k * PER_CANDIDATE;
            atomicAdd(&local[offset], fixedPixels(error));
            if (error > ${COMPARE_SMALL_PIXELS}) { atomicAdd(&local[offset + 1u], 1u); }
            if (error > ${COMPARE_LARGE_PIXELS}.0) { atomicAdd(&local[offset + 2u], 1u); }
            let flag = round(value.z);
            let claims = flag >= ${MotionFlag.Exact}.0 && flag <= ${MotionFlag.Rigid}.0;
            if (claims) {
                atomicAdd(&local[offset + 3u], fixedDepth(abs(value.w - texel.w) / max(texel.w, 1e-6)));
                atomicAdd(&local[offset + 4u], 1u);
            } else {
                atomicAdd(&local[offset + 5u], 1u);
            }
        }
    }
    workgroupBarrier();

    for (var i = lid; i < WORDS; i += 64u) {
        let value = atomicLoad(&local[i]);
        if (value != 0u) { atomicAdd(&stats[i], value); }
    }
}
`;
