import { MotionFlag } from '../../flags.js';

/**
 * Cut detection on the GPU: the depth-consistency share (MOTION.md §7) reduced
 * to a 1×1 flag a consumer reads in the same frame, with no readback.
 *
 * `count` (8×8, one invocation per render pixel): a **surface** pixel (camera
 * motion's `.z` ≠ 0) is consistent when any candidate — zero motion, camera
 * motion from depth, and up to two more contract motion fields (an engine's
 * velocity) — reprojects to an on-screen previous position whose previous
 * linear depth matches the static-scene expectation (camera motion's `.w`)
 * within a relative tolerance. The same test as the warp residual's
 * (`motionResidual.ts`), so its `noneValid − backgroundUnscored` over the
 * surface share is this pass's share, pixel for pixel. Counts are reduced per
 * workgroup and added atomically.
 *
 * `resolve` (one invocation, same compute pass): reads and zeroes the counters
 * (so no clear is needed between frames), writes the verdict into `cutOut`
 * (rgba16float 1×1: r = reset 0/1, g = inconsistent share, b = surface share,
 * a = a previous frame existed) and the raw counts into `result` for an
 * optional readback. With no previous frame only `resolve` runs: reset 1.
 * Verdict logic mirrors `cutVerdict` (`src/math/cut.ts`).
 */
export const CUT_DETECTION_SHADER = /* wgsl */ `
struct CutConstants {
    size : vec2f,
    invSize : vec2f,
    depthTolerance : f32,
    threshold : f32,
    minSurface : f32,
    motionCount : u32,
    hasPrevious : u32,
    pad0 : u32,
    pad1 : u32,
    pad2 : u32,
};

@group(0) @binding(0) var<uniform> C : CutConstants;
@group(0) @binding(1) var cameraMotion : texture_2d<f32>;
@group(0) @binding(2) var previousDepth : texture_2d<f32>;
@group(0) @binding(3) var motion1 : texture_2d<f32>;
@group(0) @binding(4) var motion2 : texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> counters : array<atomic<u32>, 2>;
@group(0) @binding(6) var<storage, read_write> result : array<u32, 4>;
@group(0) @binding(7) var cutOut : texture_storage_2d<rgba16float, write>;

var<workgroup> inconsistentCount : atomic<u32>;
var<workgroup> surfaceCount : atomic<u32>;

fn texel(uv : vec2f) -> vec2i {
    return vec2i(clamp(floor(uv * C.size), vec2f(0.0), C.size - 1.0));
}

fn consistent(uv : vec2f, motion : vec2f, expected : f32) -> bool {
    let prevUV = uv - motion * vec2f(0.5, -0.5);
    if (any(prevUV < vec2f(0.0)) || any(prevUV > vec2f(1.0))) { return false; }
    let prevDepth = textureLoad(previousDepth, texel(prevUV), 0).r;
    return expected > 0.0 && abs(prevDepth - expected) <= C.depthTolerance * expected;
}

@compute @workgroup_size(8, 8)
fn count(
    @builtin(global_invocation_id) gid : vec3u,
    @builtin(local_invocation_index) lid : u32,
) {
    if (lid == 0u) {
        atomicStore(&inconsistentCount, 0u);
        atomicStore(&surfaceCount, 0u);
    }
    workgroupBarrier();

    // No early return: every invocation must reach the barriers.
    if (all(vec2f(gid.xy) < C.size)) {
        let pixel = vec2i(gid.xy);
        let camera = textureLoad(cameraMotion, pixel, 0);
        if (round(camera.z) != ${MotionFlag.NoSurface}.0) {
            atomicAdd(&surfaceCount, 1u);
            let uv = (vec2f(gid.xy) + 0.5) * C.invSize;
            let expected = camera.w;
            var ok = consistent(uv, vec2f(0.0), expected) || consistent(uv, camera.xy, expected);
            if (!ok && C.motionCount >= 1u) { ok = consistent(uv, textureLoad(motion1, pixel, 0).xy, expected); }
            if (!ok && C.motionCount >= 2u) { ok = consistent(uv, textureLoad(motion2, pixel, 0).xy, expected); }
            if (!ok) { atomicAdd(&inconsistentCount, 1u); }
        }
    }

    workgroupBarrier();
    if (lid == 0u) {
        let inconsistent = atomicLoad(&inconsistentCount);
        let surface = atomicLoad(&surfaceCount);
        if (inconsistent != 0u) { atomicAdd(&counters[0], inconsistent); }
        if (surface != 0u) { atomicAdd(&counters[1], surface); }
    }
}

@compute @workgroup_size(1)
fn resolve() {
    let inconsistent = atomicExchange(&counters[0], 0u);
    let surface = atomicExchange(&counters[1], 0u);
    let hasPrevious = C.hasPrevious != 0u;
    let surfaceShare = f32(surface) / (C.size.x * C.size.y);
    let share = select(0.0, f32(inconsistent) / f32(max(surface, 1u)), surface > 0u);
    let judged = hasPrevious && surfaceShare >= C.minSurface;
    let cut = !hasPrevious || (judged && share > C.threshold);
    result[0] = inconsistent;
    result[1] = surface;
    result[2] = u32(cut);
    result[3] = u32(hasPrevious);
    textureStore(cutOut, vec2i(0, 0), vec4f(f32(cut), share, surfaceShare, f32(hasPrevious)));
}
`;

/** Byte size of `CutConstants`. */
export const CUT_CONSTANTS_BYTES = 48;
/** u32 words of the resolve's `result`: inconsistent, surface, cut, hasPrevious. */
export const CUT_RESULT_WORDS = 4;
