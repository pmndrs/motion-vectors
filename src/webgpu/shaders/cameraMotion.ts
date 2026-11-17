import { MotionFlag } from '../../flags.js';

/**
 * Camera motion from depth (EXPERIMENTAL, motion lab).
 *
 * Unprojects each render-res depth sample through the *jittered* inverse
 * view-projection (the one the depth was rasterized with), then projects the
 * world point through the current and previous *unjittered* view-projections.
 * The difference is the motion a static world point made on screen — exactly
 * what three's `velocity` node reports for static geometry, with no velocity
 * render target. Moving objects are wrong by construction (their own motion is
 * absent); measuring how wrong is part of the experiment.
 *
 * Output `motion` (rgba16float) is the motion contract (CONTRACT.md): xy = NDC
 * delta, current − previous (three's `velocity` convention, so it can be fed
 * to the upscaler unchanged); z = the flag: camera-only (2) on surfaces, no
 * previous position (4) behind the previous camera or without a previous frame
 * (`params.y` = 0), no surface (0) where the depth is the clear value
 * (`params.x`); w = the point's linear depth in the previous camera, 0 with no
 * previous position. On no-surface pixels xy and w describe the far plane's
 * point, so the warp residual can still score the sky. `linearDepth`
 * (r32float) keeps this frame's linear depth for the next frame's occlusion test.
 */
export const CAMERA_MOTION_SHADER = /* wgsl */ `
struct MotionConstants {
    invViewProjJittered : mat4x4f,
    viewProj : mat4x4f,
    prevViewProj : mat4x4f,
    viewZRow : vec4f,
    prevViewZRow : vec4f,
    renderSize : vec2f,
    invRenderSize : vec2f,
    params : vec4f,
};

@group(0) @binding(0) var<uniform> M : MotionConstants;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var motionOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var linearDepthOut : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= M.renderSize)) { return; }
    let pixel = vec2i(gid.xy);
    let d = textureLoad(depthTex, pixel, 0);
    let uv = (vec2f(gid.xy) + 0.5) * M.invRenderSize;
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let worldH = M.invViewProjJittered * ndc;
    let world = vec4f(worldH.xyz / worldH.w, 1.0);
    let cur = M.viewProj * world;
    let prev = M.prevViewProj * world;
    let linearDepth = -dot(M.viewZRow, world);
    let prevLinearDepth = -dot(M.prevViewZRow, world);
    // A point behind the previous camera has no previous screen position, and
    // neither has anything on the first frame (previous = current there).
    let none = prev.w <= 1e-6 || M.params.y < 0.5;
    let velocity = select(cur.xy / cur.w - prev.xy / prev.w, vec2f(0.0), none);
    let background = d == M.params.x;
    let flag = select(select(${MotionFlag.CameraOnly}.0, ${MotionFlag.NoPrevious}.0, none), ${MotionFlag.NoSurface}.0, background);
    textureStore(motionOut, pixel, vec4f(velocity, flag, select(prevLinearDepth, 0.0, none)));
    textureStore(linearDepthOut, pixel, vec4f(linearDepth, 0.0, 0.0, 0.0));
}
`;

/** Byte size of `MotionConstants` (3 mat4 + 2 vec4 + 2 vec2 + vec4). */
export const MOTION_CONSTANTS_BYTES = 256;
