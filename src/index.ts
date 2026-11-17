/**
 * `@pmndrs/motion-vectors`: the backend-neutral core. The motion contract's
 * constants and CPU reference math. GPU products live in `/webgpu`, the
 * three.js glue in `/three`. Contract: CONTRACT.md.
 */

/**
 * Converts a contract motion vector (NDC delta, current − previous, y up) to a
 * UV delta for engines whose UV origin is top-left (three, raw WebGPU):
 * `previousUV = uv − motion · MOTION_SCALE`.
 */
export const MOTION_SCALE: readonly [number, number] = [0.5, -0.5];

export { MotionFlag } from './flags.js';
export { earthMovers, jensenShannon } from './math/histogram.js';
export { cameraMotionAt } from './math/reprojection.js';
export type { CameraMotionSample } from './math/reprojection.js';
