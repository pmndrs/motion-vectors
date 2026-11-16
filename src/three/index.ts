/**
 * `@pmndrs/motion-vectors/three`: motion vectors for three.js `WebGPURenderer`
 * (docs/PLAN.md §3.2) and the glue that publishes textures as three textures.
 */
export { MotionVectors } from './MotionVectors.js';
export { createThreeTextureAllocator, getGPUTexture } from './textures.js';
