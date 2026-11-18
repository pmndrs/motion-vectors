/// <reference types="@webgpu/types" preserve="true" />
/**
 * `@pmndrs/motion-vectors/three`: motion vectors for three.js `WebGPURenderer`
 * (docs/PLAN.md §3.2) and the glue that publishes textures as three textures.
 */
export { MotionFlag } from '../flags.js';
export { MotionVectors, type PreviousPosition } from './MotionVectors.js';
export { cameraMatrices, type MotionFromDepthFrame, MotionFromDepthPass, type MotionFromDepthPassOptions } from './MotionFromDepthPass.js';
export { createThreeTextureAllocator, getGPUTexture } from './textures.js';
