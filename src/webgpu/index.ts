/**
 * `@pmndrs/motion-vectors/webgpu`: the engine-free products on a bare
 * `GPUDevice` (camera motion from depth, cut detection, warping error, the
 * warp-residual audit, the oracle comparison, luma histograms). Contract: CONTRACT.md.
 */
export { MotionAnalyzer, decodeCompare, decodeStats, RESIDUAL_SLOTS, HISTOGRAM_SLOTS } from './MotionAnalyzer.js';
export { MotionFromDepth } from './MotionFromDepth.js';
export type { CameraMatrices } from './MotionFromDepth.js';
export { CutDetector, decodeCutResult } from './CutDetector.js';
export type { CutRequest, CutResult } from './CutDetector.js';
export { WarpingError, decodeWarpingResult } from './WarpingError.js';
export type { WarpingRequest, WarpingResult } from './WarpingError.js';
export type {
    CandidateStats,
    CompareCandidateStats,
    CompareObjectStats,
    MotionFrameStats,
    ResidualRequest,
    ResidualStats,
} from './MotionAnalyzer.js';
export { COMPARE_OBJECTS, ORACLE_NO_PREVIOUS, ORACLE_VALID, oracleCode } from './shaders/motionCompare.js';
export { MotionFlag } from '../flags.js';
export { createDeviceTextureAllocator } from './textures.js';
export type { MotionTexture, MotionTextureDescriptor, MotionTextureFormat, TextureAllocator } from './textures.js';
export { MOVING_PIXELS, RESIDUAL_DEBUG_CLASSIFY, RESIDUAL_DEBUG_HEAT, RESIDUAL_DEBUG_MOTION } from './shaders/motionResidual.js';
export { HISTOGRAM_BINS } from './shaders/lumaHistogram.js';
