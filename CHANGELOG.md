# Changelog

## 0.1.0 (alpha)

The first release. The API may still change before 1.0; the motion contract
([CONTRACT.md](CONTRACT.md)) is settled and is what consumers should code
against.

### Contract

- One texel per pixel: `.xy` motion (NDC, current − previous, three's
  `velocity` convention), `.z` a `MotionFlag` (0 no surface, 1 exact, 2 camera
  only, 3 rigid fallback, 4 no previous position), `.w` the previous
  view-space linear depth (0 when unknown).
- Frame semantics (one previous frame per `beginFrame()`), jitter-free
  motion, and the per-engine conventions, with conformance bars measured
  against an exact oracle.

### Package

- ESM, TypeScript declarations, `sideEffects: false`. One dependency,
  `@webgpu/types` (types only), so the `/webgpu` and `/three` declarations
  compile in a consumer without extra installs. `three`, `pixi.js` and
  `@pixi/3d` are optional peers.

### `@pmndrs/motion-vectors`

- Contract constants (`MOTION_SCALE`, `MotionFlag`), cut-decision math
  (`cutVerdict`), histogram distances and CPU reference reprojection.

### `@pmndrs/motion-vectors/webgpu`

Engine-free passes on a bare `GPUDevice`:

- `MotionFromDepth`: camera motion from depth and the camera matrices,
  ~23 µs at 1280×720.
- `CutDetector`: cut detection keyed on depth consistency, entirely on the
  GPU, writing a 1×1 flag a consumer reads in the same frame, ~30 µs at
  1280×720.
- `WarpingError`: a reference-free temporal-stability metric for benches.
- `MotionAnalyzer`: the audit (warp residuals, exact oracle comparison, luma
  histograms).

### `@pmndrs/motion-vectors/three`

- `MotionVectors`: a drop-in for three's `velocity` (`WebGPURenderer`, three
  `>=0.186.0`). Exact where three is not: `positionNode` deformation, morph
  targets, `displacementMap`, several renders per frame, skinning history.
  `setPreviousPosition` for deformation the node can't see, flagged rigid
  otherwise. It defers each case to three once three handles it.
- `MotionFromDepthPass`: motion from depth and cut detection from a three
  depth texture and camera.

### `@pmndrs/motion-vectors/pixi` (experimental)

- `MotionVectors` and `motionOutput` for `@pixi/3d`: camera and object motion
  for tracked meshes, in the contract.

### Audits

- three's `velocity` (docs/MOTION.md §7–§8) and Babylon.js's (§11) measured
  against the oracle; the upstream issue drafts are in `docs/upstream/`.
