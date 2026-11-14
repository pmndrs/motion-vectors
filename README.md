# @pmndrs/motion-vectors

**Experimental, unpublished.** Motion vectors for any WebGPU pipeline, and a way
to prove they are right.

Temporal techniques (upscaling, TAA, denoising, GI history) reproject last
frame through motion vectors. When those vectors are wrong, history smears or
resets. three.js's `velocity` node is exact for rigid and camera motion but
reports **deformation as motion**: a mesh displaced by `positionNode` or held on a
morph target reads 3–64+ px of motion every frame while standing still
([docs/MOTION.md](docs/MOTION.md) §7). This package defines what a motion vector
must mean and measures any source against it.

- **[CONTRACT.md](CONTRACT.md)**: the motion contract (value, jitter, frame
  semantics, unknowns) and the conformance bars.
- **`@pmndrs/motion-vectors`**: contract constants and CPU reference math.
- **`@pmndrs/motion-vectors/webgpu`**: engine-free GPU passes on a bare
  `GPUDevice`: camera motion from depth, a warp-residual audit, an exact
  oracle comparison, luma histograms.
- **`@pmndrs/motion-vectors/three`**: three.js `WebGPURenderer` glue. A
  corrected velocity node is next ([docs/PLAN.md](docs/PLAN.md) Phase 2).

## The lab

`lab/` renders scenes whose every object also writes an **oracle**: its exact
previous-frame position, from the scene's own math. Because the oracle and the
source under test describe the same rasterized point, the error is measured per
pixel, in render pixels, with no image warping in the way.

```bash
npm install
npm run lab        # http://localhost:5611 (Chrome/Edge with WebGPU)
npm run measure    # every experiment, headless over CDP, exact GPU readback (~2.5 min)
npm run check:gpu  # the conformance bars; add -- --adapter swiftshader for no GPU
```

Results land in `results/<label>/` (gitignored).

## Develop

```bash
npm run typecheck && npm test && npm run build && npm run verify:pack
```

Three.js is an optional peer (≥ 0.184); development pins **0.187.0**.
Incubated in [pmndrs/upscaler](https://github.com/pmndrs/upscaler); the history
came with it. Status, phases and gates: [docs/PLAN.md](docs/PLAN.md).

MIT licensed.
