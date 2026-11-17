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
- **`@pmndrs/motion-vectors/three`**: a drop-in `velocity` for three.js
  `WebGPURenderer` that is exact where three's is not: `positionNode`
  deformation, morph targets, `displacementMap`, several renders per frame
  ([docs/MOTION.md](docs/MOTION.md) §8).

## three

```ts
import { mrt, output } from 'three/tsl';
import { MotionVectors } from '@pmndrs/motion-vectors/three';

const motion = MotionVectors.for(renderer); // one per renderer, shared page-wide
renderer.setMRT(mrt({ output, velocity: motion.velocity })); // the output must be named `velocity`

renderer.setAnimationLoop(() => {
    motion.beginFrame(); // optional with one render per loop tick; required with more
    renderer.render(scene, camera);
});
```

- **Jitter:** set `motion.projectionMatrix` to the unjittered projection (any
  `Matrix4`, read every frame); without it the camera's own projection is used.
- **Output:** the `velocity` slot gets the full contract texel
  ([CONTRACT.md](CONTRACT.md)): `.xy` motion, `.z` a `MotionFlag` (1 exact,
  3 rigid fallback, 4 no previous position), `.w` the previous linear depth.
  Give it an `rgba16float` texture to keep `.zw`.
- **Cuts:** `motion.reset()` makes the next frame report zero motion, flagged
  as no previous position.
- **`positionNode`** graphs are re-evaluated with last frame's uniforms and
  previous position automatically. Not covered yet: `toVar()` variables made
  outside a `Fn`, textures or storage buffers that change per frame
  (docs/MOTION.md §8).

## pixi-3d (experimental)

`@pixi/3d`'s own `velocityOutput` is camera motion only: a moving object reads
zero. `MotionVectors` from `@pmndrs/motion-vectors/pixi` adds object motion for
the meshes you track, in this package's contract.

```ts
import { MotionVectors, motionOutput } from '@pmndrs/motion-vectors/pixi';

const motion = new MotionVectors();
motion.track(car); // gives the mesh its own copy of its material
app.ticker.add(() => {
    car.position.x += 0.1;
    motion.update(); // once per frame, before the view renders
});
// A pass that needs it: requiredOutputs = [motionOutput]; graph.getSceneOutput(motionOutput)
```

Rigid motion (translation, rotation, scale, parenting) is exact; skinning and
morph targets report the mesh's rigid motion. `npm run check:pixi` checks it
against analytic motion on the GPU. Why each tracked mesh needs its own material,
and what pixi-3d could add to remove that, is in
[docs/PIXI.md](docs/PIXI.md).

## Examples

```bash
npm run examples   # http://localhost:5612
```

- **three-compare**: the same scene with three's `velocity` (left) and
  `MotionVectors` (right), as motion vectors, motion blur or colour. Pause it:
  the right half goes flat grey (zero motion), the left keeps reporting motion
  on every deformed object.
- **pixi-compare**: one pixi-3d frame split at the middle, pixi-3d's
  `velocityOutput` on the left and `MotionVectors` on the right. The sliding
  box, spinning box and orbiting ball read as still on the left.

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
