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
  `GPUDevice`: camera motion from depth, cut detection, warping error, a
  warp-residual audit, an exact oracle comparison, luma histograms.
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

### Motion from depth and cut detection

`MotionFromDepthPass` runs camera motion from depth and the GPU cut detector
(below) from a three depth texture and camera, and publishes both as three
textures. Use it for a reset signal, or as a velocity source where a scene has
no velocity output (static geometry only: moving objects read zero, flagged).

```ts
import { MotionFromDepthPass } from '@pmndrs/motion-vectors/three';

const depthMotion = new MotionFromDepthPass(renderer);
await depthMotion.init();

// Per frame, after rendering into `target` (which has a depthTexture), while the
// camera still holds the projection it rendered with:
depthMotion.update({
    depth: target.depthTexture,
    camera,
    projectionMatrix: unjittered, // only when jittering
    motions: [target.textures[1]], // the velocity output, optional
});
// texture(depthMotion.motion): camera motion in the contract format
// depthMotion.cutFlag: 1×1, .r = 1 resets history this frame
```

`encode(encoder, frame)` does the same into your own command encoder.
Reversed depth works; a logarithmic depth buffer can't be unprojected and is
refused.

## Cut detection (raw WebGPU)

`CutDetector` tells a temporal pass when its history stopped being valid: a
camera cut, a teleport, a scene switch. It runs on the GPU with no readback and
writes a 1×1 `rgba16float` flag the consumer reads in the same frame (`r` = 1:
reset history).

```ts
import { CutDetector, MotionFromDepth } from '@pmndrs/motion-vectors/webgpu';

const fromDepth = new MotionFromDepth(device);
const cuts = new CutDetector(device);
await Promise.all([fromDepth.init(), cuts.init()]);
fromDepth.configure(width, height);

// Per frame, after the scene render:
fromDepth.encode(encoder, depthTexture, { viewProjection, inverseViewProjectionJittered, view });
cuts.encode(encoder, fromDepth, { motions: [velocityTexture] }); // velocity is optional
fromDepth.endFrame();
// Your temporal pass binds cuts.flag.gpu and resets where .r is 1.
```

The signal is the share of surface pixels no motion (zero, camera, the
engine's velocity) maps onto a matching previous depth. Lighting changes,
fades and flashes don't touch depth, so they never fire; on the lab's 14-event
cut timeline it fires on every cut whose history is really gone and on nothing
else ([docs/MOTION.md](docs/MOTION.md) §9). It needs depth: inputs without it
fall back to the luma histogram (`jensenShannon`, `earthMovers`). A three app
uses `MotionFromDepthPass` (above). `cuts.read()` reads the verdict back if you
reset on the CPU instead.

## Warping error (benches)

`WarpingError` measures temporal stability without a reference image. Each
frame it warps the previous image onto the current one through a motion field
and averages the luma difference. A depth mask leaves out disoccluded pixels.
Flicker, smearing and ghosting under motion read higher; a stable output
approaches the scene's own shading change. It's a measurement for benches and
A/B runs, not a runtime feature.

```ts
import { WarpingError } from '@pmndrs/motion-vectors/webgpu';

const warping = new WarpingError(device);
await warping.init();

// Per frame, after the image you're judging (e.g. an upscaler's output) exists:
warping.encode(encoder, { current: outputTexture, motion: motionTexture, linearDepth });
device.queue.submit([encoder.finish()]);
const { meanError, scored, occluded } = await warping.read();
```

- `motion` is any field in the contract format ([CONTRACT.md](CONTRACT.md)).
  Its `.w` is what the mask compares against, so use a field that fills it
  (`MotionVectors`, `MotionFromDepthPass.motion`).
- `linearDepth` is this frame's linear depth, e.g. `MotionFromDepthPass.linearDepth`.
  Without it nothing is masked.
- The image can be at display resolution and the motion and depth at render
  resolution.
- The class keeps its own copy of the previous image and depth, so `current`
  needs `COPY_SRC`. `reset()` after a cut. `heat` takes an `rgba8unorm`
  storage texture for a per-pixel map.
- Read `meanError` together with `scored`: a field that leaves more pixels to
  the mask covers less of the image.

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
