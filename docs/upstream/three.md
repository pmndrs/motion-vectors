# Draft issues for mrdoob/three.js

Two drafts, one per issue, written to stand alone. Editor notes are in
[brackets]: remove them before filing.

[Before filing:
- 0.1.0 is on npm and the repo is public (2026-11-19, PLAN.md Phase 5), so
  the links to the lab and the node, the repro and the reference fix,
  resolve.
- Every number is from the lab (docs/MOTION.md §7, §8): three **r187**, WebGPU,
  Apple Metal, 1280×720, mean endpoint error in render pixels against an exact
  oracle rendered in the same pass. Re-run `npm run measure -- --experiments
  oracle,cases,node` on the three release current when filing, and update the
  version and numbers.
- The repro snippets haven't been run as written; they are cut down from the lab.
  Run each against that release before pasting.
- Upstream candidates #17 and #18 in pmndrs/upscaler's
  docs/research/UPSTREAM-CANDIDATES.md; owners there: sunag (nodes), Mugen87.
- r187 already fixed #18's other half (`setProjectionMatrix()` is now read every
  frame instead of baked in at build time). Thank them for it in issue 2.]

---

## 1. `velocity` reports vertex deformation as motion (`positionNode`, morph targets, `displacementMap`)

**Description**

`VelocityNode` computes the previous clip position from `positionPrevious`,
which is the raw geometry position. The current position is `positionLocal`
after the material's position stage, which includes morph targets,
`displacementMap` and `material.positionNode`. When that stage moves vertices,
the two endpoints describe different points, and the difference is reported as
motion:

- A mesh displaced by a static `positionNode` reports its displacement as
  motion on every frame, though nothing moves.
- An animated deformation reports its total offset from the undeformed geometry
  instead of how far it moved this frame.
- A held morph pose keeps reporting motion after the morph stops.

Skinning, `InstancedMesh` and `BatchedMesh` are correct: they update
`positionPrevious` themselves.

**Measured** (still camera, mean error in pixels against an exact oracle)

| Case | True motion | Zero motion | `velocity` |
|---|---|---|---|
| Static `positionNode` displacement | 0 | 0 | **3.24** |
| Animated displacement `sin(5x + 3t)` | 0.16 | 0.16 | **3.14** |
| `positionNode` moves an object A→B / then holds at B | 3.12 / 0 | 3.12 / 0 | **51.9 / 64+** |
| Morph target ramping 0→1 / then held at 1 | 0.19 / 0 | 0.19 / 0 | **5.70 / 11.6** |
| Water swell (`positionNode` driven by time) | 0.39 | 0.39 | **11.8** |
| `displacementMap` (static map, scale growing; whole run) | ≤ 0.05 | ≤ 0.05 | **3.68** |
| Rigid rotating box (control) | 2.37 | 2.37 | 0.000 |

On deformed geometry, `velocity` is worse than reporting no motion at all, by one
to two orders of magnitude. TAA, `TRAANode`, temporal upscalers and temporal
denoisers reproject history along these vectors, so deformed surfaces ghost or
lose their history every frame.

**Repro**

```js
import * as THREE from 'three/webgpu';
import { mrt, normalLocal, output, positionLocal, sin, velocity } from 'three/tsl';

const renderer = new THREE.WebGPURenderer();
await renderer.init();
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
camera.position.set(0, 0, 4);

// A static displacement: nothing moves between frames.
const material = new THREE.MeshBasicNodeMaterial();
material.positionNode = positionLocal.add(normalLocal.mul(sin(positionLocal.x.mul(5)).mul(0.2)));
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2, 64, 64), material));

const target = new THREE.RenderTarget(1280, 720, { count: 2, type: THREE.HalfFloatType });
target.textures[0].name = 'output';
target.textures[1].name = 'velocity';
renderer.setMRT(mrt({ output, velocity }));
renderer.setRenderTarget(target);

let frames = 0;
renderer.setAnimationLoop(async () => {
    renderer.render(scene, camera);
    if (++frames !== 3) return;
    renderer.setAnimationLoop(null);
    const motion = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 1280, 720, 1);
    // Expected: 0 everywhere. Actual: non-zero across the displaced plane.
    console.log(Math.max(...Array.from(motion, Math.abs)));
});
```

[`readRenderTargetPixelsAsync` returns half floats as raw `Uint16` for
`HalfFloatType`. Either decode them, or use `FloatType` for the target in the repro.]

**Suggested fix**

Evaluate the material's position stage a second time for the previous frame
and use the result as the previous position:

- **Morph targets:** keep last frame's influences per mesh and morph
  `positionPrevious` with them.
- **`positionNode` and `displacementMap`:** rebuild the position graph with
  `positionLocal` replaced by `positionPrevious` and each uniform replaced by
  its previous-frame value. This is exact whenever the displacement depends
  only on uniforms (time, offsets, wave parameters), which covers most
  vertex animation.
- **A `positionPreviousNode` hook on `NodeMaterial`** for what can't be
  rebuilt (displacement read from textures or storage buffers that change per
  frame), so the author can supply last frame's value.

We built this as a drop-in `velocity` replacement,
[`@pmndrs/motion-vectors`](https://github.com/pmndrs/motion-vectors) (`MotionVectors`),
and it is exact on every case in the table. It cost +19 µs (+5.4 % of the
0.356 ms scene render) on the water scene, the one deformation-heavy case, and
was within noise elsewhere. We're happy to turn it into a
PR if one of these directions works for you.

[Their likely question: "why not just use the rigid fallback?" Rendering the
deformed position for both endpoints removes the false motion, but it misses
real deformation motion. Our node uses it only where it can't do better, and
says so through a flag in `.z`.]

---

## 2. `velocity` camera history advances per display refresh, not per render

**Description**

`VelocityNode.update()` rolls the camera's previous matrices only when
`frameId` changes. Only the renderer's internal animation loop advances
`frameId`: `render()` starts that loop, and it calls `nodeFrame.update()` once
per `requestAnimationFrame`, whether or not the app uses `setAnimationLoop`. So
the history follows display refreshes, not renders. If an app renders several
frames within one refresh (offline capture, a fixed-timestep catch-up, the same
camera rendered twice), each render after the first compares the live camera
with the camera from the refresh before last. The error grows by about one frame
of motion per extra render.

Object history doesn't have this problem: `updateAfter()` hands it over on every
draw. Skinning does: `Skinning.js` rolls `skeleton.previousBoneMatrices` and
calls `skeleton.update()` only when `frameId` changes. So a second render in
one refresh reports stale bone motion, and it also *draws* the skeleton in the
previous refresh's pose. On a limb bending 0.742 px/frame, two renders per
refresh reported 1.09 px of velocity error.

**Measured** (camera moving 7.47 px/frame, mean error in pixels)

| Renders per refresh | `velocity` error | Pixels > 0.25 px off |
|---|---|---|
| 1 | 0.000 | 0 % |
| 2 | **10.55** | 98 % |
| 4 | **23.93** | 98 % |
| 2, skinned limb (bone motion) | **1.09** | — |

**Repro**

```js
// Same setup as a velocity MRT; the camera moves every render.
renderer.setAnimationLoop(() => {
    // Two renders in one refresh, the camera moving before each.
    for (let i = 0; i < 2; i++) {
        camera.position.x += 0.05;
        camera.updateMatrixWorld();
        renderer.render(scene, camera); // the second render's velocity is wrong
    }
});
```

**Suggested fix**

Give `velocity` an explicit way to say "a new frame starts here", for example
`velocity.beginFrame()` or a renderer-level frame counter that advances once
per logical frame. Fall back to today's `frameId` keying when nothing calls it.
Our node does this (`MotionVectors.beginFrame()`), and two and four renders per
refresh then read 0.000 px. If an API change is unwelcome, documenting the
limitation on `VelocityNode` would already help.

Thanks for r187's `setProjectionMatrix()` change. Reading the matrix every frame
fixed the jitter we saw after switching TAA on at runtime (0.364 px on every
pixel of a still scene → 0.000).
