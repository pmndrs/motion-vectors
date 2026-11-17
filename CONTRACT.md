# Motion contract (draft)

**Status: draft, 2026-11-17** (`.z` and `.w` specified). This is what every
producer (an engine's velocity output, our three node, motion from depth) and
every consumer (the upscaler, the denoiser, GI) agrees a motion vector means. It is checked by the motion lab's
oracle (`lab/`, `scripts/measure-motion.mjs --check`). The open items at the end
must be settled before the 0.x publish (gate P in [docs/PLAN.md](docs/PLAN.md)).

## The value

| Property | Contract |
|---|---|
| Meaning | For each pixel: where the **surface point** under it was on screen in the previous frame. A point that moves because its object moves, its camera moves, or its geometry deforms (skinning, morph targets, `positionNode`, displacement) has motion. Shading that moves across a still surface (a glint, a scrolling texture, a moving shadow) has none: that is shading motion, out of contract. |
| Value | NDC delta, **current − previous**, y up: three's `velocity` convention. |
| To UV | `motionUV = motion · motionScale`, and `previousUV = uv − motionUV`. three and raw WebGPU: `motionScale = (0.5, −0.5)`. |
| Jitter | **Jitter-free.** Both endpoints use the unjittered projection. A producer that renders under a jittered projection still reports the motion of the unjittered one. |
| Frame | "Previous" is the previous **frame** the consumer advanced, not the previous render, the previous animation-loop tick, or the last time the object was drawn. Several renders in one frame (shadows, second views) don't advance it; one render per frame from a loop that ticks less often still does. |
| First frame / reset | Zero motion, never garbage. |
| Format | `rgba16float` at render resolution, the same size as the colour and depth it describes. |
| `.xy` | The motion. |
| `.z` | The flag: where the motion comes from and how far to trust it ("Flag" below). |
| `.w` | The previous linear depth ("Previous depth" below). |
| Clear | A producer clears its target to 0, which reads as "no surface". |

## Flag (`.z`)

An exact small integer (exported as `MotionFlag`). Read it rounded:
`round(texel.z)` in WGSL, `Math.round` on the CPU.

| Value | Name | Meaning | `.xy` | `.w` |
|---|---|---|---|---|
| 0 | `NoSurface` | Nothing was drawn here: the cleared target, or the depth's clear value. | Producer-defined: 0 from a rasterizer, the far plane's camera motion from motion from depth | Producer-defined |
| 1 | `Exact` | Geometric motion of the surface point: camera, object and deformation. | Exact | Exact |
| 2 | `CameraOnly` | The camera's motion of a point assumed static. Any motion of its own is missing. Motion from depth, and pixi meshes the adapter doesn't track. | Exact if the point is static | Exact if the point is static |
| 3 | `Rigid` | Camera and object motion. The deformation's own motion is missing: the deformed position stands in for both frames, so a deformation at rest reads no false motion. | Exact if the deformation is at rest | Same |
| 4 | `NoPrevious` | The point had no previous position: it was behind the previous camera, wasn't drawn last frame, or this is the first frame after a reset or ever. | 0 | 0 |

The order is by trust: only 1 means the motion is complete. Values above 4 are
reserved, and a consumer treats an unknown value as 4.

What a consumer does with it (guidance, not contract):

- **0:** nothing to reproject. Treat it like the consumer's own background.
- **1:** reproject normally.
- **2, 3:** reproject, but distrust history more: tighten the rectification,
  or lean on a depth or colour test.
- **4:** no history. Reset the pixel, as for a disocclusion. A frame that is all
  4 (after `reset()`) is a cut.

## Previous depth (`.w`)

The **linear depth** the surface point had in the **previous** frame: its
distance along the previous camera's view axis, in world units (`−z` in view
space, positive in front of the camera). It describes the point the motion
points back to, as seen by the previous camera. Perspective or orthographic, it
uses the same unit and has no depth-buffer convention (reversed, ortho, ranged)
baked in.

- **Why:** a consumer that keeps last frame's depth can test for disocclusion
  without guessing. Linearize its stored depth at `previousUV` the same way, and
  compare it with `.w`. If they agree, the history belongs to this surface point,
  even under motion along the view axis that a current-depth test misreads
  (a receding camera, an approaching object). `current depth − .w` is the
  point's motion along the view axis.
- **0** means "not available". It is always 0 with flag 4, and a producer that
  can't compute it may leave it 0 with any flag (pixi-3d under an orthographic
  camera: it keeps no previous view matrix). A consumer skips the depth test
  where `.w ≤ 0`.
- **Precision:** `rgba16float` keeps it within 2⁻¹¹ relative and up to 65504
  world units. Scenes deeper than that need a scaled unit, which the producer
  documents.

## Unknown motion

A point with no previous position (it was behind the previous camera, it
wasn't drawn last frame, or history was reset) reports **0** and flag 4. A
producer that can't compute a deformation's previous position (texture- or
storage-driven displacement, compute-written vertices) must not guess silently:
it reports the rigid motion of the object (no false motion from the deformation
itself, but none of its real motion either) and flag 3.

## Per engine

| Engine | Native convention | `motionScale` | Notes |
|---|---|---|---|
| three (WebGPU) | NDC current − previous, y up | `(0.5, −0.5)` | three's own `velocity` breaks the contract on deformation, on frame history and on jitter after a runtime switch; see "Conformance". |
| raw WebGPU (motion from depth) | the contract | `(0.5, −0.5)` | Exact for static geometry; zero for object motion. Flags 2 on surfaces, 0 on the depth's clear value (with the far plane's camera motion in `.xy`, so a sky still reprojects), 4 on the first frame and behind the previous camera. |
| pixi-3d (WebGPU) | `velocityOutput`: UV delta current − previous, v along the texture rows, camera motion only, `rg16float` clamped to ±1 | `(0.5, −0.5)` for `@pmndrs/motion-vectors/pixi`'s `motionOutput`, which is the contract | The adapter adds object motion for tracked meshes and is exact on rigid motion within one f16 step ([docs/PIXI.md](docs/PIXI.md)). Flags 1 on tracked meshes, 3 on tracked skinned or morphing ones, 2 elsewhere, 4 on a tracked mesh's first frame back. `.w` is exact under a perspective camera and 0 under an orthographic one. A second render of a camera in one frame reads no camera motion (pixi-3d's history is per render), and a camera's first frame reads flag 2 with zero motion, not 4: the adapter can't see pixi-3d's camera history. |
| Babylon (WebGPU) | linear `0.5 · (previous − current)`, render targets Y-flipped | `(−1, −1)` | From the upscaler's [engine-contract.md](https://github.com/pmndrs/upscaler/blob/main/docs/engine-contract.md) and [babylon.md](https://github.com/pmndrs/upscaler/blob/main/docs/babylon.md). The audit has not been run on Babylon yet. |

## Conformance

The lab renders an oracle next to each producer: per-object materials that
re-evaluate the scene's analytic state at the previous frame and write the exact
contract value for the same rasterized point. The comparison is per pixel, in
render pixels of endpoint error (`src/shaders/motionCompare.ts`), so it needs no
image warping and has no aliasing noise.

A conforming producer, at 1280×720:

- **Rigid, camera and object motion:** mean error below 0.01 px. three's
  `velocity` and motion from depth (static geometry) both read 0.000 px on
  every courtyard track segment, up to 33 px/frame of motion.
- **Holds:** exactly zero wherever nothing moved, including deformed geometry at rest.
- **Deformation:** below 0.01 px when the producer claims exact; flagged otherwise.
- **Jitter:** zero on a still scene with jitter on, however the jitter was switched on.
- **Frame history:** unchanged error with several renders per animation-loop tick.
- **Flag:** no covered pixel claims no previous position where the oracle has
  one, or the reverse. The one allowed exception is an object's first frame back
  after it wasn't drawn: the oracle knows where it was, a producer that keeps
  history per draw doesn't, and 4 is the honest answer.
- **Previous depth:** where the flag claims a previous position (1–3), mean
  relative `.w` error under 2⁻¹⁰ (two f16 rounding steps) against the oracle,
  on every case where the motion itself is exact.

What three's `velocity` reads against these (lab, 2026-11-14, r187; r186.1 where
it differs): 3.2 px on a *static* displaced plane, 51.9 px while a `positionNode`
moves an object and 64+ px (the cap) once it holds, 11.6 px on a morph target
held at weight 1, 11.8 px on an animated water swell, and 8.8 / 14.4 px with 2 / 4
renders per tick (r186.1: 10.6 / 23.9). A runtime jitter switch reads 0.000 px
on r187 (r186.1: 0.36 px). Rigid and
camera motion are exact. Skinning, instancing and batching use three's own
previous data and are not measured yet. Findings: [docs/MOTION.md](docs/MOTION.md) §7.

This package's three node (`MotionVectors`, 2026-11-14, r187) reads under the
0.01 px bar on every one of those cases, including 2 and 4 renders per tick
when the app calls `beginFrame()`; without it, it follows three's frame counter
and is exact at one render per tick only (docs/MOTION.md §8).

## Open before 0.x

- ~~Specify `.z` (flag values) and `.w` (linear depth: view-space or the
  engine's own).~~ Settled 2026-11-17: the flag table and view-space previous
  depth above. Every producer here writes both, and the lab and the pixi check
  bar them.
- ~~Whether "previous frame" is advanced by an explicit `beginFrame()` only, or
  also inferred when nothing drives it.~~ Both (three node): `beginFrame()`
  when called, three's frame counter otherwise. Several renders per frame need
  `beginFrame()`.
- Babylon and pixi rows: run the audit before claiming either.
