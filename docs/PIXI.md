# pixi-3d adapter (experimental)

`@pmndrs/motion-vectors/pixi` against `@pixi/3d` 0.1.0 (public, MIT, pixi.js
8.23), 2026-11-15. Built on pixi-3d's public API only.

## What pixi-3d gives today

`velocityOutput` (`@pixi/3d/extras`) is an `rg16float` scene output with each
opaque surface's UV delta from **camera motion alone**. Its own docs say a car
passing a still camera reads zero. It is built for motion blur and its
camera-only scope is deliberate. A temporal upscaler, TAA or a denoiser needs
object motion too, or every moving object ghosts.

## What the adapter adds

`motionOutput` is a scene output in this package's contract (CONTRACT.md: NDC
delta, current − previous, y up, `rgba16float`). Its bit runs on every opaque
draw:

- **Camera motion** for every surface, from the camera's
  `uPreviousViewProjectionMatrix`, the same source as pixi-3d's own. It is
  jitter-free because both endpoints carry this frame's jitter, which cancels.
- **Object motion** for tracked meshes. `MotionVectors.track(mesh)` gives the
  mesh a copy of its material with one more bit. That bit holds last frame's
  world transform in four `vec4` material fields and sets a private
  `mvPreviousWorld` in the vertex stage, which the output bit reads.

The output fills the contract's `.z` and `.w` too:

- **`.z`:** 1 (exact) on tracked meshes, 3 (rigid) on tracked meshes with a
  skeleton or morph weights, 2 (camera only) on everything else, and 4 (no
  previous position) on a tracked mesh's first frame, its first frame back
  after being hidden, and after `reset()`. The tracked bit carries the flag in
  a fifth material field; the output's own default is 2.
- **`.w`:** the previous clip `w`, which under a perspective projection is the
  previous view depth. An orthographic camera has none to give (pixi-3d keeps
  no previous view matrix), so `.w` is 0 there, which the contract reads as
  "not available".

`MotionVectors.update()` runs once per frame, after the app moves things and
before the view renders. It writes each tracked mesh's previous transform with
the same frame-keyed history as the three node (`src/shared/history.ts`). Every
render in the frame sees last frame's transform. A mesh that was hidden last
frame starts over at zero motion, and `reset()` zeroes the next frame.

## Results (`npm run check:pixi`, 1280×720, Apple Metal)

The check scene is a backdrop at depth 15 and a 2×2 plane at depth 10, both
facing a camera that only translates. Every point of each surface therefore
moves by one NDC delta, which is projected on the CPU from the camera's
matrices. Errors are in render pixels. Edge pixels within 3 px are left out.

| Case | Surface | Motion (px) | `MotionVectors` | pixi-3d `velocityOutput` |
|---|---|---|---|---|
| Hold | both | 0 | 0.000 | 0.000 |
| Camera moves | plane / backdrop | 27.8 / 18.6 | 0.020 / 0.005 | 0.020 / 0.005 |
| Plane moves | plane | 36.4 | **0.011** | 36.416 |
| Camera and plane move | plane / backdrop | 17.3 / 11.5 | **0.000** / 0.003 | 32.748 / 0.003 |
| Plane moves, 2 renders in the frame | plane | 27.8 | **0.020** | 27.836 |
| Plane reappears after a hidden frame | plane | 0 | 0.000 | 0.000 |
| Hold after motion | both | 0 | 0.000 | 0.000 |
| Camera moves, 2 renders in the frame | backdrop | 18.6 | 18.557 | 18.557 |

The 0.005–0.020 px residuals are one f16 step of the stored value. Metal's
f32 → f16 render-target conversion doesn't round to nearest, so pixi-3d's own
exact camera motion reads the same step. SwiftShader rounds to nearest and reads
0.000 on every case. The check's bar is 0.01 px plus one f16 step. CI runs it on
SwiftShader at 640×360, rendering off screen.

The last row is a pixi-3d limit that both outputs share (below, ask 3).

`.z` and `.w` (2026-11-17): every pixel of every case carries the expected
flag (1 on the plane, 4 on its first frame back, 2 on the untracked backdrop),
and `.w` reads the previous camera's distance to the plane (10) or backdrop
(15) exactly as f16 stores it. The check bars both.

## Limits

- **Each tracked mesh gets its own material.** A material field is the only
  per-draw value a bit can read. pixi-3d throws if a scene output's own bit
  declares `bindings.material`, and the private store index (`inst.id`) isn't
  public. So meshes that shared a material stop sharing it. They still draw in
  merged batches, because pixi-3d merges materials of one kind.
- **Skinning and morph targets** report the mesh's rigid motion, as the
  contract asks of a producer that can't compute a deformation's previous
  position. pixi-3d keeps no previous bone matrices or morph weights.
- **`BatchedMesh3D` slots** aren't tracked: their transforms live in the batch's
  store, not on a mesh. They get camera motion only.
- **One camera history per render.** Same as pixi-3d's own output.
- **A camera's first frame reads flag 2, not 4.** pixi-3d doesn't expose
  whether its camera has a previous view-projection, so the adapter can't flag
  the camera's first frame or a camera `resetHistory()`. The motion there is
  zero, so only the flag is wrong.
- **No `.w` under an orthographic camera** (above).
- Blended surfaces and materials with a template of their own write nothing,
  as in pixi-3d.

## Asks for pixi-3d

`velocityOutput` exists partly because we asked for motion vectors. Each item
below would make it a full motion source, and would retire most of this
adapter. Filed upstream as pull requests (2026-11-17):
[#50](https://github.com/pixijs/3d/pull/50) (1),
[#51](https://github.com/pixijs/3d/pull/51) (2),
[#52](https://github.com/pixijs/3d/pull/52) (3),
[#53](https://github.com/pixijs/3d/pull/53) (4),
[#54](https://github.com/pixijs/3d/pull/54) (5). When they land, the adapter
shrinks to converting pixi-3d's output to the contract and re-running
`npm run check:pixi`.

1. **Object motion in `velocityOutput`.** Every mesh's transform already lives in
   the renderable store (`uRenderableStorage[inst.id].transform`, rows stable
   per mesh). Keep last frame's copy, bound as `uPreviousRenderableStorage`
   (one `copyBufferToBuffer` before the frame's upload, or a second lane), and
   expose `previousModelMatrix` to the vertex `end` slot. Then `velocityBit`
   becomes `uPreviousViewProjectionMatrix * previousModelMatrix * position`.
   That covers every mesh with no material copies, and `BatchedMesh3D` slots
   too.
2. **Previous bones and morph weights.** The same ping-pong for the bone pit
   and the morph-weight store, so skinned and morphed meshes report their
   deformation motion. It is the case three gets wrong today (MOTION.md §2).
3. **Frame-based history.** Each camera compares a render with its own previous
   render, so a view drawn twice in a frame reads no camera motion the second
   time (the last row above: 18.6 px of motion read as 0). A
   `view.advanceHistory()`, or a history that advances once per ticker frame,
   would match how temporal consumers count frames (CONTRACT.md, "Frame").
4. **An unclamped, full-precision option.** `rg16float` clamped to ±1 UV is
   right for motion blur. Reprojection wants the contract's
   `rgba16float` and no clamp (a cut's jump is information), with `.zw` free for
   a flag and previous depth. CONTRACT.md now specifies both (2026-11-17), so
   pixi-3d could fill them in the same output: its camera history knows the
   no-previous case the adapter can't see.
5. **A documented per-draw id in the vertex scope.** `inst.id` and `renderable`
   are template locals today. A public name would let bits read per-mesh
   store data without a material field.
