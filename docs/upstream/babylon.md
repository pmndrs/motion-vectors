# Draft issues for BabylonJS/Babylon.js

Three drafts, one per issue, written to stand alone. Editor notes are in
[brackets]: remove them before filing.

[Before filing:
- 0.1.0 is on npm and the repo is public (2026-11-19, PLAN.md Phase 5), so
  the links to the audit, the repro, resolve.
- Every number is from docs/MOTION.md §11: Babylon.js **9.31.0**, WebGPU,
  Apple Metal, 1280×720, frame graph geometry renderer, linear velocity, mean
  endpoint error in render pixels against exact motion. Re-run
  `npm run audit:babylon` on the release current when filing, and update the
  version and numbers.
- The repro snippets haven't been run as written; they are cut down from
  lab/src/babylon.ts. Run each against that release before pasting.
- Babylon's forum is the usual first stop for behaviour questions; issue 1 is
  a plain bug and can go straight to GitHub. Issues 2 and 3 may fit the forum
  better.]

---

## 1. Skinned meshes have no bone motion in the velocity texture when bones are stored in a texture (the WebGPU default)

**Description**

`MaterialHelperGeometryRendering.PrepareDefines` sets `BONES_VELOCITY_ENABLED`
only when `!mesh.skeleton.isUsingTextureForMatrices`. `Skeleton.useTextureToStoreBoneMatrices`
defaults to `true`, and any engine with float textures uses it, so every
skinned mesh on WebGPU takes the other branch. That branch computes both
endpoints from the **unskinned** position (`finalWorld * positionUpdated`,
before `bonesVertex`). Bone motion is reported as zero. The pixel's velocity is
also that of the bind-pose point under the world matrix, not of the skinned
point drawn there.

The same happens with the prepass renderer and the geometry buffer renderer,
which share the define. [Check the classic `GeometryBufferRenderer` path
before filing; the audit used the frame graph.]

**Measured** (a two-bone limb, still world matrix, 239 frames)

| Bone storage | True motion | Velocity reports | Error |
|---|---|---|---|
| Texture (default) | 0.864 px | 0 | **0.864 px** (236 of 239 frames > 0.25 px) |
| Uniforms (`useTextureToStoreBoneMatrices = false`) | 0.864 px | 0.864 | 0.000 |

**Repro**

```js
// Any skinned glTF on WebGPU, rendered by a FrameGraphGeometryRendererTask with
// PREPASS_VELOCITY_LINEAR_TEXTURE_TYPE. Play an animation with the mesh's root
// still: the velocity texture is zero over the moving limbs.
mesh.skeleton.useTextureToStoreBoneMatrices; // true
// Workaround: switch to uniforms and rebuild the material.
mesh.skeleton.useTextureToStoreBoneMatrices = false;
mesh.material.markAsDirty(BABYLON.Constants.MATERIAL_AllDirtyFlag);
```

**Suggested fix**

Keep previous bone matrices in a second bone texture (or a second row range
of the same one) and read them in `prePassVertex` when bones come from a
texture, the way `mPreviousBones` works for uniforms. Until then, a warning
when velocity is requested for a skinned mesh with a bone texture would save
people a long search.

---

## 2. Object velocity history rolls once per engine frame, and survives frames the mesh wasn't drawn

**Description**

`MaterialHelperGeometryRendering.Bind` rolls a mesh's previous world matrix
when `engine.frameId` changes, and only when the mesh is bound. Two
consequences:

- **Several renders per engine frame.** A second `scene.render()` between
  `beginFrame()` and `endFrame()` (offline capture, a fixed-timestep
  catch-up, a second view) compares the live world matrix with the one from
  two renders ago. The camera is unaffected: its previous view-projection
  rolls whenever the transform changes.
- **A mesh that wasn't drawn last frame** (hidden, disabled, culled) keeps the
  world matrix of the last frame it was drawn. Its first frame back reports
  all the motion it made while hidden.

**Measured** (a rotating box, 2.4 px of motion per frame)

| Case | Error |
|---|---|
| One render per engine frame | 0.000 px |
| Two renders per engine frame | 1.21 px mean, one frame of motion on every second render |
| Hidden for 30 frames, first frame shown again | **63.25 px** (true motion 2.50) |
| Leaving and re-entering the frustum along a path | 0.000 px |

The frustum case is exact only because the bounding-sphere test binds the
mesh a frame before it covers a pixel; a teleport while culled would show the
stale matrix.

**Repro**

```js
let frame = 0;
engine.runRenderLoop(() => {
    box.rotation.y += 0.05;
    box.isVisible = frame < 60 || frame >= 90; // hidden for 30 frames
    scene.render(); // frame 90: the velocity holds 30 frames of rotation
    frame++;
});
```

**Suggested fix**

Roll every tracked mesh's history per frame, whether or not it is bound, and
treat a mesh not drawn last frame as having no previous position (zero
motion, or a flag a consumer can read) rather than reusing an old matrix.
Giving the geometry renderer an explicit "a new frame starts here" call would
fix the several-renders case the same way.

---

## 3. Deformation motion (morph targets, vertex code) is never in the velocity texture

[Feature request, not a bug: Babylon's choice is safe. It never reports false
motion, unlike three's `velocity` today.]

**Description**

Both velocity endpoints use the current `positionUpdated`, after morph targets
and `CUSTOM_VERTEX_UPDATE_POSITION`. A static displacement therefore reads
exact. Real deformation motion (animated morph influences, waves, cloth
written in vertex code) reads zero, so TAA, motion blur and temporal
upscalers reproject deformed surfaces as if they were rigid.

**Measured**

| Case | True motion | Error |
|---|---|---|
| Static vertex displacement (material plugin) | 0 | 0.000 px |
| Animated displacement `sin(5x + 3t)` | 0.158 px | 0.158 px |
| Morph target influence `0.5 + 0.5 sin(2t)` | 0.089 px | 0.089 px |

These deformations are slow; the error equals the motion, so it grows with
deformation speed.

**Suggested fix**

- **Morph targets:** keep last frame's influences per mesh and morph a
  previous position with them in `prePassVertex`.
- **Vertex code:** a `CUSTOM_VERTEX_UPDATE_PREVIOUS_POSITION` injection point
  (and its node-material equivalent) where a plugin can write last frame's
  local position, defaulting to the current one.

We built this for three.js as
[`@pmndrs/motion-vectors`](https://github.com/pmndrs/motion-vectors), which is
exact on these cases there; the audit is `lab/babylon.html` in the same repo.
