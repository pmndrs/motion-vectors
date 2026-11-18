# @pmndrs/motion-vectors: agent and maintainer notes

Motion vectors for WebGPU pipelines: a contract (`CONTRACT.md`), engine-free
GPU passes (`src/webgpu`), three glue (`src/three`), and a lab that scores any
motion source against an exact oracle (`lab/`, `scripts/measure-motion.mjs`).
Plan, phases and gates: `docs/PLAN.md`. Findings: `docs/MOTION.md`.

## Commands

```bash
npm run typecheck   # src, test, lab
npm test            # vitest, GPU-free
npm run build       # tsc → dist/
npm run verify:pack # the tarball a consumer installs
npm run examples    # example gallery, Vite, :5612 (imports the package by name → src/)
npm run lab         # Vite, :5611
npm run measure     # every lab experiment on the real GPU (headless Chrome + CDP)
npm run check:gpu   # conformance bars; CI runs it with --adapter swiftshader --width 640 --height 360
npm run check:pixi  # pixi-3d adapter vs analytic motion (examples/pixi-compare?check=1); CI runs it on SwiftShader
npm run audit:babylon # Babylon's own velocity vs exact motion (lab/babylon.html); controls must read exact
```

## Releasing

`.github/workflows/release.yml`: manual dispatch only (version must equal
package.json's; dry run by default). Before a release, the real-GPU lab passes
on the commit (`npm run check:gpu`, `npm run check:pixi`). **Agents never
dispatch the Release workflow, push `v*` tags or run `npm publish`**; the
maintainer does. One-time setup (repo public, npm access) is in the
workflow's header and PLAN.md gate P.

## Rules

- **Measure, don't assume.** A claim about a motion source needs a lab number;
  a shader change needs a real-GPU run (`npm run measure`), not just SwiftShader.
- `src/index.ts` and `src/webgpu` stay engine-free: no `three` import outside
  `src/three` and `lab/`. Unit tests stay GPU-free; GPU checks go through the lab.
- Relative imports in `src/` carry `.js` (ESM output); `verify:pack` fails otherwise.
- `docs/` is maintained: a change to the contract, a pass or a finding updates
  `CONTRACT.md` / `docs/MOTION.md` / `docs/PLAN.md` in the same change.

## Landmines (three internals the lab depends on, verified on r187)

1. **three's camera history advances per `NodeFrame` frame**, which only three's
   own animation loop advances. The lab stops `renderer._animation` and advances
   `renderer._nodes.nodeFrame` itself; `rendersPerTick` models apps that don't.
2. **`setProjectionMatrix()`** was baked in at shader build until r186.1; r187
   reads it per frame. The lab still hands three one stable matrix before any build.
3. **MRT routes by texture name.** The lab's render target names its textures
   `output`, `velocity`, `oracle`; three only fills `positionPrevious` (skinning,
   instancing) when an MRT output is named `velocity`.
4. **Oracle materials need their own material per mesh** (`material.mrtNode`).
   A shared material would share one previous model matrix.
5. **The comparison's fixed-point sums fit a u32 up to 1920×1080**
   (`motionCompare.ts`); larger frames need a coarser fixed point.
6. `renderer.backend.get(texture).texture` is private three API
   (`src/three/textures.ts`): re-verify it on every three bump.
7. **`MotionVectors` patches two three prototypes, once per page**
   (`src/three/MotionVectors.ts`): `NodeMaterial.setupPosition` is replaced by
   a copy of r187's sequence that also carries `positionPrevious`, and
   `Node.prototype.build` gets a scoped substitution hook used only while a
   previous `positionNode` builds. On a three bump, diff three's
   `setupPosition` against the copy first.
8. **three's `frameId` isn't consecutive**: r187's `compileAsync` bumps it too.
   History keys on the order frames render (`frameOf`), never on raw ids.
9. **Headless Linux Chrome can't present a WebGPU canvas on SwiftShader**
   (no swapchain shared image; the instance is lost at the first present). CI
   runs the lab with `?present=0`; every check reads GPU buffers, not pixels.
   The pixi check renders into a `RenderTexture` for the same reason, and never
   passes pixi's `resizeTo`: that option renders to the canvas inside
   `app.init()`. In check mode any canvas present throws, so a Mac run catches it.
10. **Metal's f32 → f16 render-target conversion doesn't round to nearest.** An
    exact motion reads up to one f16 step off (0.02 px at 28 px of motion);
    pixi-3d's own exact camera velocity shows the same step. Compare stored
    `rgba16float` values with a one-step allowance, not a flat 0.01 px.
11. **pixi-3d scene-output bits can't declare `bindings.material`** (it throws),
    so per-mesh data reaches a bit only through a material bit on that mesh.
    The adapter's two bits share private vertex variables through a
    `requires`-deduplicated header bit (`pmndrsMotionState`).
