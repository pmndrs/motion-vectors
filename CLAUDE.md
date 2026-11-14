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
npm run lab         # Vite, :5611
npm run measure     # every lab experiment on the real GPU (headless Chrome + CDP)
npm run check:gpu   # conformance bars; CI runs it with --adapter swiftshader --width 640 --height 360
```

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
