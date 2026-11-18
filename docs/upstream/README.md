# Upstream issues

Issues we found in engines while building this package. The drafts are ready
to paste, with editor notes in [brackets]; the table tracks where each one
stands.

**Filing waits until every phase is done and 0.1.0 is on npm** (decided
2026-11-19, PLAN.md Phase 5): the drafts link this repo as the repro and the
reference fix.

| Project | Issue | Draft | Status |
|---|---|---|---|
| three.js | `velocity` reports vertex deformation as motion (`positionNode`, morph, `displacementMap`); upstream candidate #17 | [three.md](three.md) §1 | Drafted |
| three.js | `velocity` camera history per display refresh, not per render; upstream candidate #18 (r187 fixed its projection half) | [three.md](three.md) §2 | Drafted |
| pixi-3d | `velocityOutput` asks (object motion, deformation, per-frame history, …) | — | Filed: pixijs/3d#50–54 |
| Babylon.js | Audit of its velocity on the deformers scene | — | Not run yet (Phase 4) |

Numbers in the drafts come from docs/MOTION.md. Before filing, re-run the lab
on the engine's current release and update them.
