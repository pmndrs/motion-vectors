# Upstream issues

Issues we found in engines while building this package. The drafts are ready
to paste, with editor notes in [brackets]; the table tracks where each one
stands.

**Filing waited until every phase was done and 0.1.0 was on npm** (decided
2026-11-19, PLAN.md Phase 5), because the drafts link this repo as the repro
and the reference fix. Both hold since 2026-11-19 (0.1.0 published, repo
public), so the drafts can be filed.

| Project | Issue | Draft | Status |
|---|---|---|---|
| three.js | `velocity` reports vertex deformation as motion (`positionNode`, morph, `displacementMap`); upstream candidate #17 | [three.md](three.md) §1 | Drafted |
| three.js | `velocity` camera history per display refresh, not per render; upstream candidate #18 (r187 fixed its projection half) | [three.md](three.md) §2 | Drafted |
| pixi-3d | `velocityOutput` asks (object motion, deformation, per-frame history, …) | — | Filed: pixijs/3d#50–54 |
| Babylon.js | Skinned meshes lose bone motion under bone textures (the WebGPU default) | [babylon.md](babylon.md) §1 | Drafted |
| Babylon.js | Object history per engine frame, and stale after frames not drawn | [babylon.md](babylon.md) §2 | Drafted |
| Babylon.js | Deformation motion (morph targets, vertex code) never in velocity (feature request) | [babylon.md](babylon.md) §3 | Drafted |

Numbers in the drafts come from docs/MOTION.md. Before filing, re-run the lab
on the engine's current release and update them.
