import * as THREE from 'three/webgpu';

/** One case where this package's node and three's `velocity` may differ (R12). */
export interface UpstreamCase {
    /** The requirement in docs/PLAN.md §3.2 the node meets it under. */
    readonly requirement: string;
    /** What three's `velocity` gets wrong, or what it already handles. */
    readonly summary: string;
    /** The draft or filed issue (docs/upstream/), if any. */
    readonly issue: string | null;
    /**
     * The first three release whose own `velocity` is exact on this case, or
     * `null` while it isn't. The node leaves a fixed case to three, so a case
     * fixed after the oldest supported release (the peer range, `>=0.186.0`)
     * sets that floor.
     */
    readonly fixedIn: number | null;
}

/**
 * R12: what three's `velocity` gets right, release by release. The node
 * defers each case to three once three handles it, so it shrinks as three
 * improves. The lab's `upstream` experiment measures three's own velocity on
 * every case and fails when three's behaviour stops matching this table, which
 * is the signal to update `fixedIn` (and, for a newly fixed case, to retire
 * the node's code for it).
 */
export const UPSTREAM = {
    positionNode: {
        requirement: 'R6',
        summary: "`positionNode` displacement reported as motion (positionPrevious is the raw geometry).",
        issue: 'docs/upstream/three.md §1',
        fixedIn: null,
    },
    morph: {
        requirement: 'R5',
        summary: 'Morph targets reported as motion: no previous influences.',
        issue: 'docs/upstream/three.md §1',
        fixedIn: null,
    },
    displacementMap: {
        requirement: 'R9',
        summary: '`displacementMap` reported as motion.',
        issue: 'docs/upstream/three.md §1',
        fixedIn: null,
    },
    cameraHistory: {
        requirement: 'R1',
        summary: 'Camera history rolls once per display refresh, not per render.',
        issue: 'docs/upstream/three.md §2',
        fixedIn: null,
    },
    skinningHistory: {
        requirement: 'R1',
        summary: 'Previous bones (and the drawn pose) roll once per display refresh, not per render.',
        issue: 'docs/upstream/three.md §2',
        fixedIn: null,
    },
    projection: {
        requirement: 'R2',
        summary: '`setProjectionMatrix()` baked in at material build time (jitter leaks into velocity after a runtime switch).',
        issue: 'docs/upstream/three.md §2 (fixed half)',
        fixedIn: 187,
    },
    skinning: { requirement: 'R3', summary: 'Skinned meshes: previous bones applied.', issue: null, fixedIn: 185 },
    // r185 copies the previous instance matrices just before the draw (zero
    // instance motion) and leaves batched positionPrevious at the raw geometry
    // (the instance's whole offset as motion): the reason the floor is r186.
    instancing: { requirement: 'R3', summary: '`InstancedMesh`: previous instance matrices applied.', issue: null, fixedIn: 186 },
    batching: { requirement: 'R3', summary: '`BatchedMesh`: previous batch matrices applied.', issue: null, fixedIn: 186 },
} as const satisfies Record<string, UpstreamCase>;

/** A case in {@link UPSTREAM}. */
export type UpstreamCaseId = keyof typeof UPSTREAM;

/** The running three release, e.g. 187. */
export const THREE_REVISION = Number.parseInt(THREE.REVISION, 10);

/**
 * Whether three's own `velocity` handles a case in this release, so the node
 * defers to it.
 * @param id - The case
 * @param revision - The three release (default: the running one)
 * @returns `true` once three handles it
 */
export function threeHandles(id: UpstreamCaseId, revision = THREE_REVISION): boolean {
    const fixedIn: number | null = UPSTREAM[id].fixedIn;
    return fixedIn !== null && revision >= fixedIn;
}
