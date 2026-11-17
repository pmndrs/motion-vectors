import * as THREE from 'three/webgpu';
import { float, modelWorldMatrix, mrt, positionLocal, select, uniform, vec4, vertexStage } from 'three/tsl';

import { ORACLE_NO_PREVIOUS, ORACLE_VALID, oracleCode } from '../../src/webgpu/index.js';

type Vec3Node = THREE.Node<'vec3'>;
const matrixUniform = () => uniform(new THREE.Matrix4());
type MatrixUniform = ReturnType<typeof matrixUniform>;

/** The MRT slot the oracle writes; the render target's third texture carries this name. */
export const ORACLE_OUTPUT = 'oracle';

interface Tracked {
    readonly mesh: THREE.Mesh;
    readonly previous: MatrixUniform;
    readonly last: THREE.Matrix4;
    primed: boolean;
}

/**
 * Hand-written oracle motion: for every lab object, the exact previous position
 * of the surface point under each pixel, from the scene's own analytic state.
 *
 * It uses its own camera matrices (unjittered, rolled once per lab frame) and
 * its own per-object previous model matrices, so none of three's velocity
 * bookkeeping can leak into it. Deforming objects supply their previous local
 * position as a node (the deformation re-evaluated at last frame's parameters).
 *
 * Output (`oracle` MRT slot, rgba16float): `.xy` motion in three's `velocity`
 * convention, `.z` the marker (`ORACLE_VALID` / `ORACLE_NO_PREVIOUS`) and the
 * object id packed as `oracleCode(marker, id)`, `.w` the previous linear depth
 * (the contract's `.w`, 0 with no previous position). The first frame and the
 * one after {@link Oracle.reset} have no previous position.
 */
export class Oracle {
    /** This frame's unjittered view-projection. */
    readonly viewProjection = matrixUniform();
    /** Last frame's unjittered view-projection. */
    readonly previousViewProjection = matrixUniform();
    /** Last frame's view matrix (for the previous linear depth). */
    readonly previousView = matrixUniform();
    /** 1 once a previous frame exists, 0 on the first frame and after a reset. */
    readonly known = uniform(0);
    /** Scene time this frame and last frame (seconds), for deformations. */
    readonly time = uniform(0);
    readonly previousTime = uniform(0);
    /** MRT entry for anything not instrumented: static world geometry, object id 0. */
    readonly staticNode: THREE.Node<'vec4'>;

    private readonly _tracked: Tracked[] = [];
    private _primed = false;
    private readonly _view = new THREE.Matrix4();

    constructor() {
        this.staticNode = this._node(float(0), modelWorldMatrix, positionLocal);
    }

    /**
     * Gives a mesh its own oracle. The mesh must own its material.
     * @param mesh - The mesh
     * @param id - Object id in the comparison (1 … 7; 0 is static world)
     * @param previousLocal - Last frame's local position, if the mesh deforms (default: rigid)
     */
    instrument(mesh: THREE.Mesh, id: number, previousLocal?: Vec3Node): void {
        const previous = matrixUniform();
        const material = mesh.material as THREE.NodeMaterial;
        material.mrtNode = mrt({ [ORACLE_OUTPUT]: this._node(float(id), previous, previousLocal ?? positionLocal) });
        this._tracked.push({ mesh, previous, last: new THREE.Matrix4(), primed: false });
    }

    /**
     * Rolls history: call once per lab frame, after the scene's matrices update
     * and before rendering.
     * @param viewProjection - This frame's unjittered view-projection
     * @param time - This frame's scene time
     * @param view - This frame's view matrix
     */
    beginFrame(viewProjection: THREE.Matrix4, time: number, view: THREE.Matrix4): void {
        this.known.value = this._primed ? 1 : 0;
        this.previousViewProjection.value.copy(this._primed ? this.viewProjection.value : viewProjection);
        this.viewProjection.value.copy(viewProjection);
        this.previousView.value.copy(this._primed ? this._view : view);
        this._view.copy(view);
        this.previousTime.value = this._primed ? this.time.value : time;
        this.time.value = time;
        this._primed = true;
        for (const tracked of this._tracked) {
            tracked.previous.value.copy(tracked.primed ? tracked.last : tracked.mesh.matrixWorld);
            tracked.last.copy(tracked.mesh.matrixWorld);
            tracked.primed = true;
        }
    }

    /** Forgets history: the next frame reports zero motion everywhere. */
    reset(): void {
        this._primed = false;
        for (const tracked of this._tracked) tracked.primed = false;
    }

    private _node(id: THREE.Node<'float'>, previousModel: THREE.Node<'mat4'>, previousLocal: Vec3Node): THREE.Node<'vec4'> {
        // Same evaluation as three's VelocityNode: clip positions per fragment
        // from interpolated local positions. The previous local position is
        // evaluated per vertex, as the current deformation is rasterized.
        const previousInterpolated = vertexStage(previousLocal);
        const current = this.viewProjection.mul(modelWorldMatrix).mul(vec4(positionLocal, 1));
        const previousWorld = previousModel.mul(vec4(previousInterpolated, 1));
        const previous = this.previousViewProjection.mul(previousWorld);
        const valid = previous.w.greaterThan(0).and(this.known.greaterThan(0.5));
        const motion = current.xy.div(current.w).sub(previous.xy.div(previous.w));
        const previousDepth = this.previousView.mul(previousWorld).z.negate();
        return vec4(
            select(valid, motion, motion.mul(0)),
            select(valid, float(oracleCode(ORACLE_VALID, 0)), float(oracleCode(ORACLE_NO_PREVIOUS, 0))).add(id),
            select(valid, previousDepth, float(0)),
        );
    }
}
