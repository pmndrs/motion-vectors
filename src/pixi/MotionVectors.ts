import type { BaseMaterial3D, Mesh3D, SceneOutput, ShaderBit } from '@pixi/3d';

import { FrameHistory } from '../shared/history.js';

/**
 * Motion vectors for `@pixi/3d` (EXPERIMENTAL).
 *
 * pixi-3d's own `velocityOutput` carries camera motion only: a moving object
 * reads zero. {@link MotionVectors} is a scene output in this package's
 * contract (CONTRACT.md: NDC delta, current − previous, y up, `rgba16float`)
 * that adds object motion for the meshes it tracks.
 *
 * - Every opaque surface gets camera motion, from pixi-3d's previous
 *   view-projection (exact, jitter-free: both endpoints carry this frame's
 *   jitter, which cancels).
 * - A tracked mesh also gets its own motion: last frame's world transform
 *   rides in a material field, so the surface point's previous position is
 *   exact for rigid motion (translation, rotation, scale, parenting).
 *
 * Tracking a mesh gives it its own copy of its material (a material field is
 * the only per-draw value a bit can read), so meshes that shared a material
 * stop sharing it. Skinning and morph targets report the rigid motion of the
 * mesh, as the contract asks of a producer that can't compute a deformation's
 * previous position. Blended surfaces write nothing, as in pixi-3d.
 *
 * pixi-3d keeps one previous view-projection per camera and compares each
 * render with the camera's own previous render, so a view drawn twice in a
 * frame reads no camera motion the second time (the same limit as its own
 * velocity output). Object history follows {@link MotionVectors.update}.
 */

/** Private variables the material bit sets and the output bit reads, declared once. */
const stateBit: ShaderBit = {
    name: 'pmndrsMotionState',
    gpu: { header: /* wgsl */ 'var<private> mvPreviousWorld : vec4<f32>;\nvar<private> mvTracked : f32 = 0.0;' },
    gl: { header: /* glsl */ 'vec4 mvPreviousWorld;\nfloat mvTracked = 0.0;' },
};

/**
 * On a tracked mesh's material: last frame's world transform, as four columns
 * (material fields can't be matrices). Runs after the transform, so `position`
 * is the same local point the draw rasterizes.
 */
const trackedBit: ShaderBit = {
    name: 'pmndrsMotionTracked',
    requires: [stateBit],
    bindings: {
        material: {
            mvPreviousModel0: 'vec4<f32>',
            mvPreviousModel1: 'vec4<f32>',
            mvPreviousModel2: 'vec4<f32>',
            mvPreviousModel3: 'vec4<f32>',
        },
    },
    gpu: {
        vertex: {
            afterTransform: /* wgsl */ `{
    let mvPreviousModel = mat4x4<f32>(material.mvPreviousModel0, material.mvPreviousModel1, material.mvPreviousModel2, material.mvPreviousModel3);
    mvPreviousWorld = mvPreviousModel * vec4<f32>(position.xyz, 1.0);
    mvTracked = 1.0;
  }`,
        },
    },
    gl: {
        vertex: {
            afterTransform: /* glsl */ `{
    mat4 mvPreviousModel = mat4(material.mvPreviousModel0, material.mvPreviousModel1, material.mvPreviousModel2, material.mvPreviousModel3);
    mvPreviousWorld = mvPreviousModel * vec4(position.xyz, 1.0);
    mvTracked = 1.0;
  }`,
        },
    },
};

/**
 * The output's own bit, on every opaque draw: camera motion of the surface
 * point, or its own previous position where the material is tracked.
 */
const outputBit: ShaderBit = {
    name: 'pmndrsMotion',
    requires: [stateBit],
    outputs: { motion: 'vec4<f32>' },
    interStage: { vMvCurrentClip: 'vec3<f32>', vMvPreviousClip: 'vec3<f32>' },
    gpu: {
        vertex: {
            end: /* wgsl */ `{
    let mvWorld = select(worldPosition, mvPreviousWorld, mvTracked > 0.5);
    output.vMvCurrentClip = clipPosition.xyw;
    output.vMvPreviousClip = (camera.uPreviousViewProjectionMatrix * mvWorld).xyw;
  }`,
        },
        fragment: {
            end: /* wgsl */ `{
    // A point behind the previous camera has no previous position: zero (CONTRACT.md).
    let mvValid = input.vMvPreviousClip.z > 1e-6;
    let mvDelta = input.vMvCurrentClip.xy / input.vMvCurrentClip.z - input.vMvPreviousClip.xy / max(input.vMvPreviousClip.z, 1e-6);
    outputs.motion = vec4<f32>(select(vec2<f32>(0.0), mvDelta, mvValid), 0.0, 0.0);
  }`,
        },
    },
    gl: {
        vertex: {
            end: /* glsl */ `{
    vec4 mvWorld = mvTracked > 0.5 ? mvPreviousWorld : worldPosition;
    vMvCurrentClip = gl_Position.xyw;
    vMvPreviousClip = (uPreviousViewProjectionMatrix * mvWorld).xyw;
  }`,
        },
        fragment: {
            end: /* glsl */ `{
    vec2 mvDelta = vMvCurrentClip.xy / vMvCurrentClip.z - vMvPreviousClip.xy / max(vMvPreviousClip.z, 1e-6);
    outputs.motion = vec4(vMvPreviousClip.z > 1e-6 ? mvDelta : vec2(0.0), 0.0, 0.0);
  }`,
        },
    },
};

/**
 * The scene output: list it in a pass's `requiredOutputs` and read it with
 * `graph.getSceneOutput(motionOutput)`. Cleared to zero (no surface: no motion).
 */
export const motionOutput: SceneOutput = {
    name: 'motion',
    format: 'rgba16float',
    bits: { opaque: outputBit },
};

interface Tracked {
    original: BaseMaterial3D;
    material: BaseMaterial3D;
}

/** Whether the mesh and every ancestor are visible. */
function isShown(node: Mesh3D): boolean {
    for (let at: { visible: boolean; parent?: unknown } | null | undefined = node; at; at = at.parent as typeof at) if (!at.visible) return false;
    return true;
}

/** Writes a column-major 4×4 into the four material fields. */
function writeColumns(material: BaseMaterial3D, elements: ArrayLike<number>): void {
    const data = material.data as Record<string, ArrayLike<number> | number | undefined>;
    for (let column = 0; column < 4; column++) {
        const o = column * 4;
        data[`mvPreviousModel${column}`] = [elements[o], elements[o + 1], elements[o + 2], elements[o + 3]];
    }
}

/**
 * Object motion for a pixi-3d view's meshes. Track the meshes that move,
 * call {@link MotionVectors.update} once per frame before the view renders, and
 * require {@link motionOutput} from a pass.
 * @example
 * ```ts
 * const motion = new MotionVectors();
 * motion.track(car);
 * app.ticker.add(() => {
 *   car.position.x += 0.1;
 *   motion.update();
 * });
 * ```
 */
export class MotionVectors {
    /** The scene output this writes into: {@link motionOutput}. */
    readonly output = motionOutput;
    private readonly _tracked = new Map<Mesh3D, Tracked>();
    private readonly _history = new FrameHistory<Float32Array>((value) => new Float32Array(value));
    private _frame = 0;
    private _epoch = 0;

    /**
     * Gives a mesh its own motion. The mesh gets its own copy of its material
     * (with the tracking bit added); {@link MotionVectors.untrack} puts the
     * original back.
     * @param mesh - The mesh
     */
    track(mesh: Mesh3D): void {
        if (this._tracked.has(mesh)) return;
        const original = mesh.material;
        const material = original.clone({ extraBits: [trackedBit] } as never);
        // Zero motion until the first update: previous = current.
        writeColumns(material, mesh.getGlobalTransform().elements);
        mesh.material = material;
        this._tracked.set(mesh, { original, material });
    }

    /**
     * Stops tracking a mesh and gives it back its original material.
     * @param mesh - The mesh
     */
    untrack(mesh: Mesh3D): void {
        const entry = this._tracked.get(mesh);
        if (!entry) return;
        mesh.material = entry.original;
        entry.material.destroy();
        this._tracked.delete(mesh);
    }

    /** Whether a mesh is tracked. */
    isTracked(mesh: Mesh3D): boolean {
        return this._tracked.has(mesh);
    }

    /**
     * Starts a frame: each tracked mesh's previous transform becomes the one
     * it was drawn with in the previous frame (zero motion if it wasn't drawn
     * then, or after {@link MotionVectors.reset}). Call it once per frame,
     * after moving things and before the view renders.
     */
    update(): void {
        this._frame++;
        for (const [mesh, entry] of this._tracked) {
            const current = mesh.getGlobalTransform().elements as unknown as Float32Array;
            const previous = this._history.before(mesh, this._frame, current, this._epoch);
            // A hidden mesh isn't drawn: no capture, so it starts over at zero motion when it shows again.
            if (isShown(mesh)) this._history.after(mesh, current);
            writeColumns(entry.material, previous);
        }
    }

    /** Zero object motion next frame (a cut). Call the camera's `resetHistory()` too. */
    reset(): void {
        this._epoch++;
    }

    /** Puts every tracked mesh's original material back. */
    dispose(): void {
        for (const mesh of [...this._tracked.keys()]) this.untrack(mesh);
    }
}
