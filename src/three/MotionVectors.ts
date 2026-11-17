import * as THREE from 'three/webgpu';
import {
    batch,
    float,
    instancedMesh,
    int,
    ivec2,
    materialReference,
    modelViewMatrix,
    morphReference,
    normalLocal,
    positionLocal,
    positionPrevious,
    renderGroup,
    select,
    skinning,
    subBuild,
    textureLoad,
    uniform,
    vec2,
    vec4,
    vertexIndex,
} from 'three/tsl';

import { MotionFlag } from '../flags.js';
import { FrameHistory } from '../shared/history.js';

/**
 * three internals this module touches, typed narrowly. All of them are
 * re-verified on every three bump (docs/PLAN.md §7); r187 is the baseline.
 */
interface Builder {
    renderer: THREE.Renderer;
    object: THREE.Object3D;
    geometry: THREE.BufferGeometry;
    material: THREE.Material;
    needsPreviousData(): boolean;
    getCache(): unknown;
    setCache(cache: unknown): void;
    getCacheFromNode(node: THREE.Node, parent: boolean): unknown;
}
interface UpdateFrame {
    renderer: THREE.Renderer;
    camera: THREE.Camera;
    object: THREE.Object3D;
    frameId: number;
}
type AnyNode = THREE.Node & { build(builder: Builder, ...rest: unknown[]): unknown };
type Value = number | boolean | { clone(): Value; copy(value: Value): unknown };

//* Shared Slots (R10)
// Keyed on globalThis under Symbol.for, so two installed copies of this package
// share one registry, one patch and one substitution slot.
const GLOBAL = globalThis as unknown as Record<symbol, unknown>;
const REGISTRY = Symbol.for('@pmndrs/motion-vectors/three/registry');
const SUBSTITUTE = Symbol.for('@pmndrs/motion-vectors/three/substitute');
const PATCHED = Symbol.for('@pmndrs/motion-vectors/three/patched');

function registry(): WeakMap<THREE.Renderer, MotionVectors> {
    return (GLOBAL[REGISTRY] ??= new WeakMap()) as WeakMap<THREE.Renderer, MotionVectors>;
}

const cloneValue = (value: Value): Value => (typeof value === 'object' && value !== null ? value.clone() : value);

/**
 * A uniform holding the value something had during the previous frame, per
 * drawn object. `read` gives the live value; history hands over on the
 * {@link MotionVectors} frame (R1) and survives repeated draws (R4).
 */
class HistoryUniformNode extends THREE.UniformNode<string, Value> {
    readonly isHistoryUniformNode = true;
    private readonly _history = new FrameHistory<Value>(cloneValue);

    constructor(
        initial: Value,
        nodeType: string | null,
        private readonly _motion: MotionVectors,
        private readonly _read: (frame: UpdateFrame) => Value,
    ) {
        super(cloneValue(initial), nodeType);
        this.updateType = THREE.NodeUpdateType.OBJECT;
        this.updateAfterType = THREE.NodeUpdateType.OBJECT;
    }

    override update(nodeFrame: THREE.NodeFrame): undefined {
        const frame = nodeFrame as unknown as UpdateFrame;
        const previous = this._history.before(frame.object, this._motion.frameOf(frame), this._read(frame), this._motion.epoch);
        if (typeof previous === 'object') (this.value as { copy(value: Value): unknown }).copy(previous);
        else this.value = previous;
        return undefined;
    }

    override updateAfter(nodeFrame: THREE.NodeFrame): undefined {
        const frame = nodeFrame as unknown as UpdateFrame;
        this._history.after(frame.object, this._read(frame));
        return undefined;
    }

    /**
     * Whether the drawn object's value this frame is a real previous one (not
     * its current value standing in after a reset, a first draw, or a frame it
     * wasn't drawn in). Hands history over first, so it doesn't depend on which
     * node three updates first.
     * @param frame - The node frame of the draw
     * @returns 1 when known, else 0
     */
    knownFor(frame: UpdateFrame): number {
        this._history.before(frame.object, this._motion.frameOf(frame), this._read(frame), this._motion.epoch);
        return this._history.known(frame.object) ? 1 : 0;
    }
}

/** 1 when the drawn object has a previous value in `source`'s history, else 0 (the contract's no-previous flag). */
class KnownUniformNode extends THREE.UniformNode<'float', number> {
    constructor(private readonly _source: HistoryUniformNode) {
        super(0, 'float');
        this.updateType = THREE.NodeUpdateType.OBJECT;
    }

    override update(nodeFrame: THREE.NodeFrame): undefined {
        this.value = this._source.knownFor(nodeFrame as unknown as UpdateFrame);
        return undefined;
    }
}

/**
 * Builds `node` as it was last frame (R6, D1): in a cache of its own, so every
 * expression is generated again, while `positionLocal` reads the previous
 * position and each uniform reads its previous value.
 *
 * three's builder caches each node's output per cache, and a `positionNode`
 * graph has no per-frame copy to rebuild, so the substitution happens at build
 * time: a scoped hook in `Node.prototype.build` (installed once) swaps the
 * leaves while this node's subtree builds. Fn bodies are re-run in the fresh
 * cache, so globals they read (`time`, `positionLocal`) are swapped too.
 *
 * Not covered (R7/R8): `toVar()` variables created outside a Fn (three keeps
 * variables in its global cache), storage buffers and textures that change
 * per frame.
 */
class PreviousEvalNode extends THREE.Node {
    constructor(
        private readonly _node: AnyNode,
        private readonly _motion: MotionVectors,
    ) {
        super('vec3');
    }

    override getNodeType(): string {
        return 'vec3';
    }

    override generateNodeType(): string {
        return 'vec3';
    }

    override isCacheable(): boolean {
        return false;
    }

    override build(builder: never, ...rest: unknown[]): never {
        const b = builder as unknown as Builder;
        const outerCache = b.getCache();
        const outerSubstitute = GLOBAL[SUBSTITUTE];
        b.setCache(b.getCacheFromNode(this, false));
        GLOBAL[SUBSTITUTE] = (node: THREE.Node) => this._motion._previousOf(node);
        try {
            return this._node.build(b, ...rest) as never;
        } finally {
            GLOBAL[SUBSTITUTE] = outerSubstitute;
            b.setCache(outerCache);
        }
    }
}

//* Morph History (R5)
interface MorphPositions {
    texture: THREE.DataArrayTexture;
    width: number;
    count: number;
}
const _morphPositions = new WeakMap<THREE.BufferGeometry, MorphPositions>();

/** The geometry's position targets as one texture layer each (three packs its own privately). */
function morphPositions(geometry: THREE.BufferGeometry): MorphPositions {
    const targets = geometry.morphAttributes.position!;
    let entry = _morphPositions.get(geometry);
    if (entry && entry.count === targets.length) return entry;
    entry?.texture.dispose();
    const vertices = geometry.attributes.position.count;
    const width = Math.min(vertices, 4096);
    const height = Math.ceil(vertices / width);
    const data = new Float32Array(width * height * 4 * targets.length);
    for (let t = 0; t < targets.length; t++) {
        const offset = width * height * 4 * t;
        for (let v = 0; v < targets[t].count; v++) {
            data[offset + v * 4] = targets[t].getX(v);
            data[offset + v * 4 + 1] = targets[t].getY(v);
            data[offset + v * 4 + 2] = targets[t].getZ(v);
        }
    }
    const texture = new THREE.DataArrayTexture(data, width, height, targets.length);
    texture.type = THREE.FloatType;
    texture.needsUpdate = true;
    entry = { texture, width, count: targets.length };
    _morphPositions.set(geometry, entry);
    geometry.addEventListener('dispose', () => {
        texture.dispose();
        _morphPositions.delete(geometry);
    });
    return entry;
}

//* Position Stage
/**
 * three's `NodeMaterial.setupPosition` (r187) with `positionPrevious` carried
 * through every step three skips: morph targets (R5), `displacementMap` (R9)
 * and `positionNode` (R6). Skinning, batching and instancing already move
 * `positionPrevious` (R3), so they run as three wrote them. Active only for
 * renderers with an enabled {@link MotionVectors} while the MRT has an output
 * named `velocity` (three's own gate for previous-position data).
 */
function setupPositionWithPrevious(this: THREE.NodeMaterial, builder: Builder, original: (builder: Builder) => unknown): unknown {
    const motion = registry().get(builder.renderer);
    if (!motion?.enabled || !builder.needsPreviousData()) return original.call(this, builder);
    const { object, geometry } = builder;
    const mesh = object as THREE.Mesh & { isSkinnedMesh?: boolean; isBatchedMesh?: boolean; isInstancedMesh?: boolean; instanceMatrix?: THREE.InstancedBufferAttribute; morphTexture?: THREE.Texture | null };

    if (geometry.morphAttributes.position || geometry.morphAttributes.normal || geometry.morphAttributes.color) {
        morphReference(mesh);
        if (geometry.morphAttributes.position) motion._previousMorph(mesh, geometry);
    }
    if (mesh.isSkinnedMesh === true) skinning(mesh as unknown as THREE.SkinnedMesh);
    if ((this as THREE.NodeMaterial & { displacementMap?: THREE.Texture | null }).displacementMap) {
        // The current normal for both frames: exact unless the mesh also skins or morphs.
        const map = materialReference('displacementMap', 'texture') as unknown as THREE.Node<'vec4'>;
        const scale = materialReference('displacementScale', 'float') as unknown as THREE.Node<'float'>;
        const bias = materialReference('displacementBias', 'float') as unknown as THREE.Node<'float'>;
        const offset = normalLocal.normalize().mul(map.x.mul(scale).add(bias));
        positionLocal.addAssign(offset);
        positionPrevious.addAssign(offset);
    }
    if (mesh.isBatchedMesh) batch(mesh as unknown as THREE.BatchedMesh);
    if (mesh.isInstancedMesh && mesh.instanceMatrix && mesh.instanceMatrix.isInstancedBufferAttribute === true) instancedMesh(mesh as unknown as THREE.InstancedMesh);
    if (this.positionNode !== null) {
        const previous = subBuild(new PreviousEvalNode(this.positionNode as AnyNode, motion), 'POSITION', 'vec3');
        positionLocal.assign(subBuild(this.positionNode, 'POSITION', 'vec3'));
        positionPrevious.assign(previous);
    }
    return positionLocal;
}

/** Installs the two prototype hooks once per page (whatever copy of the package gets there first). */
function installPatches(): void {
    const material = THREE.NodeMaterial.prototype as unknown as Record<string | symbol, unknown>;
    if (material[PATCHED]) return;
    const originalSetup = material.setupPosition as (builder: Builder) => unknown;
    if (typeof originalSetup !== 'function') throw new Error('@pmndrs/motion-vectors: NodeMaterial.setupPosition not found (three changed).');
    material.setupPosition = function (this: THREE.NodeMaterial, builder: Builder) {
        return setupPositionWithPrevious.call(this, builder, originalSetup);
    };
    const node = THREE.Node.prototype as unknown as Record<string, unknown>;
    const originalBuild = node.build as (this: THREE.Node, builder: Builder, ...rest: unknown[]) => unknown;
    node.build = function (this: THREE.Node, builder: Builder, ...rest: unknown[]) {
        const substitute = GLOBAL[SUBSTITUTE] as ((node: THREE.Node) => AnyNode | null) | undefined;
        if (substitute !== undefined) {
            const replacement = substitute(this);
            if (replacement !== null) return replacement.build(builder, ...rest);
        }
        return originalBuild.call(this, builder, ...rest);
    };
    material[PATCHED] = true;
}

//* Velocity Node
/**
 * The contract texel (CONTRACT.md): `.xy` motion (NDC, current − previous,
 * jitter-free) with three's `velocity` math but frame-keyed history: the
 * camera's previous view-projection (R1) and each object's previous model
 * matrix (R4) hand over once per {@link MotionVectors} frame, and the
 * projection is the stable unjittered one (R2). `.z` is the flag: exact, or
 * rigid where the node can't follow a deformation (R8), or no previous
 * position (behind the previous camera, the camera's or object's first frame,
 * after a reset, or an object not drawn last frame; motion and depth 0 there).
 * `.w` is the previous linear depth.
 */
class MotionVelocityNode extends THREE.Node {
    readonly isMotionVelocityNode = true;
    private readonly _cameras = new FrameHistory<THREE.Matrix4>((matrix) => matrix.clone());
    private readonly _views = new FrameHistory<THREE.Matrix4>((matrix) => matrix.clone());
    private readonly _currentProjection = uniform(new THREE.Matrix4()).setGroup(renderGroup);
    private readonly _previousViewProjection = uniform(new THREE.Matrix4()).setGroup(renderGroup);
    private readonly _previousView = uniform(new THREE.Matrix4()).setGroup(renderGroup);
    private readonly _cameraKnown = uniform(0).setGroup(renderGroup);
    private readonly _viewProjection = new THREE.Matrix4();
    private readonly _previousModel: HistoryUniformNode;
    private readonly _objectKnown: KnownUniformNode;

    constructor(private readonly _motion: MotionVectors) {
        super('vec4');
        this.updateType = THREE.NodeUpdateType.RENDER;
        this.updateAfterType = THREE.NodeUpdateType.RENDER;
        this._previousModel = new HistoryUniformNode(new THREE.Matrix4(), 'mat4', _motion, (frame) => frame.object.matrixWorld as unknown as Value);
        this._objectKnown = new KnownUniformNode(this._previousModel);
    }

    private _current(camera: THREE.Camera): THREE.Matrix4 {
        const projection = this._motion.projectionMatrix ?? camera.projectionMatrix;
        this._currentProjection.value.copy(projection);
        return this._viewProjection.multiplyMatrices(projection, camera.matrixWorldInverse);
    }

    override update(nodeFrame: THREE.NodeFrame): undefined {
        const frame = nodeFrame as unknown as UpdateFrame;
        const current = this._current(frame.camera);
        const ordinal = this._motion.frameOf(frame);
        this._previousViewProjection.value.copy(this._cameras.before(frame.camera, ordinal, current, this._motion.epoch));
        this._previousView.value.copy(this._views.before(frame.camera, ordinal, frame.camera.matrixWorldInverse, this._motion.epoch));
        this._cameraKnown.value = this._cameras.known(frame.camera) ? 1 : 0;
        return undefined;
    }

    override updateAfter(nodeFrame: THREE.NodeFrame): undefined {
        const frame = nodeFrame as unknown as UpdateFrame;
        this._cameras.after(frame.camera, this._current(frame.camera));
        this._views.after(frame.camera, frame.camera.matrixWorldInverse);
        return undefined;
    }

    override setup(builder: THREE.NodeBuilder): THREE.Node {
        const current = this._currentProjection.mul(modelViewMatrix).mul(vec4(positionLocal, 1));
        const previousWorld = (this._previousModel as unknown as THREE.Node<'mat4'>).mul(vec4(positionPrevious, 1));
        const previous = this._previousViewProjection.mul(previousWorld);
        const motion = current.xy.div(current.w).sub(previous.xy.div(previous.w));
        const previousDepth = this._previousView.mul(previousWorld).z.negate();
        const valid = previous.w.greaterThan(0).and(this._cameraKnown.mul(this._objectKnown).greaterThan(0.5));
        const flag = this._motion._rigidFallback(builder as unknown as Builder) ? MotionFlag.Rigid : MotionFlag.Exact;
        return vec4(select(valid, motion, vec2(0)), select(valid, float(flag), float(MotionFlag.NoPrevious)), select(valid, previousDepth, float(0)));
    }
}

//* Public
/**
 * Motion vectors for a three `WebGPURenderer`: one instance per renderer
 * ({@link MotionVectors.for}), shared by every tool on the page (R10).
 *
 * ```ts
 * const motion = MotionVectors.for(renderer);
 * renderer.setMRT(mrt({ output, velocity: motion.velocity }));
 * // each frame, before rendering:
 * motion.projectionMatrix = unjitteredProjection; // optional, when jittering
 * motion.beginFrame();
 * ```
 *
 * The MRT output must be named `velocity`: three only computes previous
 * positions for skinned, instanced and batched meshes when an output has that
 * name (R3).
 */
export class MotionVectors {
    /**
     * The motion node for an MRT output named `velocity`: the contract texel
     * (CONTRACT.md), `.xy` NDC current − previous, `.z` a {@link MotionFlag},
     * `.w` the previous linear depth. Give the slot an `rgba16float` texture.
     */
    readonly velocity: THREE.Node<'vec4'>;
    /**
     * The unjittered projection both endpoints use (R2), or `null` for the
     * camera's own. Read every frame, so one stable matrix whose contents change
     * is fine, and so is swapping it.
     */
    projectionMatrix: THREE.Matrix4 | null = null;
    /**
     * When `false`, materials built afterwards get three's own position stage
     * (no morph, displacement or `positionNode` previous). Already-built
     * materials keep what they were built with.
     */
    enabled = true;

    private _frame = 0;
    private _driven = false;
    private _lastRaw = Number.NaN;
    private _ordinal = 0;
    private _epoch = 0;
    private readonly _proxies = new WeakMap<THREE.Node, HistoryUniformNode>();

    private constructor() {
        this.velocity = new MotionVelocityNode(this) as unknown as THREE.Node<'vec4'>;
    }

    /**
     * The renderer's instance, created on first use.
     * @param renderer - The `WebGPURenderer`
     * @returns Its {@link MotionVectors}
     */
    static for(renderer: THREE.Renderer): MotionVectors {
        installPatches();
        let motion = registry().get(renderer);
        if (motion === undefined) {
            motion = new MotionVectors();
            registry().set(renderer, motion);
        }
        return motion;
    }

    /**
     * Starts a frame: history hands over on the next render. Call once per
     * frame, before rendering, if the app renders more than once per animation
     * tick or doesn't use `renderer.setAnimationLoop` (R1). Until the first
     * call, frames follow three's own frame counter.
     */
    beginFrame(): void {
        this._frame++;
        this._driven = true;
    }

    /** Forgets all history: the next frame reports zero motion (a cut). */
    reset(): void {
        this._epoch++;
    }

    /** The current frame number. */
    get frame(): number {
        return this._frame;
    }

    /** @internal History generation, bumped by {@link reset}. */
    get epoch(): number {
        return this._epoch;
    }

    /**
     * @internal The frame a node update belongs to, as an ordinal of the frames
     * that rendered: consecutive, so a gap means "not drawn last frame". three's
     * own `frameId` isn't consecutive (r187 also bumps it while precompiling
     * pipelines), and an app may skip a `beginFrame()` without rendering.
     */
    frameOf(frame: UpdateFrame): number {
        const raw = this._driven ? this._frame : frame.frameId;
        if (raw !== this._lastRaw) {
            this._lastRaw = raw;
            this._ordinal++;
        }
        return this._ordinal;
    }

    /** @internal The previous-frame stand-in for a node in a `positionNode` graph, or `null` to keep it. */
    _previousOf(node: THREE.Node): AnyNode | null {
        if (node === (positionLocal as unknown as THREE.Node)) return positionPrevious as unknown as AnyNode;
        const uniformNode = node as THREE.UniformNode<string, Value> & { isUniformNode?: boolean; isHistoryUniformNode?: boolean };
        if (uniformNode.isUniformNode !== true || uniformNode.isHistoryUniformNode === true) return null;
        let proxy = this._proxies.get(node);
        if (proxy === undefined) {
            proxy = new HistoryUniformNode(uniformNode.value, uniformNode.nodeType ?? null, this, () => uniformNode.value);
            this._proxies.set(node, proxy);
        }
        return proxy as unknown as AnyNode;
    }

    /**
     * @internal Whether the draw being built reports rigid motion for a
     * deformation it can't follow (R8): per-instance morph influences, or any
     * deformation in a material built while {@link MotionVectors.enabled} was off.
     */
    _rigidFallback(builder: Builder): boolean {
        const mesh = builder.object as THREE.Mesh & { morphTexture?: THREE.Texture | null };
        const material = builder.material as (THREE.NodeMaterial & { displacementMap?: THREE.Texture | null }) | undefined;
        const geometry = builder.geometry;
        const morphs = geometry?.morphAttributes.position !== undefined;
        if (morphs && mesh.morphTexture) return true;
        if (this.enabled) return false;
        return morphs || !!material?.displacementMap || (material?.positionNode ?? null) !== null;
    }

    /** @internal Assigns last frame's morphed position to `positionPrevious` (R5). */
    _previousMorph(mesh: THREE.Mesh & { morphTexture?: THREE.Texture | null }, geometry: THREE.BufferGeometry): void {
        if (mesh.morphTexture) {
            // Per-instance influences (R8): rigid fallback, the current morph for both frames.
            positionPrevious.assign(positionLocal);
            return;
        }
        const { texture, width, count } = morphPositions(geometry);
        const influences = Array.from(
            { length: count },
            (_, i) => new HistoryUniformNode(0, 'float', this, (frame) => (frame.object as THREE.Mesh).morphTargetInfluences?.[i] ?? 0),
        );
        const texel = int(vertexIndex);
        const coordinate = ivec2(texel.mod(width), texel.div(width));
        let sum: THREE.Node<'float'> = float(0);
        for (const influence of influences) sum = sum.add(influence as unknown as THREE.Node<'float'>);
        if (!geometry.morphTargetsRelative) positionPrevious.mulAssign(float(1).sub(sum));
        influences.forEach((influence, i) => {
            positionPrevious.addAssign(
                (textureLoad(texture, coordinate) as unknown as { depth(layer: number): THREE.Node<'vec4'> }).depth(i).xyz.mul(influence as unknown as THREE.Node<'float'>),
            );
        });
    }
}
