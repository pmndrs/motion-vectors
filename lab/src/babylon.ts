import {
    Bone,
    Color3,
    Color4,
    Constants,
    CreateBox,
    CreateGround,
    CreateSphere,
    FrameGraph,
    FrameGraphClearTextureTask,
    FrameGraphCopyToBackbufferColorTask,
    FrameGraphGeometryRendererTask,
    FreeCamera,
    HemisphericLight,
    type InternalTexture,
    type Material,
    MaterialPluginBase,
    Matrix,
    Mesh,
    MorphTarget,
    MorphTargetManager,
    Scene,
    ShaderLanguage,
    Skeleton,
    StandardMaterial,
    TransformNode,
    type UniformBuffer,
    Vector3,
    VertexBuffer,
    WebGPUEngine,
} from '@babylonjs/core';

/**
 * Babylon velocity audit (EXPERIMENTAL, PLAN.md Phase 4), driven by
 * `scripts/audit-babylon.mjs` over CDP. It renders the lab's deformers objects
 * in Babylon (WebGPU, frame graph geometry renderer, linear velocity) and
 * compares Babylon's velocity, per pixel, against exact motion.
 *
 * The oracle is computed on the CPU from Babylon's own local-position output
 * (`PREPASS_LOCAL_POSITION`, the deformed local position of the surface point
 * under each pixel), the scene's analytic deformation at last frame's
 * parameters, and the lab's own copies of last frame's world and
 * view-projection matrices. Both come from the same draw, so coverage matches
 * exactly. Each run shows one object, so every covered pixel is that object's.
 *
 * Deformations are what a Babylon author writes: a material plugin moving
 * `positionUpdated` (the analogue of three's `positionNode`), and a morph
 * target. Errors are in render pixels, in three's `velocity` convention.
 */

/** Objects of the deformers scene (lab/src/scenes/deformers.ts), plus the floor for camera motion. */
const OBJECTS = ['floor', 'displaced', 'wave', 'morph', 'rigid', 'reentry', 'skinned'] as const;
type ObjectName = (typeof OBJECTS)[number];

interface AuditRequest {
    object: ObjectName;
    /** Last lab frame (default 239); frames from `start` on are measured (default 1). */
    end?: number;
    start?: number;
    /**
     * Lab frames per Babylon engine frame (`beginFrame` … `endFrame`), default 1.
     * Babylon rolls a mesh's previous world matrix once per engine frame.
     */
    rendersPerFrame?: number;
    /** Translate the camera every lab frame (the deformers scene's camera is still). */
    cameraMove?: boolean;
    /** Hide the object (`isVisible = false`) over these lab frames, inclusive, while it keeps moving. */
    hide?: [number, number];
    /**
     * Skinned limb: bone matrices as uniforms (`useTextureToStoreBoneMatrices = false`)
     * instead of Babylon's default bone texture.
     */
    boneUniforms?: boolean;
}

interface AuditFrame {
    frame: number;
    /** Covered pixels. */
    pixels: number;
    /** Mean oracle motion, render px. */
    motion: number;
    /** Mean endpoint error of Babylon's velocity, render px. */
    babylon: number;
    /** Largest per-pixel error of Babylon's velocity, render px. */
    max: number;
    /** Mean motion Babylon reports, render px. */
    reported: number;
    /** First visible frame after frames with no coverage. */
    reentered: boolean;
}

// `?width=&height=` (default 1280×720): CI runs 640×360 on SwiftShader.
const query = new URLSearchParams(location.search);
const WIDTH = Number(query.get('width') ?? 1280);
const HEIGHT = Number(query.get('height') ?? 720);
const FPS = 60;

// `?present=0` (the driver always passes it): nothing reaches a real canvas
// swapchain. Headless Linux Chrome loses the WebGPU instance at the first
// swapchain image on SwiftShader (CLAUDE.md landmine 9), and Babylon takes one
// already in `initAsync`. The canvas context hands out a plain device texture
// instead, so every platform runs the path CI runs.
if (query.get('present') === '0') {
    const configs = new WeakMap<GPUCanvasContext, { config: GPUCanvasConfiguration; texture?: GPUTexture }>();
    GPUCanvasContext.prototype.configure = function (config: GPUCanvasConfiguration) {
        configs.get(this)?.texture?.destroy();
        configs.set(this, { config });
    };
    GPUCanvasContext.prototype.getCurrentTexture = function () {
        const entry = configs.get(this);
        if (!entry) throw new Error('babylon audit: getCurrentTexture before configure');
        const { width, height } = this.canvas;
        if (!entry.texture || entry.texture.width !== width || entry.texture.height !== height) {
            entry.texture?.destroy();
            entry.texture = entry.config.device.createTexture({
                label: 'babylon-audit-backbuffer',
                size: { width, height },
                format: entry.config.format,
                usage: (entry.config.usage ?? 0) | GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }
        return entry.texture;
    };
}

//* Engine
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const engine = new WebGPUEngine(canvas, { antialias: false, adaptToDeviceRatio: false });
await engine.initAsync();
engine.setHardwareScalingLevel(1);
engine.setSize(WIDTH, HEIGHT);
const device = (engine as unknown as { _device: GPUDevice })._device;
if (!device) throw new Error('Pinned WebGPUEngine._device shape changed.');

const scene = new Scene(engine);
// Same axes and coordinates as the three lab scene.
scene.useRightHandedSystem = true;
scene.clearColor = new Color4(0.106, 0.125, 0.188, 1);
new HemisphericLight('sky', new Vector3(0, 1, 0), scene).intensity = 0.9;

const camera = new FreeCamera('camera', new Vector3(0, 3.2, 9), scene);
camera.fov = (50 * Math.PI) / 180;
camera.minZ = 0.1;
camera.maxZ = 200;

//* Displacement Plugin
/**
 * `positionUpdated.y += 0.3 · sin(5x + wave · t)` on a ground grid (local
 * normal +y): the deformers scene's `ripple`, written where a Babylon author
 * would (`CUSTOM_VERTEX_UPDATE_POSITION`, before the velocity reads it).
 */
class RipplePlugin extends MaterialPluginBase {
    time = 0;

    constructor(material: Material, private readonly _wave: number) {
        super(material, 'LabRipple', 200, { LAB_RIPPLE: false });
        this._enable(true);
    }

    override prepareDefines(defines: Record<string, unknown>): void {
        defines['LAB_RIPPLE'] = true;
    }

    override getClassName(): string {
        return 'LabRipplePlugin';
    }

    override isCompatible(shaderLanguage: ShaderLanguage): boolean {
        return shaderLanguage === ShaderLanguage.WGSL;
    }

    override getUniforms(): { ubo: { name: string; size: number; type: string }[] } {
        return {
            ubo: [
                { name: 'labTime', size: 1, type: 'float' },
                { name: 'labWave', size: 1, type: 'float' },
            ],
        };
    }

    override bindForSubMesh(uniformBuffer: UniformBuffer): void {
        uniformBuffer.updateFloat('labTime', this.time);
        uniformBuffer.updateFloat('labWave', this._wave);
    }

    override getCustomCode(shaderType: string): Record<string, string> | null {
        if (shaderType !== 'vertex') return null;
        return {
            CUSTOM_VERTEX_UPDATE_POSITION: `
#ifdef LAB_RIPPLE
positionUpdated = positionUpdated + vec3f(0.0, 0.3 * sin(5.0 * positionUpdated.x + uniforms.labWave * uniforms.labTime), 0.0);
#endif
`,
        };
    }
}

function material(name: string, color: Color3): StandardMaterial {
    const result = new StandardMaterial(name, scene);
    result.diffuseColor = color;
    result.specularColor = Color3.Black();
    result.backFaceCulling = false;
    return result;
}

//* Objects
const floor = CreateGround('floor', { width: 40, height: 40 }, scene);
floor.material = material('floor', new Color3(0.54, 0.56, 0.6));

function ripplePlane(name: string, wave: number, x: number, yaw: number): { mesh: Mesh; plugin: RipplePlugin } {
    const mesh = CreateGround(name, { width: 2.4, height: 2.4, subdivisions: 96 }, scene);
    const mat = material(name, new Color3(0.88, 0.75, 0.38));
    const plugin = new RipplePlugin(mat, wave);
    mesh.material = mat;
    // Stand the grid up: local +y (the ripple's axis) faces the camera (+z).
    mesh.rotation.set(Math.PI / 2, yaw, 0);
    mesh.position.set(x, 1.5, 0);
    return { mesh, plugin };
}
const displaced = ripplePlane('displaced', 0, -4.6, 0.55);
const wave = ripplePlane('wave', 3, -1.6, 0.2);

// Morph target: influence 0.5 + 0.5 sin(2t), bulging x and z by 1 + 0.35 sin(6y).
const morph = CreateSphere('morph', { diameter: 1.8, segments: 64 }, scene);
morph.material = material('morph', new Color3(0.82, 0.44, 0.63));
morph.position.set(1.5, 1.4, 0);
const base = morph.getVerticesData('position')!;
const bulged = new Float32Array(base.length);
for (let i = 0; i < base.length; i += 3) {
    const bulge = 1 + 0.35 * Math.sin(base[i + 1] * 6);
    bulged.set([base[i] * bulge, base[i + 1], base[i + 2] * bulge], i);
}
const manager = new MorphTargetManager(scene);
const target = new MorphTarget('bulge', 0, scene);
target.setPositions(bulged);
manager.addTarget(target);
morph.morphTargetManager = manager;
const influenceAt = (t: number) => 0.5 + 0.5 * Math.sin(t * 2);

const rigid = CreateBox('rigid', { size: 1.4 }, scene);
rigid.material = material('rigid', new Color3(0.5, 0.82, 0.5));
rigid.position.set(4.4, 1.3, 0);

const reentry = CreateSphere('reentry', { diameter: 0.9, segments: 32 }, scene);
reentry.material = material('reentry', new Color3(0.94, 0.5, 0.25));

// Skinned limb: two boxes merged into one mesh, every vertex weighted 1 to one
// bone (lower: y < 1.05, bone 0; upper: bone 1), so the oracle knows each
// pixel's bone from its local position. Bone 1 hinges at y = 1.05 on bone 0.
const lower = CreateBox('lower', { width: 0.5, height: 1, depth: 0.5 }, scene);
lower.position.y = 0.5;
const upper = CreateBox('upper', { width: 0.5, height: 1, depth: 0.5 }, scene);
upper.position.y = 1.6;
const skinned = Mesh.MergeMeshes([lower, upper], true)!;
skinned.name = 'skinned';
const limbPositions = skinned.getVerticesData(VertexBuffer.PositionKind)!;
const vertexCount = limbPositions.length / 3;
const boneIndices = new Float32Array(vertexCount * 4);
const boneWeights = new Float32Array(vertexCount * 4);
for (let v = 0; v < vertexCount; v++) {
    boneIndices[v * 4] = limbPositions[v * 3 + 1] < 1.05 ? 0 : 1;
    boneWeights[v * 4] = 1;
}
skinned.setVerticesData(VertexBuffer.MatricesIndicesKind, boneIndices);
skinned.setVerticesData(VertexBuffer.MatricesWeightsKind, boneWeights);
skinned.material = material('skinned', new Color3(0.6, 0.6, 0.9));
skinned.position.set(0, 0.3, 0);
const skeleton = new Skeleton('limb', 'limb', scene);
const hinge = Matrix.Translation(0, 1.05, 0);
const bones = [new Bone('base', skeleton, null, Matrix.Identity(), Matrix.Identity(), Matrix.Identity())];
bones.push(new Bone('tip', skeleton, bones[0], hinge.clone(), hinge.clone(), hinge.clone()));
const joints = [new TransformNode('base-joint', scene), new TransformNode('tip-joint', scene)];
joints[1].parent = joints[0];
joints[1].position.y = 1.05;
bones.forEach((bone, i) => bone.linkTransformNode(joints[i]));
skinned.skeleton = skeleton;

const meshes: Record<ObjectName, Mesh> = { floor, displaced: displaced.mesh, wave: wave.mesh, morph, rigid, reentry, skinned };

/** Poses the scene for a lab frame (lab/src/scenes/deformers.ts `update`). */
function update(frame: number, cameraMove: boolean): void {
    const t = frame / FPS;
    displaced.plugin.time = t;
    wave.plugin.time = t;
    target.influence = influenceAt(t);
    rigid.rotation.set(t * 1.2, t * 2.4, 0);
    joints[0].rotation.z = 0.3 * Math.sin(2 * t);
    joints[1].rotation.z = 0.8 * Math.sin(3 * t);
    const f = frame % 240;
    if (f < 80) reentry.position.set(-2 + (f / 80) * 13, 3.2, 1.5);
    else if (f < 120) reentry.position.set(11 + ((f - 80) / 40) * 1, 3.2 - ((f - 80) / 40) * 2.5, 1.5);
    else reentry.position.set(12 - ((f - 120) / 120) * 14, 0.7, 1.5);
    // A pure translation, ~7 px/frame at the floor's centre.
    const x = cameraMove ? -3 + frame * 0.03 : 0;
    camera.position.set(x, 3.2, 9);
    camera.setTarget(new Vector3(x, 1.4, 0));
}

/**
 * Last frame's local position of the surface point now at local `p` (the
 * deformation inverted at this frame's parameters, then re-applied at last
 * frame's). Rigid objects return `p`.
 */
function previousLocal(object: ObjectName, p: [number, number, number], t: number, previousT: number): [number, number, number] {
    if (object === 'wave') return [p[0], 0.3 * Math.sin(5 * p[0] + 3 * previousT), p[2]];
    if (object === 'morph') {
        const s = 1 + 0.35 * influenceAt(t) * Math.sin(6 * p[1]);
        const sPrevious = 1 + 0.35 * influenceAt(previousT) * Math.sin(6 * p[1]);
        return [(p[0] * sPrevious) / s, p[1], (p[2] * sPrevious) / s];
    }
    // displaced: static ripple, the same at every t.
    return p;
}

/** Babylon's row-vector transform with the perspective divide: NDC x, y. */
function project(m: Float32Array | number[], x: number, y: number, z: number): [number, number] {
    const w = x * m[3] + y * m[7] + z * m[11] + m[15];
    return [(x * m[0] + y * m[4] + z * m[8] + m[12]) / w, (x * m[1] + y * m[5] + z * m[9] + m[13]) / w];
}

//* Frame Graph
const graph = new FrameGraph(scene);
const size = { width: WIDTH, height: HEIGHT };
const color = graph.textureManager.createRenderTargetTexture('color', {
    size,
    sizeIsPercentage: false,
    // 8-bit: velocity and local position take 24 of the 32 color-attachment bytes per sample WebGPU guarantees.
    options: { createMipMaps: false, samples: 1, types: [Constants.TEXTURETYPE_UNSIGNED_BYTE], formats: [Constants.TEXTUREFORMAT_RGBA] },
});
const depth = graph.textureManager.createRenderTargetTexture('depth', {
    size,
    sizeIsPercentage: false,
    options: { createMipMaps: false, samples: 1, types: [Constants.TEXTURETYPE_FLOAT], formats: [Constants.TEXTUREFORMAT_DEPTH32_FLOAT] },
});
const clear = new FrameGraphClearTextureTask('clear', graph);
clear.targetTexture = color;
clear.depthTexture = depth;
clear.color = scene.clearColor;
clear.clearColor = true;
clear.clearDepth = true;
graph.addTask(clear);
const geometry = new FrameGraphGeometryRendererTask('geometry', graph, scene);
geometry.targetTexture = clear.outputTexture;
geometry.depthTexture = clear.outputDepthTexture;
geometry.camera = camera;
geometry.objectList = { meshes: scene.meshes, particleSystems: [] };
geometry.isMainObjectRenderer = true;
geometry.size = size;
geometry.sizeIsPercentage = false;
// Float32 so the comparison reads Babylon's value, not an f16 rounding of it.
geometry.textureDescriptions = [
    { type: Constants.PREPASS_VELOCITY_LINEAR_TEXTURE_TYPE, textureType: Constants.TEXTURETYPE_FLOAT, textureFormat: Constants.TEXTUREFORMAT_RG },
    { type: Constants.PREPASS_LOCAL_POSITION_TEXTURE_TYPE, textureType: Constants.TEXTURETYPE_FLOAT, textureFormat: Constants.TEXTUREFORMAT_RGBA },
];
graph.addTask(geometry);
const show = new FrameGraphCopyToBackbufferColorTask('show', graph);
show.sourceTexture = geometry.outputTexture;
graph.addTask(show);
await graph.buildAsync();
scene.frameGraph = graph;

function gpuTexture(handle: number): GPUTexture {
    const internal = graph.textureManager.getTextureFromHandle(handle) as InternalTexture | null;
    const resource = (internal as unknown as { _hardwareTexture?: { underlyingResource?: GPUTexture } } | null)?._hardwareTexture?.underlyingResource;
    if (!resource) throw new Error('Pinned InternalTexture._hardwareTexture.underlyingResource shape changed.');
    return resource;
}

/** Bytes per row of the velocity (rg32float) and local position (rgba32float) copies. */
const BYTES_PER_ROW = [WIDTH * 8, WIDTH * 16];
const readBuffers = BYTES_PER_ROW.map((bytes) =>
    device.createBuffer({ size: bytes * HEIGHT, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
);
/** Reads the geometry renderer's velocity (rg32float) and local position (rgba32float). */
async function readBack(): Promise<[Float32Array, Float32Array]> {
    (engine as unknown as { flushFramebuffer(): void }).flushFramebuffer();
    const encoder = device.createCommandEncoder({ label: 'babylon-audit-read' });
    [geometry.geometryLinearVelocityTexture, geometry.geometryLocalPositionTexture].forEach((handle, i) =>
        encoder.copyTextureToBuffer({ texture: gpuTexture(handle) }, { buffer: readBuffers[i], bytesPerRow: BYTES_PER_ROW[i] }, size),
    );
    device.queue.submit([encoder.finish()]);
    const out = await Promise.all(
        readBuffers.map(async (buffer) => {
            await buffer.mapAsync(GPUMapMode.READ);
            const copy = new Float32Array(buffer.getMappedRange().slice(0));
            buffer.unmap();
            return copy;
        }),
    );
    return out as [Float32Array, Float32Array];
}

//* Audit
async function run(request: AuditRequest): Promise<{ frames: AuditFrame[]; defines: Record<string, unknown> }> {
    const object = request.object;
    if (!OBJECTS.includes(object)) throw new Error(`Unknown object "${object}" (${OBJECTS.join(', ')}).`);
    const end = request.end ?? 239;
    const start = request.start ?? 1;
    const perFrame = Math.max(1, Math.round(request.rendersPerFrame ?? 1));
    const cameraMove = request.cameraMove === true;
    for (const [name, mesh] of Object.entries(meshes)) mesh.setEnabled(name === object);
    const mesh = meshes[object];
    for (const each of Object.values(meshes)) each.isVisible = true;
    const boneTexture = request.boneUniforms !== true;
    if (skeleton.useTextureToStoreBoneMatrices !== boneTexture) {
        skeleton.useTextureToStoreBoneMatrices = boneTexture;
        // The bone storage is a shader define: rebuild the limb's material.
        skinned.material!.markAsDirty(Constants.MATERIAL_AllDirtyFlag);
    }

    const frames: AuditFrame[] = [];
    let previousWorldViewProjection: Float32Array[] | null = null;
    let previousT = 0;
    let wasVisible = true;
    for (let frame = 0; frame <= end; frame++) {
        // Frame 0 gets an engine frame of its own: it primes Babylon's history
        // (a mesh keeps it across runs), then frames group `perFrame` at a time.
        const opens = frame === 0 || (frame - 1) % perFrame === 0;
        const closes = frame === 0 || frame % perFrame === 0 || frame === end;
        if (opens) engine.beginFrame();
        const t = frame / FPS;
        update(frame, cameraMove);
        mesh.isVisible = !request.hide || frame < request.hide[0] || frame > request.hide[1];
        scene.render();
        const [velocity, local] = await readBack();
        if (closes) engine.endFrame();

        // The lab's own history, one step per lab frame. Per bone for the
        // skinned limb (row vectors: local · skin · world · view-projection).
        const world = mesh.computeWorldMatrix(true).multiply(scene.getTransformMatrix());
        const skin = object === 'skinned' ? skeleton.getTransformMatrices(mesh) : null;
        const worldViewProjection = skin
            ? [0, 1].map((b) => Matrix.FromArray(skin, b * 16).multiply(world).m)
            : [world.m, world.m];
        const current = worldViewProjection.map((m) => new Float32Array(m));
        const previous = previousWorldViewProjection ?? current;
        const previousTime = previousWorldViewProjection ? previousT : t;
        previousWorldViewProjection = current;
        previousT = t;
        if (frame < start) continue;

        let pixels = 0;
        let motion = 0;
        let error = 0;
        let max = 0;
        let reported = 0;
        for (let i = 0; i < WIDTH * HEIGHT; i++) {
            const k = i * 4;
            if (!(local[k + 3] > 0)) continue;
            const p: [number, number, number] = [local[k], local[k + 1], local[k + 2]];
            const q = previousLocal(object, p, t, previousTime);
            const bone = object === 'skinned' && p[1] >= 1.05 ? 1 : 0;
            const now = project(current[bone], ...p);
            const before = project(previous[bone], ...q);
            // Oracle and Babylon in three's convention: NDC current − previous.
            // Babylon's linear velocity is 0.5 · (previous − current).
            const ox = now[0] - before[0];
            const oy = now[1] - before[1];
            const bx = -2 * velocity[i * 2];
            const by = -2 * velocity[i * 2 + 1];
            const e = Math.hypot(((ox - bx) * WIDTH) / 2, ((oy - by) * HEIGHT) / 2);
            pixels++;
            motion += Math.hypot((ox * WIDTH) / 2, (oy * HEIGHT) / 2);
            reported += Math.hypot((bx * WIDTH) / 2, (by * HEIGHT) / 2);
            error += e;
            if (e > max) max = e;
        }
        const visible = pixels > 0;
        frames.push({
            frame,
            pixels,
            motion: visible ? motion / pixels : 0,
            babylon: visible ? error / pixels : 0,
            max,
            reported: visible ? reported / pixels : 0,
            reentered: visible && !wasVisible,
        });
        wasVisible = visible;
    }
    return { frames, defines: definesOf(mesh) };
}

/** The geometry renderer's shader defines for a mesh that matter to velocity (diagnostic). */
function definesOf(mesh: Mesh): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    const wrappers = (mesh.subMeshes[0] as unknown as { _drawWrappers?: ({ defines?: Record<string, unknown> } | undefined)[] })._drawWrappers ?? [];
    for (const wrapper of wrappers) {
        const defines = wrapper?.defines;
        if (!defines || !('PREPASS' in defines)) continue;
        for (const key of ['PREPASS', 'PREPASS_VELOCITY_LINEAR', 'BONES_VELOCITY_ENABLED', 'NUM_BONE_INFLUENCERS', 'BONETEXTURE', 'MORPHTARGETS'])
            out[key] = defines[key];
    }
    return out;
}

(window as unknown as { __babylonAudit: unknown }).__babylonAudit = {
    ready: true,
    objects: OBJECTS,
    size: [WIDTH, HEIGHT],
    babylon: (await import('@babylonjs/core')).Engine.Version,
    run,
};
