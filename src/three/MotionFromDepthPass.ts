import * as THREE from 'three/webgpu';

import { CutDetector, type CutRequest, type CutResult } from '../webgpu/CutDetector.js';
import { type CameraMatrices, MotionFromDepth } from '../webgpu/MotionFromDepth.js';
import { createThreeTextureAllocator, getGPUTexture } from './textures.js';

/** Options of {@link MotionFromDepthPass}. */
export interface MotionFromDepthPassOptions {
    /** Also run the GPU cut detector and publish {@link MotionFromDepthPass.cutFlag} (default `true`). */
    cutDetection?: boolean;
}

/** One frame's inputs to {@link MotionFromDepthPass.update} / {@link MotionFromDepthPass.encode}. */
export interface MotionFromDepthFrame {
    /** The depth this frame rendered (a render target's `depthTexture`), any depth format. */
    depth: THREE.Texture;
    /** The camera that rendered it, with its matrices as they were for that render. */
    camera: THREE.Camera;
    /**
     * The unjittered projection (the one `MotionVectors.projectionMatrix` or
     * three's `velocity` gets). Default: `camera.projectionMatrix`, right when
     * nothing jitters the camera.
     */
    projectionMatrix?: THREE.Matrix4 | null;
    /**
     * Up to two more motion fields the cut detector tries after zero and camera
     * motion, at the depth's size: the scene's velocity MRT output, so objects
     * that moved don't count as inconsistent.
     */
    motions?: THREE.Texture[];
    /** Cut detector thresholds (`CutRequest`). */
    cut?: Pick<CutRequest, 'threshold' | 'minSurface' | 'depthTolerance'>;
    /** GPU timestamps around motion from depth and the cut detector. */
    timestampWrites?: { motion?: GPUComputePassTimestampWrites; cut?: GPUComputePassTimestampWrites };
}

/**
 * Camera motion from depth (and, by default, cut detection) for a three
 * `WebGPURenderer`: the engine-free `MotionFromDepth` and `CutDetector`, fed
 * from a three depth texture and camera, publishing three textures.
 *
 * ```ts
 * const depthMotion = new MotionFromDepthPass(renderer);
 * await depthMotion.init();
 * // each frame, after the scene rendered into `target` (with a depthTexture):
 * depthMotion.update({ depth: target.depthTexture, camera, motions: [target.textures[1]] });
 * material.colorNode = texture(depthMotion.motion);   // contract motion (CONTRACT.md)
 * // depthMotion.cutFlag: 1×1, r = reset history this frame
 * ```
 *
 * Call it while the camera still holds the projection the depth was rendered
 * with: a jittered camera passes its unjittered projection as
 * `projectionMatrix`, before clearing the jitter. Camera motion is exact for
 * static geometry and zero, flagged, for moving objects (CONTRACT.md flag 2).
 */
export class MotionFromDepthPass {
    readonly renderer: THREE.WebGPURenderer;
    /** The engine-free motion from depth this pass drives. */
    readonly fromDepth: MotionFromDepth<THREE.StorageTexture>;
    /** The engine-free cut detector, or `null` with `cutDetection: false`. */
    readonly cutDetector: CutDetector<THREE.StorageTexture> | null;

    /**
     * @param renderer - An initialized `WebGPURenderer`
     * @param options - Whether to run cut detection
     */
    constructor(renderer: THREE.WebGPURenderer, options: MotionFromDepthPassOptions = {}) {
        const backend = renderer.backend as unknown as { isWebGPUBackend?: boolean; device?: GPUDevice };
        if (!backend.isWebGPUBackend || !backend.device) throw new Error('MotionFromDepthPass: needs an initialized WebGPURenderer on the WebGPU backend.');
        if (renderer.logarithmicDepthBuffer) throw new Error('MotionFromDepthPass: a logarithmic depth buffer cannot be unprojected; use reversed depth instead.');
        this.renderer = renderer;
        const allocate = createThreeTextureAllocator(renderer);
        this.fromDepth = new MotionFromDepth(backend.device, allocate);
        this.cutDetector = options.cutDetection === false ? null : new CutDetector(backend.device, allocate);
    }

    /** Compiles the pipelines. */
    async init(): Promise<void> {
        await Promise.all([this.fromDepth.init(), this.cutDetector?.init()]);
    }

    /**
     * Sizes the products ahead of the first frame, so {@link motion} exists
     * when materials that sample it build. {@link update} calls it with the
     * depth's size; a new size replaces {@link motion} (rebind it) and forgets
     * history.
     * @param width - Render (depth) width
     * @param height - Render (depth) height
     */
    configure(width: number, height: number): void {
        this.fromDepth.configure(width, height);
    }

    /** Camera motion at the depth's resolution (rgba16float, CONTRACT.md). Exists after {@link configure} or the first frame. */
    get motion(): THREE.StorageTexture {
        return this.fromDepth.motion.handle;
    }

    /**
     * This frame's linear depth (r32float, render resolution, view-space
     * distance), once {@link update} or {@link encode} ran: what
     * `WarpingError` keeps for its occlusion mask. Raw `GPUTexture`; replaced
     * on resize.
     */
    get linearDepth(): GPUTexture {
        // encode() ends the frame, which makes this frame's depth the "previous" one.
        const depth = this.fromDepth.previousLinearDepth;
        if (!depth) throw new Error('MotionFromDepthPass: no frame encoded yet.');
        return depth;
    }

    /** The cut verdict (1×1 rgba16float: r = reset 0/1, g = inconsistent share, b = surface share, a = had a previous frame), or `null` without cut detection. */
    get cutFlag(): THREE.StorageTexture | null {
        return this.cutDetector?.flag.handle ?? null;
    }

    /**
     * Encodes and submits one frame. Call once per frame, after the scene
     * render and before anything samples {@link motion} or {@link cutFlag}.
     * @param frame - Depth, camera and optional motion candidates
     */
    update(frame: MotionFromDepthFrame): void {
        const device = this.fromDepth.device;
        const encoder = device.createCommandEncoder({ label: 'motion-from-depth' });
        this.encode(encoder, frame);
        device.queue.submit([encoder.finish()]);
    }

    /**
     * Encodes one frame into the caller's encoder (one submit with the
     * caller's own passes). Same contract as {@link update}.
     * @param encoder - Command encoder, submitted after the scene render
     * @param frame - Depth, camera and optional motion candidates
     */
    encode(encoder: GPUCommandEncoder, frame: MotionFromDepthFrame): void {
        const depth = getGPUTexture(this.renderer, frame.depth);
        this.configure(depth.width, depth.height);
        this.fromDepth.encode(encoder, depth, cameraMatrices(frame.camera, frame.projectionMatrix, this.renderer.getClearDepth()), frame.timestampWrites?.motion);
        this.cutDetector?.encode(encoder, this.fromDepth, {
            ...frame.cut,
            motions: frame.motions?.map((motion) => getGPUTexture(this.renderer, motion)),
            timestampWrites: frame.timestampWrites?.cut,
        });
        this.fromDepth.endFrame();
    }

    /**
     * Reads back the last submitted cut verdict (optional: {@link cutFlag}
     * already holds it on the GPU).
     * @returns The verdict and its counts
     */
    readCut(): Promise<CutResult> {
        if (!this.cutDetector) throw new Error('MotionFromDepthPass: cut detection is off.');
        return this.cutDetector.read();
    }

    /** Forgets the previous frame: the next one reports no previous position and a cut. */
    reset(): void {
        this.fromDepth.reset();
    }

    /** Releases every GPU resource. */
    dispose(): void {
        this.fromDepth.dispose();
        this.cutDetector?.dispose();
    }
}

/**
 * The matrices motion from depth needs, from a three camera (exported for tests).
 * @param camera - The camera, as it rendered the depth (its projection may be jittered)
 * @param projectionMatrix - Its unjittered projection (default: the camera's own)
 * @param backgroundDepth - The depth clear value (`renderer.getClearDepth()`: 0 with reversed depth)
 * @returns Column-major matrices for `MotionFromDepth.encode`
 */
export function cameraMatrices(camera: THREE.Camera, projectionMatrix: THREE.Matrix4 | null | undefined, backgroundDepth: number): CameraMatrices {
    const view = camera.matrixWorldInverse;
    return {
        viewProjection: new THREE.Matrix4().multiplyMatrices(projectionMatrix ?? camera.projectionMatrix, view).elements,
        inverseViewProjectionJittered: new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, view).invert().elements,
        view: view.elements,
        backgroundDepth,
    };
}
