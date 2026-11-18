import { RESIDUAL_FIXED_POINT } from './shaders/motionResidual.js';
import {
    WARPING_CONSTANTS_BYTES,
    WARPING_ERROR_SHADER,
    WARPING_FLAG_DEPTH_TEST,
    WARPING_FLAG_HEAT,
    WARPING_WORDS,
} from './shaders/warpingError.js';

/** One frame's inputs to {@link WarpingError.encode}. */
export interface WarpingRequest {
    /** This frame's image (any filterable color format); sets the measured size. The next frame warps from a copy of it, so it needs `COPY_SRC`. */
    current: GPUTexture;
    /** The motion field to judge, in the contract format (CONTRACT.md): NDC current − previous, `.z` flag, `.w` previous linear depth. */
    motion: GPUTexture;
    /**
     * This frame's linear depth (r32float, `COPY_SRC`), e.g. `MotionFromDepth`'s.
     * Kept for the next frame's occlusion mask; without it every on-screen
     * pixel with a previous position is scored, disocclusions included.
     */
    linearDepth?: GPUTexture | null;
    /** Relative tolerance of the occlusion mask (default 0.05). */
    depthTolerance?: number;
    /** Error above which a pixel counts in {@link WarpingResult.over} (default 0.02, conditioned luma). */
    threshold?: number;
    /** An `rgba8unorm` storage texture of the image's size to write a heat map into (error ×8 grey; blue = masked). */
    heat?: GPUTexture;
    /** GPU timestamps around the pass. */
    timestampWrites?: GPUComputePassTimestampWrites;
}

/** One frame's warping error, read back. Shares are of all pixels. */
export interface WarpingResult {
    /** False on the first frame and after a reset: nothing was measured. */
    hasPrevious: boolean;
    /** Mean conditioned-luma error over the scored pixels: the metric. */
    meanError: number;
    /** Share of pixels scored. */
    scored: number;
    /** Share of pixels scored with an error above the threshold. */
    over: number;
    /** Share of pixels the occlusion mask left out (disoccluded, or the motion's `.w` didn't match). */
    occluded: number;
    /** Share of pixels whose previous position is off-screen. */
    offscreen: number;
    /** Share of pixels the motion flags as having no previous position. */
    noPrevious: number;
    /** Raw counts: error sum (×1024 fixed point), scored, over, occluded, offscreen, noPrevious, pixels. */
    counts: { errorSum: number; scored: number; over: number; occluded: number; offscreen: number; noPrevious: number; pixels: number };
}

/**
 * Warping error on a bare `GPUDevice`: a reference-free temporal-stability
 * metric for benches (docs/MOTION.md §9). Every frame it warps the previous
 * image to the current one through a motion field and averages the luma
 * difference, leaving disoccluded pixels out with a depth mask. A temporal
 * pass that flickers, smears or ghosts under motion reads higher; a correct,
 * stable one approaches the scene's own shading change. Measurement only: one
 * compute pass plus two texture copies a frame, and a readback.
 *
 * Per frame, after the image and motion exist: {@link encode}, submit, then
 * optionally {@link read}. The class keeps its own copy of the previous image
 * and depth.
 */
export class WarpingError {
    readonly device: GPUDevice;

    private _pipeline: GPUComputePipeline | null = null;
    private readonly _constants: GPUBuffer;
    private readonly _counters: GPUBuffer;
    private readonly _result: GPUBuffer;
    private readonly _readback: GPUBuffer;
    private readonly _sampler: GPUSampler;
    private readonly _dummyDepth: GPUTexture;
    private readonly _dummyHeat: GPUTexture;
    private _previous: GPUTexture | null = null;
    private _previousDepth: GPUTexture | null = null;
    private _hasPrevious = false;
    private _hasPreviousDepth = false;
    private _measured = false;
    private _reading = false;

    /** @param device - The device the images live on */
    constructor(device: GPUDevice) {
        this.device = device;
        this._constants = device.createBuffer({ label: 'motion-warp-constants', size: WARPING_CONSTANTS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._counters = device.createBuffer({
            label: 'motion-warp-counters',
            size: WARPING_WORDS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._result = device.createBuffer({ label: 'motion-warp-result', size: WARPING_WORDS * 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this._readback = device.createBuffer({ label: 'motion-warp-readback', size: WARPING_WORDS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this._sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        const dummy = (label: string, format: GPUTextureFormat, usage: number) =>
            device.createTexture({ label, size: { width: 1, height: 1 }, format, usage });
        this._dummyDepth = dummy('motion-warp-dummy-depth', 'r32float', GPUTextureUsage.TEXTURE_BINDING);
        this._dummyHeat = dummy('motion-warp-dummy-heat', 'rgba8unorm', GPUTextureUsage.STORAGE_BINDING);
    }

    /** Compiles the pipeline. */
    async init(): Promise<void> {
        const label = 'motion-warp';
        this._pipeline = await this.device.createComputePipelineAsync({
            label,
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ label, code: WARPING_ERROR_SHADER }), entryPoint: 'main' },
        });
    }

    /** Forgets the previous frame: the next one measures nothing and starts over (a cut, a resize, a new run). */
    reset(): void {
        this._hasPrevious = false;
        this._hasPreviousDepth = false;
    }

    /**
     * Measures this frame against the previous one, then keeps this frame's
     * image (and depth) for the next.
     * @param encoder - Command encoder
     * @param request - Image, motion field, optional depth and options
     */
    encode(encoder: GPUCommandEncoder, request: WarpingRequest): void {
        if (!this._pipeline) throw new Error('WarpingError: await init() first.');
        const { current, motion } = request;
        const depth = request.linearDepth ?? null;
        if (depth && depth.format !== 'r32float') throw new Error('WarpingError: linearDepth must be r32float.');
        if (depth && (depth.width !== motion.width || depth.height !== motion.height)) throw new Error('WarpingError: linearDepth must match the motion size.');
        if (request.heat && (request.heat.width !== current.width || request.heat.height !== current.height))
            throw new Error('WarpingError: heat must match the image size.');
        this._prepare(current, depth);

        this._measured = this._hasPrevious;
        encoder.clearBuffer(this._counters);
        if (this._hasPrevious) {
            const depthTest = depth !== null && this._hasPreviousDepth;
            const buffer = new ArrayBuffer(WARPING_CONSTANTS_BYTES);
            const f32 = new Float32Array(buffer);
            const u32 = new Uint32Array(buffer);
            const depthSize = depthTest ? [depth.width, depth.height] : [1, 1];
            f32.set([current.width, current.height, 1 / current.width, 1 / current.height, motion.width, motion.height, depthSize[0], depthSize[1]]);
            f32[8] = request.depthTolerance ?? 0.05;
            f32[9] = request.threshold ?? 0.02;
            u32[10] = (depthTest ? WARPING_FLAG_DEPTH_TEST : 0) | (request.heat ? WARPING_FLAG_HEAT : 0);
            this.device.queue.writeBuffer(this._constants, 0, buffer);

            const single = (texture: GPUTexture) => texture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
            const pass = encoder.beginComputePass({ label: 'motion-warp', timestampWrites: request.timestampWrites });
            pass.setPipeline(this._pipeline);
            pass.setBindGroup(
                0,
                this.device.createBindGroup({
                    layout: this._pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this._constants } },
                        { binding: 1, resource: single(current) },
                        { binding: 2, resource: single(this._previous!) },
                        { binding: 3, resource: this._sampler },
                        { binding: 4, resource: single(motion) },
                        { binding: 5, resource: single(depthTest ? this._previousDepth! : this._dummyDepth) },
                        { binding: 6, resource: single(request.heat ?? this._dummyHeat) },
                        { binding: 7, resource: { buffer: this._counters } },
                    ],
                }),
            );
            pass.dispatchWorkgroups(Math.ceil(current.width / 8), Math.ceil(current.height / 8));
            pass.end();
        }
        encoder.copyBufferToBuffer(this._counters, 0, this._result, 0, WARPING_WORDS * 4);

        //* Keep This Frame
        encoder.copyTextureToTexture({ texture: current }, { texture: this._previous! }, { width: current.width, height: current.height });
        if (depth) encoder.copyTextureToTexture({ texture: depth }, { texture: this._previousDepth! }, { width: depth.width, height: depth.height });
        this._hasPrevious = true;
        this._hasPreviousDepth = depth !== null;
    }

    /**
     * Reads back the last submitted frame's result (submit it first). One read at a time.
     * @returns The warping error and its counts
     */
    async read(): Promise<WarpingResult> {
        if (this._reading) throw new Error('WarpingError: a read is already in flight.');
        this._reading = true;
        try {
            const encoder = this.device.createCommandEncoder({ label: 'motion-warp-readback' });
            encoder.copyBufferToBuffer(this._result, 0, this._readback, 0, WARPING_WORDS * 4);
            this.device.queue.submit([encoder.finish()]);
            await this._readback.mapAsync(GPUMapMode.READ);
            const words = new Uint32Array(this._readback.getMappedRange().slice(0));
            this._readback.unmap();
            return decodeWarpingResult(words, this._measured);
        } finally {
            this._reading = false;
        }
    }

    /** Releases every GPU resource. */
    dispose(): void {
        this._previous?.destroy();
        this._previousDepth?.destroy();
        for (const texture of [this._dummyDepth, this._dummyHeat]) texture.destroy();
        for (const buffer of [this._constants, this._counters, this._result, this._readback]) buffer.destroy();
    }

    /** (Re)allocates the history copies when the image or depth changes size or format; a change forgets the previous frame. */
    private _prepare(current: GPUTexture, depth: GPUTexture | null): void {
        const previous = this._previous;
        if (!previous || previous.width !== current.width || previous.height !== current.height || previous.format !== current.format) {
            previous?.destroy();
            this._previous = this.device.createTexture({
                label: 'motion-warp-previous',
                size: { width: current.width, height: current.height },
                format: current.format,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this._hasPrevious = false;
        }
        const previousDepth = this._previousDepth;
        if (depth && (!previousDepth || previousDepth.width !== depth.width || previousDepth.height !== depth.height)) {
            previousDepth?.destroy();
            this._previousDepth = this.device.createTexture({
                label: 'motion-warp-previous-depth',
                size: { width: depth.width, height: depth.height },
                format: 'r32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this._hasPreviousDepth = false;
        }
    }
}

/**
 * Decodes the warping counters (exported for tests).
 * @param words - sum lo, sum hi, scored, over, occluded, offscreen, noPrevious, pixels
 * @param hasPrevious - Whether the frame was measured
 * @returns The warping error and its counts
 */
export function decodeWarpingResult(words: Uint32Array, hasPrevious: boolean): WarpingResult {
    const [lo, hi, scored, over, occluded, offscreen, noPrevious, pixels] = words;
    const errorSum = hi * 2 ** 32 + lo;
    const share = (value: number) => (pixels ? value / pixels : 0);
    return {
        hasPrevious,
        meanError: scored ? errorSum / RESIDUAL_FIXED_POINT / scored : 0,
        scored: share(scored),
        over: share(over),
        occluded: share(occluded),
        offscreen: share(offscreen),
        noPrevious: share(noPrevious),
        counts: { errorSum, scored, over, occluded, offscreen, noPrevious, pixels },
    };
}
