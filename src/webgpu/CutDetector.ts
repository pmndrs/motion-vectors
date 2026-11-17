import { CUT_DEPTH_TOLERANCE, CUT_MIN_SURFACE, CUT_THRESHOLD, type CutVerdict } from '../math/cut.js';
import type { MotionFromDepth } from './MotionFromDepth.js';
import { CUT_CONSTANTS_BYTES, CUT_DETECTION_SHADER, CUT_RESULT_WORDS } from './shaders/cutDetection.js';
import { createDeviceTextureAllocator, type MotionTexture, type TextureAllocator } from './textures.js';

/** Options of {@link CutDetector.encode}. */
export interface CutRequest {
    /** Up to two contract motion fields at render resolution (an engine's velocity), tried after zero and camera motion. */
    motions?: GPUTexture[];
    /** Inconsistent surface share above which the frame is a cut (default 0.2, `CUT_THRESHOLD`). */
    threshold?: number;
    /** Smallest surface share that gets a verdict (default 0.05, `CUT_MIN_SURFACE`). */
    minSurface?: number;
    /** Relative tolerance of the previous-depth match (default 0.05, `CUT_DEPTH_TOLERANCE`). */
    depthTolerance?: number;
    /** GPU timestamps around the detector's one compute pass. */
    timestampWrites?: GPUComputePassTimestampWrites;
}

/** A frame's verdict read back from the GPU, with the raw counts. */
export interface CutResult extends CutVerdict {
    /** Surface pixels no candidate reprojects onto a matching previous depth. */
    inconsistentPixels: number;
    /** Pixels with a surface (camera motion's flag ≠ 0). */
    surfacePixels: number;
    /** Render pixels. */
    pixels: number;
}

/**
 * Cut detection on a bare `GPUDevice`, entirely on the GPU: one compute pass
 * per frame writes {@link flag}, a 1×1 rgba16float texture a consumer samples
 * in the same frame to reset its history (r = reset 0/1, g = inconsistent
 * share, b = surface share, a = a previous frame existed). The signal is the
 * share of surface pixels no motion candidate maps onto a matching previous
 * depth (MOTION.md §7): 13 of 14 events right at 0.2 in the lab, no false
 * positives. It needs depth; depth-less inputs fall back to the luma histogram.
 *
 * Per frame, after {@link MotionFromDepth.encode} and before its `endFrame()`:
 * {@link encode}. {@link read} is optional, for apps that reset on the CPU.
 */
export class CutDetector<THandle = GPUTexture> {
    readonly device: GPUDevice;

    private _count: GPUComputePipeline | null = null;
    private _resolve: GPUComputePipeline | null = null;
    private readonly _constants: GPUBuffer;
    private readonly _counters: GPUBuffer;
    private readonly _result: GPUBuffer;
    private readonly _readback: GPUBuffer;
    private readonly _flag: MotionTexture<THandle>;
    private readonly _dummy: GPUTexture;
    private _pixels = 0;
    private _reading = false;

    /**
     * @param device - The device the engine renders with
     * @param allocate - Allocator for the flag texture an engine samples
     */
    constructor(device: GPUDevice, allocate?: TextureAllocator<THandle>) {
        this.device = device;
        const allocator = allocate ?? (createDeviceTextureAllocator(device) as unknown as TextureAllocator<THandle>);
        this._constants = device.createBuffer({
            label: 'motion-cut-constants',
            size: CUT_CONSTANTS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Zero at creation; the resolve zeroes them after every read.
        this._counters = device.createBuffer({ label: 'motion-cut-counters', size: 8, usage: GPUBufferUsage.STORAGE });
        this._result = device.createBuffer({
            label: 'motion-cut-result',
            size: CUT_RESULT_WORDS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this._readback = device.createBuffer({
            label: 'motion-cut-readback',
            size: CUT_RESULT_WORDS * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this._flag = allocator({ label: 'motion-cut-flag', width: 1, height: 1, format: 'rgba16float' });
        this._dummy = device.createTexture({
            label: 'motion-cut-dummy',
            size: { width: 1, height: 1 },
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    /** Compiles the pipelines. */
    async init(): Promise<void> {
        const module = this.device.createShaderModule({ label: 'motion-cut', code: CUT_DETECTION_SHADER });
        [this._count, this._resolve] = await Promise.all(
            ['count', 'resolve'].map((entryPoint) =>
                this.device.createComputePipelineAsync({ label: `motion-cut-${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }),
            ),
        );
    }

    /** The verdict texture (1×1 rgba16float; r = reset this frame). Valid once this frame's {@link encode} has run. */
    get flag(): MotionTexture<THandle> {
        return this._flag;
    }

    /**
     * Encodes this frame's detection: one compute pass, no readback.
     * @param encoder - Command encoder
     * @param source - Motion from depth, encoded this frame and not yet ended
     * @param request - Extra motion candidates, thresholds and timestamps
     */
    encode(encoder: GPUCommandEncoder, source: MotionFromDepth<unknown>, request: CutRequest = {}): void {
        if (!this._count || !this._resolve) throw new Error('CutDetector: await init() first.');
        const motions = request.motions ?? [];
        if (motions.length > 2) throw new Error('CutDetector: at most two motion candidates besides zero and camera motion.');
        const { width, height } = source;
        for (const motion of motions)
            if (motion.width !== width || motion.height !== height) throw new Error('CutDetector: motion candidates must match the depth size.');
        const previous = source.previousLinearDepth;
        this._pixels = width * height;

        const buffer = new ArrayBuffer(CUT_CONSTANTS_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32.set([width, height, 1 / width, 1 / height]);
        f32[4] = request.depthTolerance ?? CUT_DEPTH_TOLERANCE;
        f32[5] = request.threshold ?? CUT_THRESHOLD;
        f32[6] = request.minSurface ?? CUT_MIN_SURFACE;
        u32[7] = motions.length;
        u32[8] = previous ? 1 : 0;
        this.device.queue.writeBuffer(this._constants, 0, buffer);

        const pass = encoder.beginComputePass({ label: 'motion-cut', timestampWrites: request.timestampWrites });
        // With no previous frame there is nothing to count: the resolve alone writes reset = 1.
        if (previous) {
            const view = (texture: GPUTexture | undefined) => (texture ?? this._dummy).createView({ baseMipLevel: 0, mipLevelCount: 1 });
            pass.setPipeline(this._count);
            pass.setBindGroup(
                0,
                this.device.createBindGroup({
                    layout: this._count.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this._constants } },
                        { binding: 1, resource: view(source.motion.gpu) },
                        { binding: 2, resource: view(previous) },
                        { binding: 3, resource: view(motions[0]) },
                        { binding: 4, resource: view(motions[1]) },
                        { binding: 5, resource: { buffer: this._counters } },
                    ],
                }),
            );
            pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        }
        pass.setPipeline(this._resolve);
        pass.setBindGroup(
            0,
            this.device.createBindGroup({
                layout: this._resolve.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this._constants } },
                    { binding: 5, resource: { buffer: this._counters } },
                    { binding: 6, resource: { buffer: this._result } },
                    { binding: 7, resource: this._flag.gpu.createView() },
                ],
            }),
        );
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /**
     * Reads back the last submitted verdict (submit the frame first). Optional:
     * the flag texture already holds it on the GPU. The copy is its own submit,
     * so a read still in flight never touches the frame's command buffer. One
     * read at a time.
     * @returns The verdict and its counts
     */
    async read(): Promise<CutResult> {
        if (this._reading) throw new Error('CutDetector: a read is already in flight.');
        this._reading = true;
        try {
            const encoder = this.device.createCommandEncoder({ label: 'motion-cut-readback' });
            encoder.copyBufferToBuffer(this._result, 0, this._readback, 0, CUT_RESULT_WORDS * 4);
            this.device.queue.submit([encoder.finish()]);
            await this._readback.mapAsync(GPUMapMode.READ);
            const words = new Uint32Array(this._readback.getMappedRange().slice(0));
            this._readback.unmap();
            return decodeCutResult(words, this._pixels);
        } finally {
            this._reading = false;
        }
    }

    /** Releases every GPU resource. */
    dispose(): void {
        this._flag.destroy();
        this._dummy.destroy();
        for (const buffer of [this._constants, this._counters, this._result, this._readback]) buffer.destroy();
    }
}

/**
 * Decodes the resolve's result words (exported for tests).
 * @param words - inconsistent, surface, cut, hasPrevious
 * @param pixels - Render pixels of the frame
 * @returns The verdict and its counts
 */
export function decodeCutResult(words: Uint32Array, pixels: number): CutResult {
    const [inconsistentPixels, surfacePixels, cut, hasPrevious] = words;
    return {
        cut: cut !== 0,
        share: surfacePixels > 0 ? inconsistentPixels / surfacePixels : 0,
        surface: pixels > 0 ? surfacePixels / pixels : 0,
        hasPrevious: hasPrevious !== 0,
        inconsistentPixels,
        surfacePixels,
        pixels,
    };
}
