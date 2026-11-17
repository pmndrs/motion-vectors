import { CAMERA_MOTION_SHADER, MOTION_CONSTANTS_BYTES } from './shaders/cameraMotion.js';
import { createDeviceTextureAllocator, type MotionTexture, type TextureAllocator } from './textures.js';

/**
 * Camera matrices for one frame, column-major (three's `Matrix4.elements`).
 * `inverseViewProjectionJittered` must invert the projection the depth was
 * rasterized with; `viewProjection` must be the unjittered one, so the derived
 * motion is jitter-free like three's `velocity` node.
 */
export interface CameraMatrices {
    viewProjection: ArrayLike<number>;
    inverseViewProjectionJittered: ArrayLike<number>;
    view: ArrayLike<number>;
    /** The depth buffer's clear value: 1 (default), or 0 for reversed depth. Marks background pixels. */
    backgroundDepth?: number;
}

/**
 * Camera motion from depth on a bare `GPUDevice`: depth + this and last
 * frame's camera matrices → a contract motion texture (CONTRACT.md: NDC delta,
 * flag 2 on surfaces, 0 on background, 4 with no previous position; `.w` the
 * previous linear depth) and this frame's linear depth, kept for one frame.
 * Exact for static geometry; object motion is absent (that's what flag 2 says).
 *
 * Per frame: {@link encode} once, then everything that reads
 * {@link previousLinearDepth}, then {@link endFrame}.
 */
export class MotionFromDepth<THandle = GPUTexture> {
    readonly device: GPUDevice;

    private readonly _allocate: TextureAllocator<THandle>;
    private _pipeline: GPUComputePipeline | null = null;
    private readonly _constants: GPUBuffer;
    private _width = 0;
    private _height = 0;
    private _motion: MotionTexture<THandle> | null = null;
    private _linearDepth: GPUTexture[] = [];
    private _depthIndex = 0;
    private _hasPrevious = false;
    private readonly _prevViewProjection = new Float32Array(16);
    private readonly _prevViewZRow = new Float32Array(4);

    /**
     * @param device - The device the engine renders with
     * @param allocate - Allocator for the motion texture an engine samples
     */
    constructor(device: GPUDevice, allocate?: TextureAllocator<THandle>) {
        this.device = device;
        this._allocate = allocate ?? (createDeviceTextureAllocator(device) as unknown as TextureAllocator<THandle>);
        this._constants = device.createBuffer({
            label: 'motion-camera-constants',
            size: MOTION_CONSTANTS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /** Compiles the pipeline. */
    async init(): Promise<void> {
        const label = 'motion-camera';
        this._pipeline = await this.device.createComputePipelineAsync({
            label,
            layout: 'auto',
            compute: { module: this.device.createShaderModule({ label, code: CAMERA_MOTION_SHADER }), entryPoint: 'main' },
        });
    }

    /** Render width the products are sized to (0 before {@link configure}). */
    get width(): number {
        return this._width;
    }

    /** Render height the products are sized to (0 before {@link configure}). */
    get height(): number {
        return this._height;
    }

    /** Camera motion at render resolution (rgba16float, the contract; see `cameraMotion.ts`). */
    get motion(): MotionTexture<THandle> {
        if (!this._motion) throw new Error('MotionFromDepth: configure() first.');
        return this._motion;
    }

    /** This frame's linear depth (r32float, render resolution). */
    get linearDepth(): GPUTexture {
        if (!this._motion) throw new Error('MotionFromDepth: configure() first.');
        return this._linearDepth[this._depthIndex];
    }

    /** Last frame's linear depth, or null on the first frame and after {@link reset}. */
    get previousLinearDepth(): GPUTexture | null {
        return this._hasPrevious ? this._linearDepth[1 - this._depthIndex] : null;
    }

    /** True once a previous frame's depth and matrices exist. */
    get hasPrevious(): boolean {
        return this._hasPrevious;
    }

    /**
     * (Re)allocates the render-resolution products and forgets all history.
     * @param width - Render width
     * @param height - Render height
     */
    configure(width: number, height: number): void {
        if (width === this._width && height === this._height && this._motion) return;
        this._width = width;
        this._height = height;
        this._motion?.destroy();
        for (const texture of this._linearDepth) texture.destroy();
        this._motion = this._allocate({ label: 'motion-camera', width, height, format: 'rgba16float' });
        this._linearDepth = [0, 1].map((index) =>
            this.device.createTexture({
                label: `motion-linear-depth-${index}`,
                size: { width, height },
                format: 'r32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
            }),
        );
        this.reset();
    }

    /** Forgets the previous frame: the next one reports no previous position (flag 4) everywhere it has a surface. */
    reset(): void {
        this._hasPrevious = false;
    }

    /**
     * Depth → camera motion + linear depth. Call once per frame, before anything reads them.
     * @param encoder - Command encoder
     * @param depth - Render-res depth (any depth format)
     * @param matrices - This frame's camera matrices
     * @param timestampWrites - GPU timestamps around the pass
     */
    encode(encoder: GPUCommandEncoder, depth: GPUTexture, matrices: CameraMatrices, timestampWrites?: GPUComputePassTimestampWrites): void {
        if (!this._pipeline) throw new Error('MotionFromDepth: await init() first.');
        const motion = this.motion;
        const data = new Float32Array(MOTION_CONSTANTS_BYTES / 4);
        const viewZRow = [matrices.view[2], matrices.view[6], matrices.view[10], matrices.view[14]];
        data.set(Array.from(matrices.inverseViewProjectionJittered), 0);
        data.set(Array.from(matrices.viewProjection), 16);
        // First frame: previous = current, so the motion is zero, not garbage.
        data.set(this._hasPrevious ? this._prevViewProjection : Array.from(matrices.viewProjection), 32);
        data.set(viewZRow, 48);
        data.set(this._hasPrevious ? this._prevViewZRow : viewZRow, 52);
        data.set([this._width, this._height, 1 / this._width, 1 / this._height], 56);
        data[60] = matrices.backgroundDepth ?? 1;
        data[61] = this._hasPrevious ? 1 : 0;
        this.device.queue.writeBuffer(this._constants, 0, data);
        this._prevViewProjection.set(Array.from(matrices.viewProjection));
        this._prevViewZRow.set(viewZRow);

        const depthView = depth.createView({
            aspect: depth.format.includes('stencil') ? 'depth-only' : 'all',
            baseMipLevel: 0,
            mipLevelCount: 1,
        });
        const bindGroup = this.device.createBindGroup({
            layout: this._pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this._constants } },
                { binding: 1, resource: depthView },
                { binding: 2, resource: motion.gpu.createView() },
                { binding: 3, resource: this.linearDepth.createView() },
            ],
        });
        const pass = encoder.beginComputePass({ label: 'motion-camera', timestampWrites });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this._width / 8), Math.ceil(this._height / 8));
        pass.end();
    }

    /** Keeps this frame's depth and matrices as the previous ones; call after the frame's last reader. */
    endFrame(): void {
        this._depthIndex = 1 - this._depthIndex;
        this._hasPrevious = true;
    }

    /** Releases every GPU resource. */
    dispose(): void {
        this._motion?.destroy();
        for (const texture of this._linearDepth) texture.destroy();
        this._constants.destroy();
    }
}
