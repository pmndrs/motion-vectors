import { WarpingError, type WarpingResample, type WarpingResult } from '../../src/webgpu/index.js';
import { warpingErrorShader } from '../../src/webgpu/shaders/warpingError.js';

/**
 * Warping error resampling lab (EXPERIMENTAL), driven by
 * `scripts/measure-resample.mjs` over CDP. Engine-free: a bare device with no
 * optional features beyond timestamps, so `rgba32float` is unfilterable here.
 *
 * Each case renders a synthetic image that translates by an exact, constant
 * pixel offset per frame (4×4 box-supersampled, like a supersampled render)
 * and judges it with `WarpingError` through that exact motion. With nothing
 * flickering, any error left is the resampler's own: the bias that makes a
 * bilinear warp read sharp images as unstable (docs/MOTION.md §9). The
 * `noise` and `pulse` knobs add genuine flicker, which every filter must
 * still read.
 */

/** `zoneplate`: radial chirp up to `maxFrequency` cycles/px; `checker`: 8 px squares; `edges`: one vertical step per 32 px. */
type Pattern = 'zoneplate' | 'checker' | 'edges';

interface ResampleRequest {
    pattern: Pattern;
    /** Pixels per frame, current − previous. */
    velocity: [number, number];
    resample: WarpingResample;
    /** Image format (default `rgba16float`). */
    format?: 'rgba16float' | 'rgba32float' | 'rgba8unorm';
    /** Highest zone plate frequency, cycles/px (default 0.35). */
    maxFrequency?: number;
    /** Per-pixel, per-frame uniform noise amplitude (genuine flicker). */
    noise?: number;
    /** Whole-image brightness alternating ±`pulse` every frame (genuine flicker). */
    pulse?: number;
    /** Experiment only: skip Catmull-Rom's 2×2 deringing clamp. */
    noDering?: boolean;
    frames?: number;
    width?: number;
    height?: number;
}

interface ResampleRun {
    /** Frames 1… (frame 0 has no previous). */
    frames: WarpingResult[];
    /** Error-weighted pooled mean over every measured frame. */
    meanError: number;
    /** Mean scored share. */
    scored: number;
    /** GPU ms of the warping pass per frame, where timestamps exist. */
    gpuMs: number[];
}

const PATTERN_SHADER = /* wgsl */ `
struct Params {
    offset : vec2f,
    size : vec2f,
    maxFrequency : f32,
    noise : f32,
    gain : f32,
    frame : u32,
    pattern : u32,
    pad0 : u32,
    pad1 : u32,
    pad2 : u32,
};
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var image : texture_storage_2d<FORMAT, write>;

const PI : f32 = 3.14159265358979;

// Continuous scene value at image position x (px), already moved by the frame's offset.
fn scene(x : vec2f) -> f32 {
    if (P.pattern == 0u) {
        let c = 0.5 * P.size;
        let radius = 0.5 * min(P.size.x, P.size.y);
        let r = length(x - c);
        // Local frequency k r cycles/px reaches maxFrequency at the radius; flat grey beyond.
        let k = P.maxFrequency / radius;
        let fade = 1.0 - smoothstep(0.9 * radius, radius, r);
        return 0.5 + 0.4 * fade * cos(PI * k * r * r);
    }
    if (P.pattern == 1u) {
        let cell = vec2i(floor(x / 8.0));
        return select(0.1, 0.9, ((cell.x + cell.y) & 1) == 0);
    }
    return select(0.1, 0.9, fract(x.x / 64.0) < 0.5);
}

fn hash(p : vec3u) -> f32 {
    var h = p.x * 1664525u + p.y * 1013904223u + p.z * 2654435761u;
    h ^= h >> 16u; h *= 2246822519u; h ^= h >> 13u; h *= 3266489917u; h ^= h >> 16u;
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= P.size)) { return; }
    var sum = 0.0;
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let sub = (vec2f(f32(i), f32(j)) + 0.5) / 4.0;
            sum += scene(vec2f(gid.xy) + sub - P.offset);
        }
    }
    let n = P.noise * (hash(vec3u(gid.xy, P.frame)) - 0.5) * 2.0;
    let v = max(sum / 16.0 * P.gain + n, 0.0);
    textureStore(image, vec2i(gid.xy), vec4f(v, v, v, 1.0));
}
`;

const PATTERNS: Pattern[] = ['zoneplate', 'checker', 'edges'];

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter.');
const timing = adapter.features.has('timestamp-query');
// Deliberately no float32-filterable: the 16-tap filters must take rgba32float without it.
const device = await adapter.requestDevice({ requiredFeatures: timing ? ['timestamp-query'] : [] });

const generators = new Map<string, GPUComputePipeline>();
function generator(format: GPUTextureFormat): GPUComputePipeline {
    let pipeline = generators.get(format);
    if (!pipeline) {
        const code = PATTERN_SHADER.replace('FORMAT', format);
        pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
        generators.set(format, pipeline);
    }
    return pipeline;
}

async function run(request: ResampleRequest): Promise<ResampleRun> {
    const width = request.width ?? 1280;
    const height = request.height ?? 720;
    const format = request.format ?? 'rgba16float';
    const frames = request.frames ?? 16;
    const [vx, vy] = request.velocity;

    const warping = new WarpingError(device, { resample: request.resample });
    await warping.init();
    if (request.noDering) {
        // Experiment only: the same filter without its deringing clamp.
        const code = warpingErrorShader(request.resample, false);
        (warping as unknown as { _pipeline: GPUComputePipeline })._pipeline = await device.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
        });
    }

    const image = device.createTexture({
        size: { width, height },
        format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    // Contract motion, one texel for the whole image: NDC current − previous, flag 1 (exact), .w unused (no mask).
    const motion = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: motion }, new Float32Array([(2 * vx) / width, (-2 * vy) / height, 1, 1]), {}, { width: 1, height: 1 });
    const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipeline = generator(format);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: image.createView() },
        ],
    });
    const querySet = timing ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const resolve = timing ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
    const readTimes = timing ? device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;

    const results: WarpingResult[] = [];
    const gpuMs: number[] = [];
    device.pushErrorScope('validation');
    for (let frame = 0; frame < frames; frame++) {
        const data = new ArrayBuffer(48);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        const pulse = request.pulse ? (frame % 2 === 0 ? 1 + request.pulse : 1 - request.pulse) : 1;
        f32.set([vx * frame, vy * frame, width, height, request.maxFrequency ?? 0.35, request.noise ?? 0, pulse]);
        u32[7] = frame;
        u32[8] = PATTERNS.indexOf(request.pattern);
        device.queue.writeBuffer(params, 0, data);

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        warping.encode(encoder, {
            current: image,
            motion,
            timestampWrites: querySet && frame > 0 ? { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
        });
        if (querySet && frame > 0) {
            encoder.resolveQuerySet(querySet, 0, 2, resolve!, 0);
            encoder.copyBufferToBuffer(resolve!, 0, readTimes!, 0, 16);
        }
        device.queue.submit([encoder.finish()]);
        const result = await warping.read();
        if (frame > 0) {
            results.push(result);
            if (readTimes) {
                await readTimes.mapAsync(GPUMapMode.READ);
                const t = new BigUint64Array(readTimes.getMappedRange().slice(0));
                readTimes.unmap();
                gpuMs.push(Number(t[1] - t[0]) / 1e6);
            }
        }
    }
    const error = await device.popErrorScope();
    warping.dispose();
    for (const resource of [image, motion, params, querySet, resolve, readTimes]) resource?.destroy();
    if (error) throw new Error(`${request.resample} on ${format}: ${error.message}`);

    const scored = results.reduce((total, r) => total + r.counts.scored, 0);
    return {
        frames: results,
        meanError: scored ? results.reduce((total, r) => total + r.meanError * r.counts.scored, 0) / scored : 0,
        scored: results.reduce((total, r) => total + r.scored, 0) / results.length,
        gpuMs,
    };
}

/**
 * GPU time of the warping pass per filter: `frames` back-to-back frames per
 * filter, filters interleaved over `reps`, no readback in between (a readback
 * per frame lets the GPU drop to a low clock and inflates every pass).
 * @returns Per filter, every frame's pass time in ms
 */
async function cost({ width = 1280, height = 720, frames = 64, reps = 4 }: { width?: number; height?: number; frames?: number; reps?: number }): Promise<Record<WarpingResample, number[]>> {
    if (!timing) throw new Error('No timestamp-query on this adapter.');
    const filters: WarpingResample[] = ['bilinear', 'catmull-rom'];
    const format = 'rgba16float';
    const image = device.createTexture({
        size: { width, height },
        format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const motion = device.createTexture({ size: { width: 1, height: 1 }, format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: motion }, new Float32Array([(2 * 0.37) / width, (-2 * 0.21) / height, 1, 1]), {}, { width: 1, height: 1 });
    const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new ArrayBuffer(48);
    new Float32Array(data).set([0, 0, width, height, 0.35, 0, 1]);
    device.queue.writeBuffer(params, 0, data);
    const pipeline = generator(format);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: image.createView() },
        ],
    });
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * frames });
    const resolveBuffer = device.createBuffer({ size: 16 * frames, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readBuffer = device.createBuffer({ size: 16 * frames, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const meters = await Promise.all(
        filters.map(async (resample) => {
            const warping = new WarpingError(device, { resample });
            await warping.init();
            return warping;
        }),
    );
    const times: Record<WarpingResample, number[]> = { bilinear: [], 'catmull-rom': [] };
    for (let rep = 0; rep < reps; rep++) {
        for (const [index, warping] of meters.entries()) {
            for (let frame = 0; frame <= frames; frame++) {
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
                pass.end();
                // Frame 0 only fills the history; frames 1… are timed.
                const timestampWrites = frame > 0 ? { querySet, beginningOfPassWriteIndex: 2 * (frame - 1), endOfPassWriteIndex: 2 * (frame - 1) + 1 } : undefined;
                warping.encode(encoder, { current: image, motion, timestampWrites });
                device.queue.submit([encoder.finish()]);
            }
            const encoder = device.createCommandEncoder();
            encoder.resolveQuerySet(querySet, 0, 2 * frames, resolveBuffer, 0);
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16 * frames);
            device.queue.submit([encoder.finish()]);
            await readBuffer.mapAsync(GPUMapMode.READ);
            const t = new BigUint64Array(readBuffer.getMappedRange().slice(0));
            readBuffer.unmap();
            for (let frame = 0; frame < frames; frame++) times[filters[index]].push(Number(t[2 * frame + 1] - t[2 * frame]) / 1e6);
            warping.reset();
        }
    }
    for (const warping of meters) warping.dispose();
    for (const resource of [image, motion, params, querySet, resolveBuffer, readBuffer]) resource.destroy();
    return times;
}

declare global {
    interface Window {
        __resampleLab?: {
            ready: boolean;
            timing: boolean;
            run: (request: ResampleRequest) => Promise<ResampleRun>;
            cost: typeof cost;
        };
    }
}
window.__resampleLab = { ready: true, timing, run, cost };
