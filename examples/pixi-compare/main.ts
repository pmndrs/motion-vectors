import {
    Container3D,
    CubeGeometry,
    DirectionalLight,
    FullscreenPass,
    Material3D,
    Mesh3D,
    Pass3D,
    PlaneGeometry,
    PostEffect,
    SphereGeometry,
    View3D,
    passthroughFragment,
    type EffectFragment,
    type EffectIO,
    type PassContext,
    type RenderGraph,
} from '@pixi/3d';
import { velocityOutput } from '@pixi/3d/extras';
import { Application, RenderTexture, type WebGPURenderer } from 'pixi.js';
import { MotionVectors, motionOutput } from '@pmndrs/motion-vectors/pixi';

/**
 * pixi-3d's own `velocityOutput` (left of the divider) next to this package's
 * `MotionVectors` (right), over the same frame. pixi-3d's velocity is camera
 * motion only, so the sliding box, the spinning box and the orbiting ball read
 * as still there; `MotionVectors` tracks them.
 *
 * `?check=1` builds a deterministic scene instead and exposes
 * `window.__pixiMotion.check()`, which `scripts/check-pixi.mjs` drives.
 */

const CHECK = new URLSearchParams(location.search).get('check') === '1';

//* Application
const app = new Application();
try {
    await app.init({ resizeTo: window, preference: 'webgpu', antialias: false, backgroundColor: 0x1b1f26 });
    if (!(app.renderer as { gpu?: unknown }).gpu) throw new Error('WebGPU is not available in this browser.');
} catch (error) {
    const box = document.getElementById('error')!;
    box.style.display = 'grid';
    box.textContent = `This example needs WebGPU. ${(error as Error).message}`;
    throw error;
}
document.body.appendChild(app.canvas);

const view = new View3D({ autoResize: true, resolution: 1, environment: { skyColor: 0xc8d8ff, groundColor: 0x403830, intensity: 0.9 } });
app.stage.addChild(view);
const sun = new DirectionalLight({ color: 0xffffff, intensity: 2.4 });
sun.rotation.set(-Math.PI / 3, Math.PI / 5, 0);
view.root.addChild(sun);

const motion = new MotionVectors();
const box = (color: number, size: [number, number, number]) =>
    new Mesh3D({ geometry: new CubeGeometry({ width: size[0], height: size[1], depth: size[2] }), material: new Material3D({ baseColor: color, roughness: 0.7 }) });

//* Comparison View
// Grey is still; hue is direction, saturation is speed (the three example's coding).
const COMPARE_FRAGMENT: EffectFragment = {
    gpu: /* wgsl */ `
fn mvColor(pixels: vec2<f32>) -> vec4<f32> {
  let c = vec3<f32>(0.5) + vec3<f32>(pixels.x, pixels.y, -(pixels.x + pixels.y)) * 0.06;
  return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
fn effect(uv: vec2<f32>) -> vec4<f32> {
  if (uv.x < 0.5) {
    // pixi-3d: UV delta with v along the texture rows; uFxClip.w's sign says which way is up.
    let v = sampleColor2(uv).xy;
    return mvColor(vec2<f32>(v.x, v.y * sign(fxh.uFxClip.w)) / texel());
  }
  // MotionVectors: NDC delta, y up.
  return mvColor(sampleColor(uv).xy * 0.5 / texel());
}
`,
    gl: /* glsl */ `
vec4 mvColor(vec2 pixels) {
  return vec4(clamp(vec3(0.5) + vec3(pixels.x, pixels.y, -(pixels.x + pixels.y)) * 0.06, 0.0, 1.0), 1.0);
}
vec4 effect(vec2 uv) {
  if (uv.x < 0.5) {
    vec2 v = sampleColor2(uv).xy;
    return mvColor(vec2(v.x, v.y * sign(uFxClip.w)) / texel());
  }
  return mvColor(sampleColor(uv).xy * 0.5 / texel());
}
`,
};

class CompareViewer extends PostEffect {
    readonly requiredOutputs = [motionOutput, velocityOutput];
    constructor() {
        super({ stage: 'afterToneMapping' });
    }
    build(graph: RenderGraph, io: EffectIO): void {
        graph.addPass(
            new FullscreenPass({
                renderer: io.renderer,
                label: 'motion-compare',
                fragment: COMPARE_FRAGMENT,
                reads: { color: graph.getSceneOutput(motionOutput), color2: graph.getSceneOutput(velocityOutput) },
                writes: io.output,
            }),
        );
    }
}

//* Readback (check mode)
/** Copies both outputs into textures this page owns, so they outlive the frame. */
class CapturePass extends Pass3D {
    readonly stage = 'afterTransparent' as const;
    readonly requiredOutputs = [motionOutput, velocityOutput];
    constructor(
        private readonly ours: RenderTexture,
        private readonly theirs: RenderTexture,
    ) {
        super();
    }
    build(graph: RenderGraph, ctx: PassContext): void {
        for (const [output, target] of [
            [motionOutput, this.ours],
            [velocityOutput, this.theirs],
        ] as const)
            graph.addPass(
                new FullscreenPass({
                    renderer: ctx.renderer,
                    label: `capture-${output.name}`,
                    fragment: passthroughFragment,
                    reads: { color: graph.getSceneOutput(output) },
                    writes: graph.importColor(() => target),
                }),
            );
    }
}

/** IEEE half → float. */
function half(bits: number): number {
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    const sign = bits & 0x8000 ? -1 : 1;
    if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
    if (exponent === 31) return mantissa ? NaN : sign * Infinity;
    return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

async function readRgba16(texture: RenderTexture): Promise<{ width: number; height: number; data: Float32Array }> {
    const renderer = app.renderer as WebGPURenderer;
    const device = renderer.gpu.device;
    const gpuTexture = renderer.texture.getGpuSource(texture.source);
    const { width, height } = gpuTexture;
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: gpuTexture }, { buffer, bytesPerRow }, { width, height });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint16Array(buffer.getMappedRange());
    const data = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++)
        for (let i = 0; i < width * 4; i++) data[y * width * 4 + i] = half(words[(y * bytesPerRow) / 2 + i]);
    buffer.unmap();
    buffer.destroy();
    return { width, height, data };
}

//* Demo Scene
function demo(): void {
    const viewer = new CompareViewer();
    view.passes = [viewer];

    const floor = new Mesh3D({
        geometry: new PlaneGeometry({ width: 24, height: 24 }),
        material: new Material3D({ baseColor: 0x8a8f99, roughness: 0.9 }),
        rotation: { x: -Math.PI / 2, y: 0, z: 0 },
    });
    view.root.addChild(floor);
    for (let x = -6; x <= 6; x += 3) {
        const pillar = box(0x60c0e0, [0.6, 2.4, 0.6]);
        pillar.position.set(x, 1.2, -4);
        view.root.addChild(pillar);
    }
    const slider = box(0xe0a050, [1, 1, 1]);
    slider.position.set(0, 0.5, 2);
    const spinner = box(0x80d080, [1.2, 1.2, 1.2]);
    spinner.position.set(3.4, 1.2, 0);
    const pivot = new Container3D();
    pivot.position.set(-3, 1, 0);
    const ball = new Mesh3D({ geometry: new SphereGeometry({ radius: 0.5 }), material: new Material3D({ baseColor: 0xd070a0, roughness: 0.5 }) });
    ball.position.set(1.4, 0, 0);
    pivot.addChild(ball);
    view.root.addChild(slider, spinner, pivot);
    for (const mesh of [slider, spinner, ball]) motion.track(mesh);

    let paused = false;
    const pauseButton = document.getElementById('pause')!;
    pauseButton.addEventListener('click', () => {
        paused = !paused;
        pauseButton.setAttribute('aria-pressed', String(paused));
    });
    for (const button of document.querySelectorAll<HTMLButtonElement>('[data-view]'))
        button.addEventListener('click', () => {
            viewer.enabled = button.dataset.view === 'motion';
            for (const other of document.querySelectorAll('[data-view]')) other.setAttribute('aria-pressed', String(other === button));
        });

    let t = 0;
    app.ticker.add((ticker) => {
        if (!paused) t += Math.min(ticker.deltaMS / 1000, 0.1);
        slider.position.x = Math.sin(t * 1.3) * 2.5;
        spinner.rotation.set(t * 0.6, t * 0.9, 0);
        pivot.rotation.y = t * 1.5;
        const angle = 0.5 + Math.sin(t * 0.25) * 0.3;
        view.camera.position.set(Math.sin(angle) * 11, 4.5, Math.cos(angle) * 11);
        view.camera.lookAt({ x: 0, y: 0.8, z: 0 });
        motion.update();
    });
}

//* Check Scene
interface CheckState {
    camera: [number, number, number];
    plane: [number, number];
    visible?: boolean;
    renders?: number;
}

function check(): void {
    // CI diagnostics: which adapter, and why a device was lost.
    const gpu = (app.renderer as WebGPURenderer).gpu;
    const info = gpu.adapter.info;
    console.log(`pixi check: adapter ${info.vendor} ${info.architecture} ${info.device} ${info.description}`);
    void gpu.device.lost.then((lost) => console.error(`pixi check: device lost (${lost.reason}): ${lost.message}`));
    for (const element of document.querySelectorAll<HTMLElement>('.label, #divider, #controls')) element.style.display = 'none';
    app.ticker.stop();
    const width = app.renderer.width;
    const height = app.renderer.height;
    const ours = RenderTexture.create({ width, height, format: 'rgba16float', resolution: 1 });
    const theirs = RenderTexture.create({ width, height, format: 'rgba16float', resolution: 1 });
    // Render off screen: headless Linux Chrome (CI) loses the WebGPU instance at
    // the first canvas present on SwiftShader, and nothing here needs the canvas.
    const target = RenderTexture.create({ width, height, resolution: 1 });
    view.passes = [new CapturePass(ours, theirs)];

    // A backdrop at depth 15 and a 2×2 plane at depth 10, both facing the camera,
    // which only translates: every point of each moves by one NDC delta.
    const backdrop = new Mesh3D({ geometry: new PlaneGeometry({ width: 80, height: 50 }), material: new Material3D({ baseColor: 0x707880 }) });
    backdrop.position.set(0, 0, -5);
    const plane = new Mesh3D({ geometry: new PlaneGeometry({ width: 2, height: 2 }), material: new Material3D({ baseColor: 0xe0a050 }) });
    view.root.addChild(backdrop, plane);
    motion.track(plane);
    view.camera.fov = 50;

    const apply = (state: CheckState) => {
        view.camera.position.set(...state.camera);
        plane.position.set(state.plane[0], state.plane[1], 0);
        plane.visible = state.visible !== false;
        motion.update();
        for (let i = 0; i < (state.renders ?? 1); i++) app.renderer.render({ container: app.stage, target });
    };
    const project = (vp: Float32Array, x: number, y: number, z: number): [number, number] => {
        const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
        const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
        const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        return [cx / cw, cy / cw];
    };
    const viewProjection = () => new Float32Array(view.camera.viewProjection.elements);

    // The output is rgba16float: compare with the expectation as f16 stores it, or a
    // 28 px motion reads ~0.017 px of pure rounding (for pixi-3d's own velocity too).
    const f16 = (Math as unknown as { f16round?: (x: number) => number }).f16round ?? ((x: number) => x);

    /** Max / mean endpoint error (render pixels) of `ours` against an expected NDC delta over a pixel rect. */
    function measure(
        data: Float32Array,
        exact: [number, number],
        inside: (x: number, y: number) => boolean,
        scale: [number, number] = [1, 1],
    ): { pixels: number; max: number; mean: number; value: [number, number] } {
        // `scale` maps the stored value to NDC (pixi-3d stores UV), so round in the stored units.
        const expected: [number, number] = [f16(exact[0] * scale[0]) / scale[0], f16(exact[1] * scale[1]) / scale[1]];
        let pixels = 0;
        let max = 0;
        let sum = 0;
        let sx = 0;
        let sy = 0;
        for (let y = 0; y < height; y++)
            for (let x = 0; x < width; x++) {
                if (!inside(x, y)) continue;
                const o = (y * width + x) * 4;
                const ex = (data[o] - expected[0]) * 0.5 * width;
                const ey = (data[o + 1] - expected[1]) * 0.5 * height;
                const error = Math.hypot(ex, ey);
                pixels++;
                sum += error;
                sx += data[o];
                sy += data[o + 1];
                if (error > max) max = error;
            }
        return { pixels, max, mean: pixels ? sum / pixels : 0, value: [sx / Math.max(pixels, 1), sy / Math.max(pixels, 1)] };
    }

    async function runCase(name: string, previous: CheckState, current: CheckState) {
        apply(previous);
        const vpPrevious = viewProjection();
        apply(current);
        const vp = viewProjection();
        const [oursData, theirsData] = await Promise.all([readRgba16(ours), readRgba16(theirs)]);

        // The plane's screen rect now, shrunk by 3 px (edge pixels mix surfaces).
        const corners = [-1, 1].flatMap((dx) => [-1, 1].map((dy) => project(vp, current.plane[0] + dx, current.plane[1] + dy, 0)));
        const px = corners.map(([x]) => (x * 0.5 + 0.5) * width);
        const py = corners.map(([, y]) => (0.5 - y * 0.5) * height);
        const rect = { x0: Math.min(...px), x1: Math.max(...px), y0: Math.min(...py), y1: Math.max(...py) };
        const onPlane = (x: number, y: number) => x + 0.5 > rect.x0 + 3 && x + 0.5 < rect.x1 - 3 && y + 0.5 > rect.y0 + 3 && y + 0.5 < rect.y1 - 3;
        const offPlane = (x: number, y: number) =>
            current.visible === false || x + 0.5 < rect.x0 - 3 || x + 0.5 > rect.x1 + 3 || y + 0.5 < rect.y0 - 3 || y + 0.5 > rect.y1 + 3;

        const delta = (point: [number, number, number], previousPoint: [number, number, number]): [number, number] => {
            const now = project(vp, ...point);
            const before = project(vpPrevious, ...previousPoint);
            return [now[0] - before[0], now[1] - before[1]];
        };
        const planeNow: [number, number, number] = [current.plane[0], current.plane[1], 0];
        // Zero where the plane had no previous frame (contract: no previous position → 0).
        const planeExpected: [number, number] = previous.visible === false ? [0, 0] : delta(planeNow, [previous.plane[0], previous.plane[1], 0]);
        const backdropExpected = delta([0, 0, -5], [0, 0, -5]);

        // pixi-3d's velocity, in the contract's units: UV delta → NDC (WebGPU rows run down).
        const PIXI_UV: [number, number] = [0.5, -0.5];
        const theirsNdc = new Float32Array(theirsData.data.length);
        for (let i = 0; i < theirsNdc.length; i += 4) {
            theirsNdc[i] = theirsData.data[i] / 0.5;
            theirsNdc[i + 1] = theirsData.data[i + 1] / -0.5;
        }
        return {
            name,
            plane: current.visible === false ? null : { expected: planeExpected, ours: measure(oursData.data, planeExpected, onPlane), pixi: measure(theirsNdc, planeExpected, onPlane, PIXI_UV) },
            backdrop: { expected: backdropExpected, ours: measure(oursData.data, backdropExpected, offPlane), pixi: measure(theirsNdc, backdropExpected, offPlane, PIXI_UV) },
        };
    }

    const still: CheckState = { camera: [0, 0, 10], plane: [0, 0] };
    (window as unknown as { __pixiMotion: unknown }).__pixiMotion = {
        ready: true,
        size: [width, height],
        async check() {
            // Warm up: build pipelines, start every history.
            apply(still);
            apply(still);
            return [
                await runCase('hold', still, still),
                await runCase('camera moves', still, { camera: [0.3, 0.2, 10], plane: [0, 0] }),
                await runCase('plane moves', still, { camera: [0, 0, 10], plane: [0.4, -0.25] }),
                await runCase('both move', { camera: [0, 0, 10], plane: [0.4, 0] }, { camera: [-0.2, 0.1, 10], plane: [0.1, 0.3] }),
                await runCase('plane moves, 2 renders', still, { camera: [0, 0, 10], plane: [-0.3, 0.2], renders: 2 }),
                // Known pixi-3d limit (docs/PIXI.md): its camera compares each render with
                // its own previous render, so the second render of a frame reads no camera motion.
                await runCase('camera moves, 2 renders', still, { camera: [0.3, 0.2, 10], plane: [0, 0], renders: 2 }),
                await runCase('plane reappears', { ...still, visible: false }, { camera: [0, 0, 10], plane: [0.5, 0] }),
                await runCase('hold after motion', { camera: [0, 0, 10], plane: [0.5, 0] }, { camera: [0, 0, 10], plane: [0.5, 0] }),
            ];
        },
    };
}

if (CHECK) check();
else demo();
