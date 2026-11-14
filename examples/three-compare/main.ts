import * as THREE from 'three/webgpu';
import {
    clamp,
    color,
    Fn,
    float,
    Loop,
    mrt,
    normalLocal,
    output,
    positionLocal,
    screenSize,
    select,
    sin,
    texture,
    uniform,
    uv,
    vec2,
    vec3,
    vec4,
    velocity,
} from 'three/tsl';
import { MotionVectors } from '@pmndrs/motion-vectors/three';

/**
 * The same scene twice, side by side: three's `velocity` on the left, this
 * package's `MotionVectors` on the right. Each half renders its own copy (own
 * materials, own camera) into its own half-width MRT target, and a full-screen
 * quad puts the two next to each other.
 *
 * Every animation runs off one uniform, `sceneTime`, so "Pause" freezes the
 * scene completely: the correct motion is then exactly zero everywhere. three
 * still reports motion on every deformed object; `MotionVectors` reads zero.
 */

//* Renderer
const renderer = new THREE.WebGPURenderer({ antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
try {
    await renderer.init();
    if (!(renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) throw new Error('WebGPU is not available in this browser.');
} catch (error) {
    const box = document.getElementById('error')!;
    box.style.display = 'grid';
    box.textContent = `This example needs WebGPU. ${(error as Error).message}`;
    throw error;
}

const motion = MotionVectors.for(renderer);

//* Scene
/** Drives every animation, so a pause holds the whole scene still. */
const sceneTime = uniform(0);

function checker(a: number, b: number, scale: number): THREE.MeshStandardNodeMaterial {
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.7 });
    const cell = uv().mul(scale).floor();
    material.colorNode = select(cell.x.add(cell.y).mod(2).lessThan(1), color(a), color(b));
    return material;
}

interface Copy {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    animate(t: number): void;
}

/** One copy of the scene; each half gets its own, so their shader builds stay apart. */
function buildCopy(): Copy {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1b1f26);
    scene.add(new THREE.HemisphereLight(0xc8d8ff, 0x403830, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(4, 8, 5);
    scene.add(sun);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(24, 24), checker(0x8a8f99, 0x2c3038, 24));
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    // A wave across a plane: positionNode deformation driven by a uniform.
    const wave = new THREE.Mesh(new THREE.PlaneGeometry(5, 5, 160, 160), checker(0x4aa3d8, 0x1d3a52, 10));
    wave.material.positionNode = positionLocal.add(normalLocal.mul(sin(positionLocal.x.mul(2.2).add(sceneTime.mul(2.5))).mul(0.3)));
    wave.rotation.x = -Math.PI / 2;
    wave.position.set(-3.2, 0.6, 0);
    scene.add(wave);

    // A box slid sideways by positionNode (the object itself never moves).
    const slider = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), checker(0xe0a050, 0x503010, 4));
    slider.material.positionNode = positionLocal.add(vec3(sin(sceneTime.mul(1.3)).mul(1.4), 0, 0));
    slider.position.set(0.4, 0.5, 2.6);
    scene.add(slider);

    // A morph target pulsing.
    const sphereGeometry = new THREE.SphereGeometry(0.9, 96, 64);
    const base = sphereGeometry.attributes.position;
    const target = new Float32Array(base.count * 3);
    for (let i = 0; i < base.count; i++) {
        const [x, y, z] = [base.getX(i), base.getY(i), base.getZ(i)];
        const bulge = 1 + 0.45 * Math.sin(y * 6);
        target.set([x * bulge, y, z * bulge], i * 3);
    }
    sphereGeometry.morphAttributes.position = [new THREE.Float32BufferAttribute(target, 3)];
    const morph = new THREE.Mesh(sphereGeometry, checker(0xd070a0, 0x40243a, 12));
    morph.position.set(1.4, 1.1, -0.6);
    morph.updateMorphTargets();
    scene.add(morph);

    // A rigid spinner: both sources are exact here.
    const spinner = new THREE.Mesh(new THREE.TorusKnotGeometry(0.55, 0.18, 160, 24), checker(0x80d080, 0x24402a, 8));
    spinner.position.set(3.6, 1.1, 0.4);
    scene.add(spinner);

    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
    return {
        scene,
        camera,
        animate(t) {
            morph.morphTargetInfluences![0] = 0.5 + 0.5 * Math.sin(t * 1.7);
            spinner.rotation.set(t * 0.6, t * 0.9, 0);
            const angle = 0.6 + Math.sin(t * 0.25) * 0.35;
            camera.position.set(Math.sin(angle) * 14, 5.5, Math.cos(angle) * 14);
            camera.lookAt(0.3, 0.6, 0);
        },
    };
}

const left = buildCopy();
const right = buildCopy();

//* Targets
function target(): THREE.RenderTarget {
    const rt = new THREE.RenderTarget(1, 1, { count: 2, type: THREE.HalfFloatType, depthBuffer: true });
    // MRT routes outputs to attachments by texture name.
    rt.textures[0].name = 'output';
    rt.textures[1].name = 'velocity';
    return rt;
}
const leftTarget = target();
const rightTarget = target();
// The output must be named `velocity`: three only computes previous positions
// for skinned and instanced meshes when it is.
const leftMrt = mrt({ output, velocity });
const rightMrt = mrt({ output, velocity: motion.velocity });

//* Present
const view = uniform(0); // 0 motion vectors, 1 motion blur, 2 colour
const blurScale = uniform(1);

/** Motion in pixels → a colour: grey is still, hue is direction, saturation is speed. */
const showMotion = Fn(([motionNdc]: [THREE.Node<'vec2'>]) => {
    const pixels = motionNdc.mul(vec2(0.5, 0.5)).mul(screenSize);
    return clamp(vec3(0.5).add(vec3(pixels.x, pixels.y, pixels.x.add(pixels.y).negate()).mul(0.06)), 0, 1);
});

/** Colour smeared along the motion (UV = NDC · (0.5, −0.5)), 16 taps. */
function blur(colorTexture: THREE.Texture, motionTexture: THREE.Texture, at: THREE.Node<'vec2'>): THREE.Node<'vec4'> {
    return Fn(() => {
        const motionUV = texture(motionTexture, at).xy.mul(vec2(0.5, -0.5)).mul(blurScale);
        const sum = vec4(0).toVar();
        Loop(16, ({ i }) => {
            const offset = float(i).div(15).sub(0.5);
            sum.addAssign(texture(colorTexture, at.sub(motionUV.mul(offset))));
        });
        return sum.div(16);
    })() as THREE.Node<'vec4'>;
}

/** One half of the screen: `rt` stretched over [x0, x0 + 0.5] of the quad. */
function half(rt: THREE.RenderTarget, x0: number): THREE.Node<'vec4'> {
    const [colorTexture, motionTexture] = rt.textures;
    const at = vec2(uv().x.sub(x0).mul(2), uv().y);
    const vectors = vec4(showMotion(texture(motionTexture, at).xy), 1);
    return select(view.equal(0), vectors, select(view.equal(1), blur(colorTexture, motionTexture, at), texture(colorTexture, at)));
}

const quadMaterial = new THREE.MeshBasicNodeMaterial();
quadMaterial.colorNode = select(uv().x.lessThan(0.5), half(leftTarget, 0), half(rightTarget, 0.5));
const quad = new THREE.QuadMesh(quadMaterial);

//* Controls
let paused = false;
const pauseButton = document.getElementById('pause')!;
pauseButton.addEventListener('click', () => {
    paused = !paused;
    pauseButton.setAttribute('aria-pressed', String(paused));
});
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-view]')) {
    button.addEventListener('click', () => {
        view.value = ['motion', 'blur', 'color'].indexOf(button.dataset.view!);
        for (const other of document.querySelectorAll('[data-view]')) other.setAttribute('aria-pressed', String(other === button));
    });
}

function resize(): void {
    renderer.setSize(innerWidth, innerHeight);
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const halfWidth = Math.max(1, Math.floor(size.x / 2));
    leftTarget.setSize(halfWidth, size.y);
    rightTarget.setSize(halfWidth, size.y);
    for (const copy of [left, right]) {
        copy.camera.aspect = innerWidth / 2 / innerHeight;
        copy.camera.updateProjectionMatrix();
    }
}
addEventListener('resize', resize);
resize();

//* Frame
const timer = new THREE.Timer();
let t = 0;
renderer.setAnimationLoop((now) => {
    timer.update(now);
    // A hidden or collapsed page has no drawing buffer to render into.
    if (innerWidth === 0 || innerHeight === 0) return;
    const delta = Math.min(timer.getDelta(), 0.1);
    if (!paused) t += delta;
    sceneTime.value = t;
    left.animate(t);
    right.animate(t);
    // Two renders this frame (one per half): tell MotionVectors where the frame starts.
    motion.beginFrame();

    // Left: three's own position stage and velocity.
    motion.enabled = false;
    renderer.setMRT(leftMrt);
    renderer.setRenderTarget(leftTarget);
    renderer.render(left.scene, left.camera);

    // Right: MotionVectors.
    motion.enabled = true;
    renderer.setMRT(rightMrt);
    renderer.setRenderTarget(rightTarget);
    renderer.render(right.scene, right.camera);

    renderer.setMRT(null);
    renderer.setRenderTarget(null);
    // Motion colours are data: no tone mapping on them.
    renderer.toneMapping = view.value === 2 || view.value === 1 ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    quad.render(renderer);
});
