import * as THREE from 'three/webgpu';
import {
    attribute,
    clamp,
    instanceIndex,
    int,
    ivec2,
    normalGeometry,
    normalLocal,
    positionGeometry,
    positionLocal,
    positionPrevious,
    texture,
    textureLoad,
    uniform,
    uniformArray,
    uv,
    vec4,
} from 'three/tsl';

import { MotionVectors } from '../../../src/three/index.js';
import { checkerMaterial, ramp, segmentAt, sunLight, timeline, type SceneFactory } from './common';

/** Move over these frames, then hold. */
const MOVE_START = 30;
const MOVE_FRAMES = 60;
const INSTANCES = 6;
const BATCHED = 3;
/** Height texture size and displacement scale of the texture-driven planes (R7). */
const HEIGHTS = 64;
const HEIGHT_SCALE = 0.5;

/**
 * R3 and R7's lab cases (docs/PLAN.md §3.2), still camera, move-then-hold.
 * R3, the deformations three moves `positionPrevious` for itself: 1 a skinned
 * limb whose upper bone bends, 2 an `InstancedMesh` whose instances rise and
 * turn, 3 a `BatchedMesh` whose instances slide. R7, a `positionNode` reading
 * a height texture the CPU rewrites every frame (a bump sliding across), which
 * the node's rebuild can't see: 4 with the author's previous position
 * (`MotionVectors.setPreviousPosition`, last frame's heights), 5 marked
 * `'rigid'`, 6 with no hook (the gap the hook exists for). R9: 7 a static
 * `displacementMap` whose `displacementScale` grows.
 *
 * Every oracle keeps its own copy of last frame's bone / instance matrices,
 * computed here from the analytic pose, so none of three's previous-data
 * bookkeeping reaches it. The pose is a function of the frame only, so a
 * frame's matrices are rebuilt from scratch, never accumulated.
 */
export const createRigs: SceneFactory = (oracle) => {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1c2128);
    scene.add(new THREE.HemisphereLight(0xc8d8ff, 0x403830, 0.9));
    scene.add(sunLight(2.4, [3, 8, 6], 9));
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), checkerMaterial(0x8a8f99, 0x30343c, 30));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const progressAt = (frame: number) => ramp(frame, MOVE_START, MOVE_FRAMES) / MOVE_FRAMES;

    //* Skinned Limb
    // A 3-unit cylinder, two bones (0–1.5, 1.5–3), weights blended across the joint.
    const limbGeometry = new THREE.CylinderGeometry(0.32, 0.32, 3, 32, 48);
    limbGeometry.translate(0, 1.5, 0);
    const skinIndex: number[] = [];
    const skinWeight: number[] = [];
    const limbPositions = limbGeometry.attributes.position;
    for (let i = 0; i < limbPositions.count; i++) {
        const upper = THREE.MathUtils.clamp((limbPositions.getY(i) - 1.2) / 0.6, 0, 1);
        skinIndex.push(0, 1, 0, 0);
        skinWeight.push(1 - upper, upper, 0, 0);
    }
    limbGeometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndex, 4));
    limbGeometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeight, 4));
    const root = new THREE.Bone();
    const joint = new THREE.Bone();
    joint.position.y = 1.5;
    root.add(joint);
    const limb = new THREE.SkinnedMesh(limbGeometry, checkerMaterial(0xe0a050, 0x403020, 6));
    limb.add(root);
    limb.bind(new THREE.Skeleton([root, joint]));
    limb.position.set(-4.2, 0, 0);
    limb.castShadow = true;
    limb.frustumCulled = false;
    const boneCount = limb.skeleton.bones.length;
    const previousBones = Array.from({ length: boneCount }, () => new THREE.Matrix4());
    const currentBones = Array.from({ length: boneCount }, () => new THREE.Matrix4());
    const previousBonesNode = uniformArray(previousBones, 'mat4');
    const bindMatrix = uniform(new THREE.Matrix4());
    const bindMatrixInverse = uniform(new THREE.Matrix4());
    {
        const index = attribute('skinIndex', 'uvec4');
        const weight = attribute('skinWeight', 'vec4');
        const bound = bindMatrix.mul(vec4(positionGeometry, 1));
        const bone = (i: THREE.Node<'uint'>) => previousBonesNode.element(int(i)) as unknown as THREE.Node<'mat4'>;
        const skinned = bone(index.x)
            .mul(bound)
            .mul(weight.x)
            .add(bone(index.y).mul(bound).mul(weight.y))
            .add(bone(index.z).mul(bound).mul(weight.z))
            .add(bone(index.w).mul(bound).mul(weight.w));
        oracle.instrument(limb, 1, bindMatrixInverse.mul(skinned).xyz);
    }

    //* Instanced Boxes
    const boxes = new THREE.InstancedMesh(new THREE.BoxGeometry(0.55, 0.55, 0.55), checkerMaterial(0x60c0e0, 0x304050, 3), INSTANCES);
    boxes.castShadow = true;
    boxes.frustumCulled = false;
    const previousInstances = Array.from({ length: INSTANCES }, () => new THREE.Matrix4());
    const previousInstancesNode = uniformArray(previousInstances, 'mat4');
    oracle.instrument(
        boxes,
        2,
        (previousInstancesNode.element(instanceIndex) as unknown as THREE.Node<'mat4'>).mul(vec4(positionGeometry, 1)).xyz,
    );

    //* Batched Shapes
    // One instance per geometry, in the order added, so the geometry's own
    // `labBatchId` attribute names its instance.
    const shapes = [new THREE.BoxGeometry(0.8, 0.8, 0.8), new THREE.SphereGeometry(0.5, 48, 24), new THREE.ConeGeometry(0.5, 1, 48)];
    const shapeIds = shapes.map((geometry, id) => {
        geometry.setAttribute('labBatchId', new THREE.Float32BufferAttribute(new Float32Array(geometry.attributes.position.count).fill(id), 1));
        return geometry;
    });
    const vertices = shapeIds.reduce((total, geometry) => total + geometry.attributes.position.count, 0);
    const indices = shapeIds.reduce((total, geometry) => total + (geometry.index?.count ?? 0), 0);
    const batched = new THREE.BatchedMesh(BATCHED, vertices, indices, checkerMaterial(0xd070a0, 0x403040, 4));
    batched.sortObjects = false;
    batched.perObjectFrustumCulled = false;
    batched.frustumCulled = false;
    batched.castShadow = true;
    for (const geometry of shapeIds) batched.addInstance(batched.addGeometry(geometry));
    const previousBatched = Array.from({ length: BATCHED }, () => new THREE.Matrix4());
    const previousBatchedNode = uniformArray(previousBatched, 'mat4');
    oracle.instrument(
        batched as unknown as THREE.Mesh,
        3,
        (previousBatchedNode.element(int(attribute('labBatchId', 'float'))) as unknown as THREE.Node<'mat4'>).mul(vec4(positionGeometry, 1)).xyz,
    );

    //* Texture-Driven Displacement (R7)
    const heightTexture = () => {
        const texture = new THREE.DataTexture(new Float32Array(HEIGHTS * HEIGHTS * 4), HEIGHTS, HEIGHTS, THREE.RGBAFormat, THREE.FloatType);
        texture.magFilter = texture.minFilter = THREE.NearestFilter;
        texture.needsUpdate = true;
        return texture;
    };
    const currentHeights = heightTexture();
    const previousHeights = heightTexture();
    const heightAt = (texture: THREE.DataTexture) => {
        const texel = ivec2(clamp(uv().mul(HEIGHTS).floor(), 0, HEIGHTS - 1));
        return (textureLoad(texture, texel) as unknown as THREE.Node<'vec4'>).x.mul(HEIGHT_SCALE);
    };
    const texturePlanes = (['texture-hook', 'texture-rigid', 'texture-auto'] as const).map((name, i) => {
        const plane = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 1.5, 64, 64), checkerMaterial(0x70d0a0, 0x304038, 8));
        const material = plane.material as THREE.NodeMaterial;
        material.positionNode = positionLocal.add(normalLocal.mul(heightAt(currentHeights)));
        if (name === 'texture-hook') MotionVectors.setPreviousPosition(material, positionPrevious.add(normalLocal.mul(heightAt(previousHeights))));
        if (name === 'texture-rigid') MotionVectors.setPreviousPosition(material, 'rigid');
        plane.position.set(-2.6 + i * 2.6, 3.4, -2.5);
        oracle.instrument(plane, 4 + i, positionGeometry.add(normalGeometry.mul(heightAt(previousHeights))));
        return plane;
    });
    /** A bump sliding across U as progress goes 0 → 1. */
    function fillHeights(texture: THREE.DataTexture, progress: number): void {
        const data = texture.image.data as Float32Array;
        const center = 0.2 + 0.6 * progress;
        for (let y = 0; y < HEIGHTS; y++)
            for (let x = 0; x < HEIGHTS; x++) {
                const du = (x + 0.5) / HEIGHTS - center;
                const dv = (y + 0.5) / HEIGHTS - 0.5;
                data[(y * HEIGHTS + x) * 4] = Math.exp(-(du * du + dv * dv) / 0.02);
            }
        texture.needsUpdate = true;
    }

    //* displacementMap with an Animated Scale (R9)
    const displacementTexture = heightTexture();
    fillHeights(displacementTexture, 0.5);
    const previousDisplacementScale = uniform(0);
    const displaced = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 1.5, 64, 64), checkerMaterial(0xa0a0e0, 0x303048, 8));
    Object.assign(displaced.material as THREE.MeshStandardNodeMaterial, { displacementMap: displacementTexture, displacementScale: 0, displacementBias: 0.05 });
    displaced.position.set(5.2, 3.4, -2.5);
    oracle.instrument(
        displaced,
        7,
        positionGeometry.add(
            normalGeometry.normalize().mul((texture(displacementTexture, uv()) as unknown as THREE.Node<'vec4'>).x.mul(previousDisplacementScale).add(0.05)),
        ),
    );
    const displacementScaleAt = (progress: number) => 0.1 + 1.5 * progress;

    scene.add(limb, boxes, batched as unknown as THREE.Object3D, ...texturePlanes, displaced);

    //* Poses
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3(1, 1, 1);
    const position = new THREE.Vector3();
    /** Poses every rig for `progress` and returns their matrices (bones relative to the bind pose). */
    function pose(progress: number, bones: THREE.Matrix4[], instances: THREE.Matrix4[], batch: THREE.Matrix4[]): void {
        joint.rotation.z = 1.1 * progress;
        root.rotation.y = 0.6 * progress;
        limb.updateMatrixWorld(true);
        limb.skeleton.bones.forEach((bone, i) => bones[i].multiplyMatrices(bone.matrixWorld, limb.skeleton.boneInverses[i]));
        for (let i = 0; i < INSTANCES; i++) {
            position.set(-1.5 + i * 0.6, 0.5 + progress * (0.4 + 0.25 * i), 0);
            quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), progress * (0.6 + 0.3 * i));
            instances[i].compose(position, quaternion, scale);
        }
        for (let i = 0; i < BATCHED; i++) {
            position.set(3 + i * 1.2 + progress * (0.8 + 0.4 * i), 0.6, -0.4 * i);
            quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), progress * 0.8 * i);
            batch[i].compose(position, quaternion, scale);
        }
    }
    const currentInstances = Array.from({ length: INSTANCES }, () => new THREE.Matrix4());
    const currentBatched = Array.from({ length: BATCHED }, () => new THREE.Matrix4());
    let lastFrame = Number.NaN;

    const segments = timeline([
        ['still', MOVE_START],
        ['move', MOVE_FRAMES],
        ['hold', 60],
    ]);
    return {
        name: 'rigs',
        scene,
        objects: { 0: 'static', 1: 'skinned', 2: 'instanced', 3: 'batched', 4: 'texture-hook', 5: 'texture-rigid', 6: 'texture-auto', 7: 'displacement-map' },
        end: segments[segments.length - 1].end,
        segments,
        update(frame, camera) {
            // Last frame's pose first (the oracle's previous), then this frame's,
            // which stays applied for three. A run that doesn't continue the
            // previous frame starts with previous = current.
            const previousProgress = progressAt(frame === lastFrame + 1 ? frame - 1 : frame);
            pose(previousProgress, previousBones, previousInstances, previousBatched);
            pose(progressAt(frame), currentBones, currentInstances, currentBatched);
            fillHeights(previousHeights, previousProgress);
            fillHeights(currentHeights, progressAt(frame));
            previousDisplacementScale.value = displacementScaleAt(previousProgress);
            (displaced.material as THREE.MeshStandardNodeMaterial).displacementScale = displacementScaleAt(progressAt(frame));
            lastFrame = frame;
            bindMatrix.value.copy(limb.bindMatrix);
            bindMatrixInverse.value.copy(limb.bindMatrixInverse);
            currentInstances.forEach((instance, i) => boxes.setMatrixAt(i, instance));
            boxes.instanceMatrix.needsUpdate = true;
            currentBatched.forEach((instance, i) => batched.setMatrixAt(i, matrix.copy(instance)));
            camera.position.set(0, 3, 9.5);
            camera.lookAt(0, 1.2, 0);
            return { segment: segmentAt(segments, frame).name, cut: false };
        },
    };
};
