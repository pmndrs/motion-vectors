import { PerspectiveCamera, Vector3, Vector4 } from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import { cameraMatrices } from '../src/three/MotionFromDepthPass.js';

/** A camera that moved, with its projection jittered by a sub-pixel offset like the lab's `applyJitter`. */
function jitteredCamera() {
    const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 200);
    camera.position.set(1, 2, 6);
    camera.lookAt(0, 0.5, 0);
    camera.updateMatrixWorld(true);
    const unjittered = camera.projectionMatrix.clone();
    camera.projectionMatrix.elements[8] += 2 * 0.3 / 1280;
    camera.projectionMatrix.elements[9] += 2 * -0.2 / 720;
    return { camera, unjittered };
}

const apply = (elements: ArrayLike<number>, v: Vector4) => {
    const e = Array.from(elements);
    return new Vector4(
        e[0] * v.x + e[4] * v.y + e[8] * v.z + e[12] * v.w,
        e[1] * v.x + e[5] * v.y + e[9] * v.z + e[13] * v.w,
        e[2] * v.x + e[6] * v.y + e[10] * v.z + e[14] * v.w,
        e[3] * v.x + e[7] * v.y + e[11] * v.z + e[15] * v.w,
    );
};

describe('cameraMatrices (MotionFromDepthPass)', () => {
    it('unprojects through the rendered (jittered) projection and reprojects through the unjittered one', () => {
        const { camera, unjittered } = jitteredCamera();
        const matrices = cameraMatrices(camera, unjittered, 1);
        const world = new Vector3(0.4, 0.7, -0.3);
        // What the rasterizer wrote: the point through the jittered projection.
        const rendered = new Vector4(world.x, world.y, world.z, 1).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
        const back = apply(matrices.inverseViewProjectionJittered, rendered);
        expect(back.x / back.w).toBeCloseTo(world.x, 5);
        expect(back.y / back.w).toBeCloseTo(world.y, 5);
        expect(back.z / back.w).toBeCloseTo(world.z, 5);
        // The motion's endpoints are jitter-free.
        const clean = apply(matrices.viewProjection, new Vector4(world.x, world.y, world.z, 1));
        const expected = new Vector4(world.x, world.y, world.z, 1).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(unjittered);
        expect(clean.x / clean.w).toBeCloseTo(expected.x / expected.w, 6);
        expect(clean.y / clean.w).toBeCloseTo(expected.y / expected.w, 6);
    });

    it("defaults to the camera's own projection and passes the view and clear depth through", () => {
        const { camera } = jitteredCamera();
        const matrices = cameraMatrices(camera, null, 0);
        const own = camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).elements;
        Array.from(matrices.viewProjection).forEach((value, i) => expect(value).toBeCloseTo(own[i], 9));
        expect(Array.from(matrices.view)).toEqual(camera.matrixWorldInverse.elements);
        expect(matrices.backgroundDepth).toBe(0);
    });
});
