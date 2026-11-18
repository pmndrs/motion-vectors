import { describe, expect, it } from 'vitest';

import { decodeWarpingResult } from '../src/webgpu/WarpingError.js';
import { WARPING_CONSTANTS_BYTES, WARPING_ERROR_SHADER, WARPING_WORDS } from '../src/webgpu/shaders/warpingError.js';
import { MOTION_RESIDUAL_SHADER } from '../src/webgpu/shaders/motionResidual.js';

describe('decodeWarpingResult', () => {
    it('decodes the mean over scored pixels and the shares of all pixels', () => {
        // 500 scored pixels at a mean error of 0.01 (×1024 fixed point).
        const result = decodeWarpingResult(new Uint32Array([5120, 0, 500, 40, 300, 150, 50, 1000]), true);
        expect(result.meanError).toBeCloseTo(0.01, 9);
        expect(result).toMatchObject({ hasPrevious: true, scored: 0.5, over: 0.04, occluded: 0.3, offscreen: 0.15, noPrevious: 0.05 });
        expect(result.counts.pixels).toBe(1000);
    });

    it('carries the 64-bit sum (8K-sized images cannot overflow it)', () => {
        const result = decodeWarpingResult(new Uint32Array([1024, 3, 1, 0, 0, 0, 0, 1]), true);
        expect(result.counts.errorSum).toBe(3 * 2 ** 32 + 1024);
        expect(result.meanError).toBe((3 * 2 ** 32 + 1024) / 1024);
    });

    it('reads zero on an unmeasured frame', () => {
        expect(decodeWarpingResult(new Uint32Array(WARPING_WORDS), false)).toMatchObject({ hasPrevious: false, meanError: 0, scored: 0 });
    });
});

describe('warping error shader structure', () => {
    it('has one 8×8 entry over contiguous bindings, the UBO at 0', () => {
        expect(WARPING_ERROR_SHADER).toMatch(/@compute @workgroup_size\(8, 8\)\s*\nfn main\(/);
        const bindings = [...WARPING_ERROR_SHADER.matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
        expect(bindings).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
        expect(WARPING_ERROR_SHADER).toMatch(/@binding\(0\) var<uniform>/);
        expect(WARPING_ERROR_SHADER).toMatch(new RegExp(`array<atomic<u32>, ${WARPING_WORDS}>`));
        expect(WARPING_ERROR_SHADER).not.toMatch(/\b(let|var) (target|filter|enum|mat)\b/);
    });

    it('declares a uniform struct the CPU layout fills exactly', () => {
        const struct = WARPING_ERROR_SHADER.match(/struct WarpConstants \{([\s\S]*?)\};/)![1];
        const bytes = [...struct.matchAll(/: (vec2f|f32|u32)/g)].reduce((total, [, type]) => total + (type === 'vec2f' ? 8 : 4), 0);
        expect(bytes).toBe(WARPING_CONSTANTS_BYTES);
    });

    it('scores in the same domain and rounding as the warp residual (the lab bars them equal)', () => {
        const luma = /fn conditionedLuma[\s\S]*?\n}/;
        expect(WARPING_ERROR_SHADER.match(luma)![0]).toBe(MOTION_RESIDUAL_SHADER.match(luma)![0]);
        expect(WARPING_ERROR_SHADER).toMatch(/u32\(round\(error \* FIXED\)\)/);
        expect(MOTION_RESIDUAL_SHADER).toMatch(/u32\(round\(residual\[k\] \* FIXED\)\)/);
    });
});
