import { afterEach, describe, expect, it, vi } from 'vitest';

import { decodeWarpingResult, WarpingError } from '../src/webgpu/WarpingError.js';
import {
    WARPING_CONSTANTS_BYTES,
    WARPING_ERROR_SHADER,
    WARPING_RESAMPLE_FILTERS,
    WARPING_WORDS,
    resolveWarpingResample,
    warpingErrorShader,
} from '../src/webgpu/shaders/warpingError.js';
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

describe('warping error resample option', () => {
    it('defaults to bilinear and accepts every listed filter', () => {
        expect(resolveWarpingResample(undefined)).toBe('bilinear');
        expect(WARPING_RESAMPLE_FILTERS).toEqual(['bilinear', 'catmull-rom']);
        for (const name of WARPING_RESAMPLE_FILTERS) expect(resolveWarpingResample(name)).toBe(name);
    });

    it('rejects anything else, before touching the device', () => {
        for (const value of ['lanczos2', 'Catmull-Rom', 'bicubic', '', null, 1]) expect(() => resolveWarpingResample(value)).toThrow(/resample must be one of/);
        // An empty device object: the option is checked before any GPU call.
        expect(() => new WarpingError({} as GPUDevice, { resample: 'bicubic' as never })).toThrow(/resample must be one of 'bilinear', 'catmull-rom'/);
    });

    afterEach(() => vi.unstubAllGlobals());

    it('creates a sampler only for bilinear', () => {
        vi.stubGlobal('GPUBufferUsage', { UNIFORM: 0, STORAGE: 0, COPY_SRC: 0, COPY_DST: 0, MAP_READ: 0 });
        vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 0, STORAGE_BINDING: 0 });
        const created: string[] = [];
        const device = {
            createBuffer: () => ({}),
            createTexture: () => ({}),
            createSampler: () => created.push('sampler'),
        } as unknown as GPUDevice;
        expect(new WarpingError(device).resample).toBe('bilinear');
        expect(created).toEqual(['sampler']);
        expect(new WarpingError(device, { resample: 'catmull-rom' }).resample).toBe('catmull-rom');
        expect(created).toEqual(['sampler']);
    });
});

describe('catmull-rom warping shader', () => {
    const code = warpingErrorShader('catmull-rom');

    it('keeps the default shader byte-for-byte (bilinear is unchanged)', () => {
        expect(warpingErrorShader()).toBe(WARPING_ERROR_SHADER);
        expect(warpingErrorShader('bilinear')).toBe(WARPING_ERROR_SHADER);
        expect(WARPING_ERROR_SHADER).toMatch(/textureSampleLevel\(previous, linearSampler, prevUV, 0\.0\)/);
    });

    it('loads the previous image without a sampler, so unfilterable formats bind', () => {
        const bindings = [...code.matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
        expect(bindings).toEqual([0, 1, 2, 4, 5, 6, 7]);
        expect(code).not.toMatch(/\bsampler\b|textureSample/);
        expect(code).toMatch(/conditionedLuma\(samplePrevious\(prevUV\)\)/);
        expect(code).toMatch(/for \(var j = 0; j < 4; j\+\+\)[\s\S]*for \(var i = 0; i < 4; i\+\+\)[\s\S]*textureLoad\(previous,/);
        expect(code).toMatch(/const DERING : bool = true;/);
        expect(warpingErrorShader('catmull-rom', false)).toMatch(/const DERING : bool = false;/);
        expect(code).not.toMatch(/\b(let|var) (target|filter|enum|mat)\b/);
    });

    it('shares the uniform layout and the luma domain with the default', () => {
        const struct = /struct WarpConstants \{[\s\S]*?\};/;
        expect(code.match(struct)![0]).toBe(WARPING_ERROR_SHADER.match(struct)![0]);
        const luma = /fn conditionedLuma[\s\S]*?\n}/;
        expect(code.match(luma)![0]).toBe(WARPING_ERROR_SHADER.match(luma)![0]);
    });

    // The shader's own weight polynomials, evaluated on the CPU (they are plain arithmetic in both languages).
    const body = code.match(/fn tapWeights\(t : f32\) -> vec4f \{[\s\S]*?return vec4f\(([\s\S]*?)\);\n}/)![1];
    const tapWeights = new Function('t', `const t2 = t * t; const t3 = t2 * t; return [${body}];`) as (t: number) => number[];
    const phases = Array.from({ length: 33 }, (_, k) => k / 32);

    it('interpolates: weights sum to 1, and a whole-texel sample is that texel', () => {
        for (const t of phases) expect(tapWeights(t).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
        tapWeights(0).forEach((w, k) => expect(w).toBeCloseTo([0, 1, 0, 0][k], 12));
        tapWeights(1).forEach((w, k) => expect(w).toBeCloseTo([0, 0, 1, 0][k], 12));
    });

    it('is symmetric and reproduces quadratics exactly (taps at −1, 0, 1, 2)', () => {
        for (const t of phases) {
            const w = tapWeights(t);
            tapWeights(1 - t).forEach((m, k) => expect(m).toBeCloseTo(w[3 - k], 12));
            const at = (f: (x: number) => number) => w.reduce((sum, wk, k) => sum + wk * f(k - 1), 0);
            expect(at((x) => x)).toBeCloseTo(t, 12);
            expect(at((x) => x * x)).toBeCloseTo(t * t, 12);
        }
    });
});
