import { describe, expect, it } from 'vitest';

import { CUT_MIN_SURFACE, CUT_THRESHOLD, cutVerdict } from '../src/math/cut.js';
import { decodeCutResult } from '../src/webgpu/CutDetector.js';
import { CUT_CONSTANTS_BYTES, CUT_DETECTION_SHADER, CUT_RESULT_WORDS } from '../src/webgpu/shaders/cutDetection.js';

describe('cutVerdict (CPU mirror of the GPU resolve)', () => {
    const pixels = 1000;

    it('fires above the threshold share of surface pixels, background excluded', () => {
        // 30 % of a half-sky frame's surface: a cut, though only 15 % of all pixels.
        const verdict = cutVerdict({ inconsistent: 150, surface: 500, pixels, hasPrevious: true });
        expect(verdict).toEqual({ cut: true, share: 0.3, surface: 0.5, hasPrevious: true });
        expect(cutVerdict({ inconsistent: 100, surface: 500, pixels, hasPrevious: true }).cut).toBe(false);
    });

    it('is strict at the threshold and keeps the measured defaults', () => {
        expect(CUT_THRESHOLD).toBe(0.2);
        expect(CUT_MIN_SURFACE).toBe(0.05);
        expect(cutVerdict({ inconsistent: 100, surface: 500, pixels, hasPrevious: true }).share).toBe(CUT_THRESHOLD);
        expect(cutVerdict({ inconsistent: 100, surface: 500, pixels, hasPrevious: true }, { threshold: 0.19 }).cut).toBe(true);
    });

    it('gives no verdict on an almost all-sky frame', () => {
        expect(cutVerdict({ inconsistent: 40, surface: 40, pixels, hasPrevious: true }).cut).toBe(false);
        expect(cutVerdict({ inconsistent: 50, surface: 50, pixels, hasPrevious: true }).cut).toBe(true);
        expect(cutVerdict({ inconsistent: 0, surface: 0, pixels, hasPrevious: true })).toMatchObject({ cut: false, share: 0, surface: 0 });
    });

    it('resets when there is no previous frame, whatever the counts', () => {
        expect(cutVerdict({ inconsistent: 0, surface: 0, pixels, hasPrevious: false })).toEqual({ cut: true, share: 0, surface: 0, hasPrevious: false });
    });
});

describe('decodeCutResult', () => {
    it('decodes the resolve words into the verdict and its counts', () => {
        expect(decodeCutResult(new Uint32Array([300, 600, 1, 1]), 1000)).toEqual({
            cut: true,
            share: 0.5,
            surface: 0.6,
            hasPrevious: true,
            inconsistentPixels: 300,
            surfacePixels: 600,
            pixels: 1000,
        });
        expect(decodeCutResult(new Uint32Array([0, 0, 1, 0]), 1000)).toMatchObject({ cut: true, share: 0, hasPrevious: false });
    });
});

describe('cut shader structure', () => {
    it('has an 8×8 count entry and a single-invocation resolve over contiguous bindings', () => {
        expect(CUT_DETECTION_SHADER).toMatch(/@compute @workgroup_size\(8, 8\)\s*\nfn count\(/);
        expect(CUT_DETECTION_SHADER).toMatch(/@compute @workgroup_size\(1\)\s*\nfn resolve\(/);
        const bindings = [...CUT_DETECTION_SHADER.matchAll(/@binding\((\d+)\)/g)].map((match) => Number(match[1]));
        expect(bindings).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
        expect(CUT_DETECTION_SHADER).toMatch(/@binding\(0\) var<uniform>/);
        expect(CUT_DETECTION_SHADER).toMatch(new RegExp(`array<u32, ${CUT_RESULT_WORDS}>`));
        expect(CUT_DETECTION_SHADER).not.toMatch(/\b(let|var) (target|filter|enum|mat)\b/);
    });

    it('declares a uniform struct the CPU layout fills exactly', () => {
        const struct = CUT_DETECTION_SHADER.match(/struct CutConstants \{([\s\S]*?)\};/)![1];
        const bytes = [...struct.matchAll(/: (vec2f|f32|u32)/g)].reduce((total, [, type]) => total + (type === 'vec2f' ? 8 : 4), 0);
        expect(bytes).toBe(CUT_CONSTANTS_BYTES);
    });

    it('zeroes its counters in the resolve, so frames need no clear', () => {
        expect(CUT_DETECTION_SHADER).toMatch(/atomicExchange\(&counters\[0\], 0u\)/);
        expect(CUT_DETECTION_SHADER).toMatch(/atomicExchange\(&counters\[1\], 0u\)/);
    });
});
