import { describe, expect, it } from 'vitest';

import { FrameHistory } from '../src/three/history.js';

// The node's history rules (PLAN R1/R4), on plain numbers: every draw in a
// frame sees the value from the previous frame's draw, whatever the draw count.
describe('FrameHistory', () => {
    const box = () => ({});
    const draw = (history: FrameHistory<number>, key: object, frame: number, value: number) => {
        const previous = history.before(key, frame, value);
        history.after(key, value);
        return previous;
    };

    it('reports the current value the first time a key is drawn (zero motion)', () => {
        const history = new FrameHistory<number>((v) => v);
        expect(draw(history, box(), 1, 5)).toBe(5);
    });

    it('hands over once per frame, not per draw', () => {
        const history = new FrameHistory<number>((v) => v);
        const key = box();
        draw(history, key, 1, 10);
        // A shadow pass and a main pass in frame 2: both see frame 1's value,
        // even though the first draw already recorded 20.
        expect(draw(history, key, 2, 20)).toBe(10);
        expect(draw(history, key, 2, 20)).toBe(10);
        expect(draw(history, key, 3, 30)).toBe(20);
    });

    it('starts over when the key was not drawn last frame', () => {
        const history = new FrameHistory<number>((v) => v);
        const key = box();
        draw(history, key, 1, 10);
        // Culled in frame 2: its capture is from frame 1, not frame 2.
        expect(draw(history, key, 3, 30)).toBe(30);
        expect(draw(history, key, 4, 40)).toBe(30);
    });

    it('starts over in a new epoch (reset)', () => {
        const history = new FrameHistory<number>((v) => v);
        const key = box();
        expect(history.before(key, 1, 10, 0)).toBe(10);
        history.after(key, 10);
        expect(history.before(key, 2, 20, 1)).toBe(20);
    });

    it('clones, so live objects never alias history', () => {
        const history = new FrameHistory<number[]>((v) => [...v]);
        const key = box();
        const live = [1];
        history.before(key, 1, live);
        history.after(key, live);
        live[0] = 2;
        expect(history.before(key, 2, live)).toEqual([1]);
    });
});
