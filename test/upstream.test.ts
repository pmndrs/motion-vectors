import { describe, expect, it } from 'vitest';

import { threeHandles, UPSTREAM } from '../src/three/upstream.js';

describe('upstream table (R12)', () => {
    it('defers a case to three from the release that fixed it, never before', () => {
        expect(threeHandles('projection', 186)).toBe(false);
        expect(threeHandles('projection', 187)).toBe(true);
        expect(threeHandles('instancing', 185)).toBe(false);
        expect(threeHandles('instancing', 186)).toBe(true);
        expect(threeHandles('positionNode', 999)).toBe(false);
    });

    it('keeps every open #17/#18 case open until the lab measures three exact', () => {
        for (const id of ['positionNode', 'morph', 'displacementMap', 'cameraHistory', 'skinningHistory'] as const) expect(UPSTREAM[id].fixedIn).toBeNull();
    });

    it("never claims a fix older than three's velocity node", () => {
        for (const entry of Object.values(UPSTREAM)) if (entry.fixedIn !== null) expect(entry.fixedIn).toBeGreaterThanOrEqual(184);
    });
});
