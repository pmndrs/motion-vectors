/**
 * Frame-keyed history: the value something had during the previous frame,
 * per key (an object, a camera), however many times a frame renders it.
 *
 * three's velocity keys its history on the draw or the animation-loop tick, so
 * a second render in one tick, or a second draw of one object (shadow pass,
 * second view), overwrites last frame's state (docs/MOTION.md §7, PLAN R1/R4).
 * Here, `before()` hands over to a new frame only when the frame number changes,
 * and `after()` records what was drawn, so every draw in a frame sees the same
 * previous value. Frames are consecutive integers: a key whose last capture is
 * older than the previous frame reports its current value (zero motion).
 */
export class FrameHistory<T> {
    private readonly _states = new WeakMap<object, { frame: number; epoch: number; previous: T; known: boolean; captured: T | undefined; capturedFrame: number }>();

    /**
     * @param clone - Copies a value (history must not alias live state)
     */
    constructor(private readonly clone: (value: T) => T) {}

    /**
     * The previous frame's value for `key`, called before each draw.
     * @param key - What the history belongs to
     * @param frame - The current frame number
     * @param current - The value now (the previous value of a key seen for the first time)
     * @param epoch - History generation: a key last seen in another epoch starts over (a reset)
     * @returns The value `key` had when last drawn in an earlier frame
     */
    before(key: object, frame: number, current: T, epoch = 0): T {
        let state = this._states.get(key);
        if (state === undefined || state.epoch !== epoch) {
            state = { frame, epoch, previous: this.clone(current), known: false, captured: undefined, capturedFrame: -1 };
            this._states.set(key, state);
        } else if (state.frame !== frame) {
            // A key not drawn last frame (culled, hidden) has no previous value:
            // the capture is from an older frame, so start over (zero motion).
            const fresh = state.captured !== undefined && state.capturedFrame === frame - 1;
            state.frame = frame;
            state.known = fresh;
            state.previous = fresh ? state.captured! : this.clone(current);
        }
        return state.previous;
    }

    /**
     * Whether the value {@link FrameHistory.before} last handed out for `key` is
     * a real previous value, and not the current one standing in for a key seen
     * for the first time, after a reset, or not drawn in the previous frame.
     * @param key - What the history belongs to
     * @returns `false` until `before()` has run for `key` in this epoch
     */
    known(key: object): boolean {
        return this._states.get(key)?.known ?? false;
    }

    /**
     * Records the value `key` was drawn with, called after each draw.
     * @param key - What the history belongs to
     * @param value - The value used this frame
     */
    after(key: object, value: T): void {
        const state = this._states.get(key);
        if (state === undefined) return;
        state.captured = this.clone(value);
        state.capturedFrame = state.frame;
    }
}
