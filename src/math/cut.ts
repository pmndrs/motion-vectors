/**
 * Cut (scene-change) verdict from the depth-consistency counts, the CPU
 * reference for the GPU detector (`/webgpu` `CutDetector`). The signal is the
 * share of **surface** pixels that no motion candidate maps onto a matching
 * previous depth (MOTION.md §7, "Cut detection, side by side"). Background
 * pixels (no surface) carry no evidence and are left out of both counts.
 */

/** Share of inconsistent surface pixels above which a frame is a cut (MOTION.md §7: cuts 0.30–1.0, look-alikes ≤ 0.16). */
export const CUT_THRESHOLD = 0.2;
/** Smallest surface share (of all pixels) that gets a verdict; below it the frame is mostly sky and never a cut. */
export const CUT_MIN_SURFACE = 0.05;
/** Relative tolerance of the previous-depth match. */
export const CUT_DEPTH_TOLERANCE = 0.05;

/** Options of a cut verdict (defaults: the measured ones above). */
export interface CutOptions {
    /** Inconsistent share above which the frame is a cut (default {@link CUT_THRESHOLD}). */
    threshold?: number;
    /** Smallest surface share that gets a verdict (default {@link CUT_MIN_SURFACE}). */
    minSurface?: number;
}

/** What the counts say about one frame. */
export interface CutVerdict {
    /** Reset history this frame: a cut, or no previous frame to keep. */
    cut: boolean;
    /** Inconsistent surface pixels over surface pixels (0 with no surface). */
    share: number;
    /** Surface pixels over all pixels. */
    surface: number;
    /** A previous frame existed, so the counts mean something. */
    hasPrevious: boolean;
}

/**
 * The verdict for one frame's counts, exactly as the GPU resolve computes it.
 * @param counts - Inconsistent and surface pixel counts, the frame's pixel total, and whether a previous frame existed
 * @param options - Threshold and minimum surface
 * @returns The verdict
 */
export function cutVerdict(
    counts: { inconsistent: number; surface: number; pixels: number; hasPrevious: boolean },
    options: CutOptions = {},
): CutVerdict {
    const threshold = options.threshold ?? CUT_THRESHOLD;
    const minSurface = options.minSurface ?? CUT_MIN_SURFACE;
    const surface = counts.pixels > 0 ? counts.surface / counts.pixels : 0;
    const share = counts.surface > 0 ? counts.inconsistent / counts.surface : 0;
    const judged = counts.hasPrevious && surface >= minSurface;
    return { cut: !counts.hasPrevious || (judged && share > threshold), share, surface, hasPrevious: counts.hasPrevious };
}
