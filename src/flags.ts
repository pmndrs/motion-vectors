/**
 * The motion contract's `.z` flag values (CONTRACT.md). Its own module so
 * every subpath can import it without the core's CPU math.
 */

/**
 * The contract's `.z`: where a pixel's motion comes from and how far to trust
 * it (CONTRACT.md, "Flag"). Stored as an exact small integer in `rgba16float`;
 * read it rounded (`round(texel.z)` in WGSL). Ordered by trust: anything but
 * {@link MotionFlag.Exact} leaves some real motion out.
 */
export const MotionFlag = {
    /** Nothing was drawn here (the cleared target). `.xy` and `.w` are producer-defined; motion from depth gives the far plane's camera motion. */
    NoSurface: 0,
    /** Geometric motion of the surface point: camera, object and deformation. */
    Exact: 1,
    /** Camera motion of a point assumed static (motion from depth): any motion of its own is missing. */
    CameraOnly: 2,
    /** Camera and object motion, but the deformation's own motion is missing (the deformed position stands in for both frames). */
    Rigid: 3,
    /** The point had no previous position: behind the previous camera, not drawn last frame, or the first frame after a reset. `.xy` and `.w` are 0. */
    NoPrevious: 4,
} as const;
/** A {@link MotionFlag} value. */
export type MotionFlag = (typeof MotionFlag)[keyof typeof MotionFlag];
