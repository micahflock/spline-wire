// Focal lengths between stills and video frames (web-build.md D17, step 5).
//
// The core takes a focal length either in pixels of the image it measures
// (live.analyse, for video frames) or as a 35 mm equivalent (a photo's
// measurement), and converts one to the other by the image's diagonal, as
// splinewire/camera.py does: focal_px = f35 × diagonal_px / 43.27 mm.
//
// That conversion assumes the image shows the whole field of view the 35 mm
// figure was given for. A phone's still is the full 4:3 sensor; a video frame
// may be the same 4:3 field (S3: iPhone streams 3024×4032), or a 16:9 crop of
// it at full width. So a focal length learned from a still is carried to a
// frame by the long sides, never by the diagonals: that is right for both.

export const FULL_FRAME_DIAGONAL_MM = Math.hypot(36, 24);

export interface Size {
  width: number;
  height: number;
}

export function focalPxFrom35(f35: number, size: Size): number {
  return (f35 * Math.hypot(size.width, size.height)) / FULL_FRAME_DIAGONAL_MM;
}

export function focal35FromPx(px: number, size: Size): number {
  return (px * FULL_FRAME_DIAGONAL_MM) / Math.hypot(size.width, size.height);
}

/** A phone camera's still when its size is not known: the full 4:3 sensor,
 * oriented like the frame. */
export function fourThirds(frame: Size): Size {
  const long = Math.max(frame.width, frame.height);
  return frame.width >= frame.height ? { width: long, height: (long * 3) / 4 } : { width: (long * 3) / 4, height: long };
}

/** The focal length in px of a video frame, from a 35 mm equivalent that holds
 * for `still` (the size of the photo it came from; null: a 4:3 still). The
 * frame is assumed to span the still's long side: true for a 4:3 stream and
 * for a 16:9 crop at full width. Stabilisation can narrow the frame further
 * (S5 assumed 10%): the camera view's live-view scale (fovRatio) measures it. */
export function frameFocalPx(f35: number, still: Size | null, frame: Size): number {
  const s = still ?? fourThirds(frame);
  return (focalPxFrom35(f35, s) * Math.max(frame.width, frame.height)) / Math.max(s.width, s.height);
}

/** The same focal length as a 35 mm equivalent for an image of the frame's
 * size: what a full-resolution frame saved as a photo is measured with. */
export function frame35(f35: number, still: Size | null, frame: Size): number {
  return focal35FromPx(frameFocalPx(f35, still, frame), frame);
}

/** How the live view's scale compares with the still's: 1 when the frame shows
 * the still's field of view (the still's focal length then applies to frames
 * unchanged); 1.1 when the frame is cropped 10% tighter. From the median link
 * length of the chain in each, in their own pixels, normalised by the long
 * sides. null with fewer than 3 pins in either. */
export function fovRatio(frameChain: readonly [number, number][], frame: Size, stillPins: readonly [number, number][], still: Size): number | null {
  const link = (pts: readonly [number, number][]) => {
    if (pts.length < 3) return null;
    const d = pts.slice(1).map((p, k) => Math.hypot(p[0] - pts[k]![0], p[1] - pts[k]![1])).sort((a, b) => a - b);
    return d[d.length >> 1]!;
  };
  const a = link(frameChain), b = link(stillPins);
  if (!a || !b) return null;
  const perFrame = a / Math.max(frame.width, frame.height), perStill = b / Math.max(still.width, still.height);
  return perFrame / perStill;
}
