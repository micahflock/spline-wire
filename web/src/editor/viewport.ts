// The pin editor's view of the photo: which part of it is on screen.
//
// A ViewBox is a rectangle in photo pixels, exactly an SVG viewBox. The
// editor's SVG uses the photo's own coordinates, so zooming and panning only
// change this box, and the markers never need converting. The screen element
// has a size in CSS px; `scale` (CSS px per photo px) links the two.
//
// All functions are pure: the gesture layer (gestures.ts) decides what the
// fingers or the mouse did, this decides what that does to the view.

export interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Pt {
  x: number;
  y: number;
}

/** How far in and out the view may go. In: the view at least ZOOM_IN_RADII
 * marker radii wide, so a pin never fills the screen. Out: twice the photo. */
export const ZOOM_IN_RADII = 6;
export const ZOOM_OUT_PHOTOS = 2;

/** CSS px per photo px. The box always has the element's aspect (fit() and
 * the functions below keep it so), so one number does. */
export function scale(box: ViewBox, el: Size): number {
  return el.width / box.w;
}

/** A screen point (CSS px from the element's top-left) in photo pixels. */
export function toPhoto(box: ViewBox, el: Size, screen: Pt): Pt {
  const k = scale(box, el);
  return { x: box.x + screen.x / k, y: box.y + screen.y / k };
}

export function toScreen(box: ViewBox, el: Size, photo: Pt): Pt {
  const k = scale(box, el);
  return { x: (photo.x - box.x) * k, y: (photo.y - box.y) * k };
}

/** The smallest box with the element's aspect that contains `box`, centred on it. */
export function fit(box: ViewBox, el: Size): ViewBox {
  const aspect = el.width / el.height;
  let { x, y, w, h } = box;
  if (w / h < aspect) {
    const nw = h * aspect;
    x -= (nw - w) / 2;
    w = nw;
  } else {
    const nh = w / aspect;
    y -= (nh - h) / 2;
    h = nh;
  }
  return { x, y, w, h };
}

/** The whole photo's width across the element, centred top to bottom: a photo taller than the element
 * runs past its top and bottom rather than leaving bars at its sides (the phone, a photo with no chain). */
export function fitWidth(photo: Size, el: Size): ViewBox {
  const h = photo.width * (el.height / el.width);
  return { x: 0, y: (photo.height - h) / 2, w: photo.width, h };
}

/** Zoom by `factor` (>1 zooms out) keeping the photo point under `at` (screen px) where it is. */
export function zoomAt(box: ViewBox, el: Size, at: Pt, factor: number, limits: ZoomLimits): ViewBox {
  const w = clampWidth(box.w * factor, limits);
  const k = w / box.w;
  const p = toPhoto(box, el, at);
  return { x: p.x - (p.x - box.x) * k, y: p.y - (p.y - box.y) * k, w, h: box.h * k };
}

/** Move the view so the content follows the pointer: a drag of (dx, dy) screen px. */
export function pan(box: ViewBox, el: Size, dx: number, dy: number): ViewBox {
  const k = scale(box, el);
  return { ...box, x: box.x - dx / k, y: box.y - dy / k };
}

/** Two fingers: the view that keeps the photo points first under the fingers
 * (in `start`) under them now, as near as a zoom and a pan can (no rotation).
 * Each finger pair is given in screen px. Computed from the gesture's start
 * every time, so it does not drift. */
export function pinch(start: ViewBox, el: Size, from: [Pt, Pt], to: [Pt, Pt], limits: ZoomLimits): ViewBox {
  const d0 = Math.hypot(from[1].x - from[0].x, from[1].y - from[0].y);
  const d1 = Math.hypot(to[1].x - to[0].x, to[1].y - to[0].y);
  const k0 = scale(start, el);
  const k1 = el.width / clampWidth(start.w * (d1 > 0 && d0 > 0 ? d0 / d1 : 1), limits);
  // The photo point at the fingers' first midpoint stays under their current midpoint.
  const m0 = mid(from), m1 = mid(to);
  const p = { x: start.x + m0.x / k0, y: start.y + m0.y / k0 };
  const w = el.width / k1;
  return { x: p.x - m1.x / k1, y: p.y - m1.y / k1, w, h: w * (el.height / el.width) };
}

export interface ZoomLimits {
  minW: number;
  maxW: number;
}

export function zoomLimits(photo: Size, markerRadius: number): ZoomLimits {
  return { minW: ZOOM_IN_RADII * markerRadius, maxW: ZOOM_OUT_PHOTOS * Math.max(photo.width, photo.height) };
}

/** Keep at least `keep` (a fraction of the view) of the photo on screen, so a
 * fling cannot lose it. */
export function keepPhotoInView(box: ViewBox, photo: Size, keep = 0.2): ViewBox {
  const mx = keep * box.w, my = keep * box.h;
  return {
    ...box,
    x: Math.min(Math.max(box.x, mx - box.w), photo.width - mx),
    y: Math.min(Math.max(box.y, my - box.h), photo.height - my),
  };
}

/** The box that frames the chain, padded. Stray detections can be anywhere on
 * the table, so they only count when there is no chain to frame. */
export function chainBox(
  view: { width: number; height: number; r: number; pins: Pt[]; unused: Pt[]; removed: Pt[] },
): ViewBox {
  const pts = view.pins.length >= 3 ? [...view.pins, ...view.removed] : [...view.pins, ...view.unused, ...view.removed];
  if (!pts.length) return { x: 0, y: 0, w: view.width, h: view.height };
  const xs = pts.map((m) => m.x), ys = pts.map((m) => m.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = Math.max(0.12 * Math.max(x1 - x0, y1 - y0), 4 * view.r);
  return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
}

function clampWidth(w: number, l: ZoomLimits): number {
  return Math.min(Math.max(w, l.minW), l.maxW);
}

function mid(p: [Pt, Pt]): Pt {
  return { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
}
