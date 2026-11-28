// The photo with its pins, as both pages show it for editing: an SVG whose
// viewBox is the part of the photo on screen (src/editor/viewport.ts), the
// photo as its <image>, and the markers (draw.ts) in its <g>. Pointer events
// go through src/editor/gestures.ts, so the phone's touch (tap, drag, pinch)
// and the desktop's mouse (click, drag, wheel) are one code path.
//
// It knows nothing of photo records or the core: the page says what to show
// and is told where the person tapped or clicked, in photo px, with the
// target size for that pointer (MIN_TARGET_PX) converted to photo px.

import type { PhotoView } from "../core/types";
import { MIN_TARGET_PX } from "../editor/edits";
import { Gestures, wheelFactor, type Intent, type PointerInput, type PointerKind } from "../editor/gestures";
import {
  chainBox, fit, fitWidth, keepPhotoInView, scale, toPhoto, zoomAt, zoomLimits, type Pt, type Size, type ViewBox,
} from "../editor/viewport";
import { overlaySvg, type Extra } from "./draw";

/** Where pixel (0, 0)'s top-left corner is in photo px (setImage). */
export const PIXEL_ORIGIN = -0.5;

export type TapHandler = (at: Pt, slackPhotoPx: number, kind: PointerKind) => void;

export class PhotoEditor {
  private box: ViewBox | null = null;
  private pixels: Size | null = null;
  private view: PhotoView | null = null;
  private readonly image: SVGImageElement;
  private readonly layer: SVGGElement;
  private readonly gestures: Gestures;

  /** `svg` holds an <image> and a <g> for the markers. `wholePhoto`: how a photo with no chain in it is framed.
   * "contain" (the desktop page): the stray dots found, else all of the photo. "width" (the phone): the photo's
   * width across the view, whatever strays were found (framing them ran past the photo's edge). */
  constructor(private readonly svg: SVGSVGElement, private readonly onTap: TapHandler,
    private readonly opts: { wholePhoto?: "contain" | "width" } = {}) {
    this.image = svg.querySelector("image")!;
    this.layer = svg.querySelector("g")!;
    this.gestures = new Gestures(() => this.box ?? { x: 0, y: 0, w: 1, h: 1 }, () => this.el(), () => this.limits());
    svg.addEventListener("pointerdown", (e) => {
      if (!this.box || (e.pointerType === "mouse" && e.button !== 0)) return;
      svg.setPointerCapture(e.pointerId);
      this.intent(this.gestures.pointerDown(this.pointer(e)));
    });
    svg.addEventListener("pointermove", (e) => this.intent(this.gestures.pointerMove(this.pointer(e))));
    svg.addEventListener("pointerup", (e) => this.intent(this.gestures.pointerUp(this.pointer(e))));
    svg.addEventListener("pointercancel", () => this.gestures.cancel());
    svg.addEventListener("wheel", (e) => {
      const size = this.size();
      if (!this.box || !size) return;
      e.preventDefault();
      const b = svg.getBoundingClientRect();
      this.box = keepPhotoInView(zoomAt(this.box, this.el(), { x: e.clientX - b.left, y: e.clientY - b.top },
        wheelFactor(e.deltaY, e.deltaMode, e.ctrlKey), this.limits()), size);
      this.apply();
    }, { passive: false });
  }

  /** Another photo: nothing shown until setImage or show. */
  clear(): void {
    this.image.removeAttribute("href");
    this.box = null;
    this.pixels = null;
    this.view = null;
    this.layer.innerHTML = "";
  }

  /** The photo to draw under the markers: a downscaled copy (`url`) stretched
   * over the original's pixel size, which is what the markers are in. Detection
   * puts a pixel's centre at its integer coordinates (OpenCV's convention), and
   * an SVG image at 0 puts it half a pixel further on: the photo starts at -0.5. */
  setImage(url: string | null, pixels: Size | null): void {
    if (url && pixels) {
      this.image.setAttribute("href", url);
      this.image.setAttribute("x", String(PIXEL_ORIGIN));
      this.image.setAttribute("y", String(PIXEL_ORIGIN));
      this.image.setAttribute("width", String(pixels.width));
      this.image.setAttribute("height", String(pixels.height));
    } else {
      this.image.removeAttribute("href");
    }
    this.pixels = pixels;
  }

  /** Draw a measurement's markers. The first one with markers frames the chain;
   * later ones (after an edit) keep the view where the person left it. */
  show(view: PhotoView | null, extra: Extra = {}): void {
    const had = this.view;
    this.view = view;
    this.layer.innerHTML = view ? overlaySvg(view, extra) : "";
    if (view && (!had || !this.box)) this.frame();
  }

  /** Frame the chain (without one, see wholePhoto). */
  frame(): void {
    const size = this.size();
    if (!size) return;
    const v = this.view;
    this.box = this.opts.wholePhoto === "width" && !v?.pins.length ? fitWidth(size, this.el())
      : v && (v.pins.length || v.unused.length) ? fit(chainBox(v), this.el())
      : fit({ x: 0, y: 0, w: size.width, h: size.height }, this.el());
    this.apply();
  }

  /** The element changed size (window resize, layout): keep the same part in view. */
  refit(): void {
    if (!this.box) return;
    this.box = fit(this.box, this.el());
    this.apply();
  }

  get viewBox(): ViewBox | null {
    return this.box;
  }

  private size(): Size | null {
    const v = this.view;
    return v ? { width: v.width, height: v.height } : this.pixels;
  }

  private el(): Size {
    const b = this.svg.getBoundingClientRect();
    return { width: Math.max(1, b.width), height: Math.max(1, b.height) };
  }

  private limits() {
    const size = this.size() ?? { width: 1000, height: 1000 };
    return zoomLimits(size, this.view?.r ?? Math.max(size.width, size.height) / 200);
  }

  private apply(): void {
    if (this.box) this.svg.setAttribute("viewBox", `${this.box.x} ${this.box.y} ${this.box.w} ${this.box.h}`);
  }

  private pointer(e: PointerEvent): PointerInput {
    const b = this.svg.getBoundingClientRect();
    const kind: PointerKind = e.pointerType === "touch" || e.pointerType === "pen" ? e.pointerType : "mouse";
    return { id: e.pointerId, kind, x: e.clientX - b.left, y: e.clientY - b.top, t: e.timeStamp };
  }

  private intent(i: Intent): void {
    if (i.type === "view") {
      const size = this.size();
      this.box = size ? keepPhotoInView(i.box, size) : i.box;
      this.apply();
    } else if (i.type === "tap" && this.box) {
      const el = this.el();
      this.onTap(toPhoto(this.box, el, i.at), MIN_TARGET_PX[i.kind] / scale(this.box, el), i.kind);
    }
  }
}
