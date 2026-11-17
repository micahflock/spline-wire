// The pairing link as a QR code: one SVG path, dark modules on a white quiet
// zone whatever the page's theme (phone cameras read dark-on-light best).

import qrcode from "qrcode-generator";
import { esc } from "./wording";

/** Quiet zone in modules; the standard asks for 4. */
const MARGIN = 4;

export function qrSvg(text: string, label: string): string {
  // Level M survives a smudged screen or glare; the link (about 60 characters
  // with the secret) still fits a small code.
  const qr = qrcode(0, "M");
  qr.addData(text, "Byte");
  qr.make();
  const n = qr.getModuleCount();
  let d = "";
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      // Runs of dark modules as one rectangle each keep the path short.
      if (!qr.isDark(r, c)) continue;
      let run = 1;
      while (c + run < n && qr.isDark(r, c + run)) run++;
      d += `M${c + MARGIN} ${r + MARGIN}h${run}v1h-${run}z`;
      c += run - 1;
    }
  }
  const size = n + 2 * MARGIN;
  return `<svg class="qr" viewBox="0 0 ${size} ${size}" role="img" aria-label="${esc(label)}" shape-rendering="crispEdges">` +
    `<rect width="${size}" height="${size}" fill="#fff"/><path d="${d}" fill="#000"/></svg>`;
}
