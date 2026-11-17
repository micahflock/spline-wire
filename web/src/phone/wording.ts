// Plain words for what the measurement reports, on the phone, where the
// photo can be fixed by touch. The wording follows the current app's
// readWarning and readError (splinewire/webapp/static/desktop.html and
// phone.html). Warnings are shown in orange, notes in grey; a person's error
// (a Python ValueError) is shown as is, under a plain heading.

import type { Failure } from "../core/protocol";
import type { FocalSource, MeasurementResult } from "../core/types";

export interface Reading {
  title: string;
  text: string;
}

export const plural = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;

export function readNote(n: string): Reading {
  const m = n.match(/^ignored (\d+) look-alike/);
  if (m) {
    return { title: `${plural(+m[1]!, "stray dot")} ignored`,
      text: "Dots that aren't on the chain, such as washers or a patterned table. Tap one to use it as a pin." };
  }
  return { title: "Note", text: n };
}

export function readWarning(w: string): Reading {
  let m: RegExpMatchArray | null;
  if ((m = w.match(/^(\d+) pin\(s\) you added were left out/))) {
    const k = +m[1]!;
    return { title: `${plural(k, "pin")} you added ${k === 1 ? "doesn't" : "don't"} fit the chain`,
      text: "Each pin must be about one link from its neighbours, in line with the chain. Tap it to delete it." };
  }
  if ((m = w.match(/^(\d+) pin\(s\) you added have no fiducial/))) {
    return { title: `${plural(+m[1]!, "pin")} placed where you tapped`,
      text: "There was no dot under the tap, so the pin was fitted to the links around it. Zoom in and tap the middle of the dot if you can see it." };
  }
  if ((m = w.match(/^(\d+) pin\(s\) not detected/))) {
    return { title: `${plural(+m[1]!, "pin")} not found`, text: "The curve bridges the gap. Tap the missing dot on the photo to add it." };
  }
  if ((m = w.match(/^the chain seems to go on past its end \((\d+) more/))) {
    return { title: "The chain seems to go on past its end", text: `${plural(+m[1]!, "more dot")} about one link on. Is the pin count right?` };
  }
  if ((m = w.match(/^found (\d+) pins on the chain(?:,| but)[^\d]*(\d+)/))) {
    return { title: `Found ${m[1]} pins, expected ${m[2]}`,
      text: "Is the whole chain in the photo, with no glare on the links? If the chain has a different number of pins, change Pins below." };
  }
  if (/^no focal length/.test(w)) {
    return { title: "Focal length estimated",
      text: "The photo had no camera data, so the focal length was worked out from the chain, which is unreliable for a nearly straight chain. Save your phone's focal length in Settings." };
  }
  if ((m = w.match(/^link lengths deviate from the pitch by up to ([\d.]+) mm/))) {
    return { title: `A link is ${m[1]} mm off its length`, text: "Check the pins on the photo. A pin on the wrong dot, or a badly printed dot, does this." };
  }
  return { title: "Check this measurement", text: w };
}

/** A heading for a person's error; the message itself is shown as is beneath it. */
export function errorTitle(message: string): string {
  let m: RegExpMatchArray | null;
  if ((m = message.match(/found (\d+) fiducials/)) || (m = message.match(/need at least 3 detections to find a chain, got (\d+)/))) {
    return +m[1]! < 3 ? "No chain found in this photo." : `Only ${m[1]} dots found.`;
  }
  if ((m = message.match(/found (\d+) pins in a row, joined by/))) return `Only ${m[1]} pins in a row.`;
  if ((m = message.match(/need at least \d+ links to rectify, got (\d+)/))) return `Only ${+m[1]! + 1} pins in a row.`;
  if (/is not a photo|could not read the photo/.test(message)) return "Couldn't read this photo.";
  if (/could not fit a plane/.test(message)) return "Couldn't work out the camera angle.";
  return "Couldn't measure this photo.";
}

/** What to do about it, for the commonest person's errors. */
export function errorAdvice(message: string, fixable: boolean): string {
  if (/is not a photo|could not read the photo/.test(message)) {
    return "Take it again with the camera button, or pick a JPEG from the library.";
  }
  const byHand = fixable ? " Or tap each pin on the photo below." : "";
  if (/fiducials|detections to find a chain/.test(message)) {
    return "Get the whole chain in the frame with the dots facing up, in focus and without glare, and take the photo again." + byHand;
  }
  if (/pins in a row|links to rectify/.test(message)) {
    return "Glare or a shadow may be hiding some dots: move the lamp or turn the chain and take it again." + byHand;
  }
  if (/could not fit a plane/.test(message)) return "Check the pins on the photo, and the focal length in Settings.";
  return fixable ? "Check the pins on the photo below." : "";
}

/** A failure with no document: a person's error as is, a bug with its type. */
export function failureText(f: Failure): { title: string; text: string; bug: boolean } {
  if (f.kind === "person") return { title: errorTitle(f.message), text: f.message, bug: false };
  return {
    title: "Something went wrong in Spline Wire.",
    text: `${f.type ?? "Error"}: ${f.message}`,
    bug: true,
  };
}

export const FOCAL_SOURCE: Record<FocalSource, string> = {
  exif: "from the photo",
  override: "your override (Settings)",
  default: "your saved focal length (Settings)",
  estimated: "estimated from the chain",
};

export const SIDE_TEXT = {
  inside: "inside of the bend (chain around the object)",
  outside: "outside of the bend (chain in a hollow)",
};

/** The numbers under the photo, as [label, value, detail, warn]. */
export function numbers(r: MeasurementResult, expectedPins: number): [string, string, string, boolean][] {
  const found = r.pins;
  return [
    ["Pins", `${found} of ${expectedPins}`, r.missing_pins ? `${r.missing_pins} missing, bridged` : "found on the chain",
      found + r.missing_pins !== expectedPins || r.missing_pins > 0],
    ["Tilt", `${r.tilt_deg.toFixed(1)}°`, "camera to table", false],
    ["Focal length", `${r.focal_35mm.toFixed(1)} mm`, FOCAL_SOURCE[r.focal_source], r.focal_source === "estimated"],
    ["Links", `${r.link_residual_rms_mm.toFixed(3)} mm`, `rms off the pitch, worst ${r.link_residual_max_mm.toFixed(3)} mm`,
      r.warnings.some((w) => w.startsWith("link lengths deviate"))],
    ["Curve", `${r.curve_points} points`, `object on the ${r.object_side}`, false],
  ];
}
