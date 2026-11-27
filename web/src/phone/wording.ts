// Plain words for what the measurement reports, where the photo can be fixed
// by hand: on the phone by touch, and on the desktop page with the mouse (a
// photo dropped there, D9), hence `input`. The wording follows the retired
// Windows app's readWarning and readError. Warnings are shown in orange, notes in grey; a person's error
// (a Python ValueError) is shown as is, under a plain heading.

import type { Failure, Progress } from "../core/protocol";
import type { FocalSource, MeasurementResult } from "../core/types";

export interface Reading {
  title: string;
  text: string;
}

export const plural = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;

/** How the photo is edited: a tap on the phone, a click on the desktop page. */
export type Input = "touch" | "mouse";
const tap = (input: Input) => (input === "mouse" ? "click" : "tap");
const Tap = (input: Input) => (input === "mouse" ? "Click" : "Tap");

export function readNote(n: string, input: Input = "touch"): Reading {
  const m = n.match(/^ignored (\d+) look-alike/);
  if (m) {
    return { title: `${plural(+m[1]!, "stray dot")} ignored`,
      text: `Dots that aren't on the chain, such as washers or a patterned table. ${Tap(input)} one to use it as a pin.` };
  }
  return { title: "Note", text: n };
}

export function readWarning(w: string, input: Input = "touch"): Reading {
  let m: RegExpMatchArray | null;
  if ((m = w.match(/^(\d+) pin\(s\) you added were left out/))) {
    const k = +m[1]!;
    return { title: `${plural(k, "pin")} you added ${k === 1 ? "doesn't" : "don't"} fit the chain`,
      text: `Each pin must be about one link from its neighbours, in line with the chain. ${Tap(input)} it to delete it.` };
  }
  if ((m = w.match(/^(\d+) pin\(s\) you added have no fiducial/))) {
    return { title: `${plural(+m[1]!, "pin")} placed where you ${input === "mouse" ? "clicked" : "tapped"}`,
      text: `There was no dot under the ${tap(input)}, so the pin was fitted to the links around it. Zoom in and ${tap(input)} the middle of the dot if you can see it.` };
  }
  if ((m = w.match(/^(\d+) pin\(s\) not detected/))) {
    return { title: `${plural(+m[1]!, "pin")} not found`, text: `The curve bridges the gap. ${Tap(input)} the missing dot on the photo to add it.` };
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
export function errorAdvice(message: string, fixable: boolean, input: Input = "touch"): string {
  if (/is not a photo|could not read the photo/.test(message)) {
    return input === "mouse" ? "Use the chain's photo as the phone saved it, a JPEG or HEIC file."
      : "Take it again with the camera button, or pick a JPEG from the library.";
  }
  const byHand = fixable ? ` Or ${tap(input)} each pin on the photo.` : "";
  if (/fiducials|detections to find a chain/.test(message)) {
    return "Get the whole chain in the frame with the dots facing up, in focus and without glare, and take the photo again." + byHand
      + ` If the chain's dots or rims are darker than its links, choose that under ${input === "mouse" ? "Colours" : "Colors in Settings"}.`;
  }
  if (/pins in a row|links to rectify/.test(message)) {
    return "Glare or a shadow may be hiding some dots: move the lamp or turn the chain and take it again." + byHand;
  }
  if (/could not fit a plane/.test(message)) return `Check the pins on the photo, and the focal length${input === "mouse" ? "" : " in Settings"}.`;
  return fixable ? "Check the pins on the photo." : "";
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
  override: "the focal override (Settings, debug)",
  default: "the focal length in Settings",
  estimated: "estimated from the chain",
};

export const SIDE_TEXT = {
  inside: "inside of the bend (chain around the object)",
  outside: "outside of the bend (chain in a hollow)",
};

/** Where a photo's focal length came from when the page supplied it (main.ts,
 * PhotoRecord.focal): the core only sees an override. */
export function photoFocalText(f: { why: "lens" | "frame"; lens: string | null }): string {
  if (f.why === "lens") return `remembered for ${f.lens ?? "this lens"} (the photo had no 35 mm focal length)`;
  return f.lens ? `this camera's last photo (a video frame has no camera data)` : "the focal length in Settings (a video frame has no camera data)";
}

/** The numbers under the photo, as [label, value, detail, warn]. `focalDetail`
 * replaces the focal length's source when the page supplied it. */
export function numbers(r: MeasurementResult, expectedPins: number, focalDetail?: string): [string, string, string, boolean][] {
  const found = r.pins;
  return [
    ["Pins", `${found} of ${expectedPins}`, r.missing_pins ? `${r.missing_pins} missing, bridged` : "found on the chain",
      found + r.missing_pins !== expectedPins || r.missing_pins > 0],
    ["Tilt", `${r.tilt_deg.toFixed(1)}°`, "camera to table", false],
    ["Focal length", `${r.focal_35mm.toFixed(1)} mm`, focalDetail ?? FOCAL_SOURCE[r.focal_source], r.focal_source === "estimated"],
    ["Links", `${r.link_residual_rms_mm.toFixed(3)} mm`, `rms off the pitch, worst ${r.link_residual_max_mm.toFixed(3)} mm`,
      r.warnings.some((w) => w.startsWith("link lengths deviate"))],
    ["Curve", `${r.curve_points} points`, `object on the ${r.object_side}`, false],
  ];
}

/** The phone's Measurement table (for UI review/phone/BRIEF.md), as [label, value, detail, warn]: the same figures as
 * `numbers` in the brief's rows and words; no object side (Swap side shows it on the photo)
 * and no curve points (the chain style and its pins are set above the table). */
export function measurementRows(r: MeasurementResult, expectedPins: number, focalDetail?: string): [string, string, string, boolean][] {
  const found = r.pins;
  return [
    ["Pins found", `${found} / ${expectedPins}`, r.missing_pins ? `${r.missing_pins} missing, bridged` : "",
      found + r.missing_pins !== expectedPins || r.missing_pins > 0],
    ["Camera tilt", `${r.tilt_deg.toFixed(1)}°`, "camera to table", false],
    ["Link length error", `${r.link_residual_rms_mm.toFixed(3)} mm`, `rms · max ${r.link_residual_max_mm.toFixed(3)}`,
      r.warnings.some((w) => w.startsWith("link lengths deviate"))],
    ["Focal length", `${r.focal_35mm.toFixed(1)} mm`, focalDetail ?? FOCAL_SOURCE[r.focal_source], r.focal_source === "estimated"],
  ];
}

/** What the engine is doing while it loads (8-11 s on a desktop, S1). */
export const ENGINE_STAGE: Record<Progress["stage"], string> = {
  ready: "Ready",
};

/** What the phone keeps (the list of photos): photos with their size, curves whose photo is gone, and the limit. */
export function spaceText(photos: number, bytes: number, curvesOnly: number): string {
  const mb = bytes / 1e6;
  const size = mb >= 10 ? `${Math.round(mb)} MB` : `${mb.toFixed(1)} MB`;
  const kept = [photos ? `${photos === 1 ? "1 photo" : `${photos} photos`} (${size})` : "",
    curvesOnly ? `${curvesOnly === 1 ? "1 curve" : `${curvesOnly} curves`} without ${curvesOnly === 1 ? "its photo" : "their photos"}` : ""]
    .filter(Boolean).join(" and ");
  return kept || "Nothing yet";
}

/** Settings, Focal length: where the value for photos without camera data comes from. */
export function focalStatus(saved: { focal35: number; lens: { name: string } | null } | null, learned: { name: string; focal35: number } | null): string {
  const use = "Used for photos without camera data, which iPhones often strip.";
  if (saved?.lens) return `${saved.focal35} mm, learned from ${saved.lens.name}. ${use}`;
  if (saved) return `${use} Clear it to use ${learned ? `the ${learned.focal35} mm learned from ${learned.name}` : "one learned from a photo"}.`;
  return "Not known yet: photos without camera data, which iPhones often strip, are estimated from the chain, which is less "
    + "accurate. Teach it from a photo, or type it: 24 mm on iPhone 14 Pro, 15 and later; 26 mm on earlier models.";
}
