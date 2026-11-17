// Plain words for what the measurement reports, as the current app's
// readWarning says them (splinewire/webapp/static/desktop.html). Here the
// photo is on the phone, so the advice points there.
//
// Python's warnings are shown in orange and its notes in grey: notes are
// information (stray dots on a busy table are normal, pipeline.py says why).

export interface Reading {
  level: "warn" | "info";
  title: string;
  text: string;
}

export const plural = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;

export function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

export function readNote(n: string): Reading {
  const m = n.match(/^ignored (\d+) look-alike/);
  if (m) {
    return { level: "info", title: `${plural(+m[1]!, "stray dot")} ignored`,
      text: "Dots that aren't on the chain, such as washers or a patterned table." };
  }
  return { level: "info", title: "Note", text: n };
}

export function readWarning(w: string): Reading {
  const warn = (title: string, text: string): Reading => ({ level: "warn", title, text });
  let m: RegExpMatchArray | null;
  if ((m = w.match(/^(\d+) pin\(s\) you added were left out/))) {
    const k = +m[1]!;
    return warn(`${plural(k, "pin")} added by hand ${k === 1 ? "doesn't" : "don't"} fit the chain`,
      "Each pin must be about one link from its neighbours, in line with the chain. Remove it on the phone.");
  }
  if ((m = w.match(/^(\d+) pin\(s\) you added have no fiducial/))) {
    return warn(`${plural(+m[1]!, "pin")} placed by hand`,
      "There was no dot under the tap, so the pin was fitted to the links around it.");
  }
  if ((m = w.match(/^(\d+) pin\(s\) not detected/))) {
    return warn(`${plural(+m[1]!, "pin")} not found`, "The curve bridges the gap. Tap the missing dot on the phone to add it.");
  }
  if ((m = w.match(/^the chain seems to go on past its end \((\d+) more/))) {
    return warn("The chain seems to go on past its end", `${plural(+m[1]!, "more dot")} about one link on. Is the pin count right?`);
  }
  if ((m = w.match(/^found (\d+) pins on the chain(?:,| but)[^\d]*(\d+)/))) {
    return warn(`Found ${m[1]} pins, expected ${m[2]}`,
      "Is the whole chain in the photo, with no glare on the links? If the chain has a different number of pins, change it on the phone.");
  }
  if (/^no focal length/.test(w)) {
    return warn("Focal length estimated",
      "The photo had no camera data, so the focal length was worked out from the chain. Set the phone's focal length in its settings for a better measurement.");
  }
  if ((m = w.match(/^link lengths deviate from the pitch by up to ([\d.]+) mm/))) {
    return warn(`A link is ${m[1]} mm off its length`, "Check the pins on the phone. A pin on the wrong dot, or a badly printed dot, does this.");
  }
  return warn("Check this measurement", w);
}

export const FOCAL_SOURCE: Record<string, string> = {
  exif: "from the photo",
  override: "set on the phone",
  default: "the phone's saved setting",
  estimated: "estimated from the chain",
};
