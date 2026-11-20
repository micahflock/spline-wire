// The downloads a curve offers and what they are called: the names
// splinewire/session.py OUTPUT_NAMES gives them, after the photo's stem, so a
// file saved here has the name the CLI and the phone give it.

import type { FileKind } from "../relay/files";

export type DownloadKind = FileKind | "tsv";

export const DOWNLOADS: { kind: DownloadKind; label: string; name: string; type: string }[] = [
  { kind: "dxf", label: "DXF", name: "{stem}-curve.dxf", type: "application/dxf" },
  { kind: "svg", label: "SVG 1:1", name: "{stem}-curve.svg", type: "image/svg+xml" },
  { kind: "csv", label: "CSV", name: "{stem}-curve.csv", type: "text/csv" },
  { kind: "json", label: "JSON", name: "{stem}.json", type: "application/json" },
  { kind: "fusion_csv", label: "Fusion CSV", name: "{stem}-fusion-cm.csv", type: "text/csv" },
  // Python names no file for the clipboard table (bridge.outputs); this one is ours.
  { kind: "tsv", label: "TSV", name: "{stem}-points.tsv", type: "text/tab-separated-values" },
];

/** The photo's name without its folder or extension, as bridge._stem:
 * PurePosixPath(PureWindowsPath(name).name).stem, or "photo". */
export function stemOf(photo: string): string {
  let base = photo.split(/[\\/]/).pop() ?? "";
  if (base === ".") base = "";              // pathlib's name for "." is empty
  const dot = base.lastIndexOf(".");
  // pathlib's suffix rule: a dot that is neither first nor last ("foo.", ".hidden" have none).
  const stem = dot > 0 && dot < base.length - 1 ? base.slice(0, dot) : base;
  return stem || "photo";
}

/** `photo`'s download, or with `title` (a person's name for the curve) the name's. */
export function downloadName(kind: DownloadKind, photo: string, title = ""): string {
  const d = DOWNLOADS.find((x) => x.kind === kind)!;
  return d.name.replace("{stem}", title ? fileStem(title) : stemOf(photo));
}

/** A person's name for a curve as a file's stem: what file systems refuse
 * replaced, whole otherwise (a name's dot is not an extension: "Pipe 1.5in"). */
export function fileStem(title: string): string {
  return title.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "-").replace(/[. ]+$/, "").trim().slice(0, 100) || "curve";
}

/** A file named after `photo` (the phone's names, from the core) renamed after `title`. */
export function renamedFile(file: string, photo: string, title: string): string {
  const stem = stemOf(photo);
  return title && file.startsWith(stem) ? fileStem(title) + file.slice(stem.length) : file;
}
