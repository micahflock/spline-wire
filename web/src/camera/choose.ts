// Which camera the phone page opens (web-build.md O10), from the cameras'
// labels alone: browsers give no focal length for a camera before a photo.
//
// What the phones showed (S3, 2026-11-17):
//   iPhone, Safari     facingMode "environment" opens the virtual "Back Triple
//                      Camera", which switches lenses on its own (to the
//                      ultra-wide when close), so the focal length could change
//                      between the frames and the still. The physical main lens
//                      is "Back Camera".
//   Galaxy S23 Ultra   facingMode opened "camera 2, facing back": the
//                      ultra-wide (13 mm). The main camera is "camera 0, facing back".
//   Pixel 7 Pro        one back camera, "camera 0, facing back".
//   Pixel 10, Firefox  "Facing back:0", "Facing front:1".
// So: the plain main camera first, numbered back cameras by number, virtual
// multi-lens cameras after them, ultra-wides and telephotos last. A camera a
// still showed to be an ultra-wide (lenses.ts isUltraWide) is avoided.

export interface CameraInfo {
  deviceId: string;
  label: string;
}

export function isFrontLabel(label: string): boolean {
  return /\bfront\b|facing front|\buser\b|selfie|truedepth/i.test(label);
}

/** The cameras that are not known to face the person. Unlabelled ones count
 * (labels are empty before permission is given). */
export function rearCameras(devices: readonly CameraInfo[]): CameraInfo[] {
  return devices.filter((d) => !isFrontLabel(d.label));
}

/** Lower is better. */
export function cameraRank(label: string): number {
  const l = label.trim();
  if (/^back camera$/i.test(l)) return 0;                         // iPhone's physical main lens
  if (/ultra ?wide/i.test(l)) return 50;
  if (/tele/i.test(l)) return 40;
  if (/triple|dual|virtual/i.test(l)) return 30;                    // lenses switch on their own
  const n = l.match(/(?:camera2?\s*(\d+)\s*,\s*facing back)|(?:facing back\s*:\s*(\d+))/i);
  if (n) return 1 + Math.min(20, Number(n[1] ?? n[2]));            // camera 0 is the main one on Android
  if (/back|rear|environment/i.test(l)) return 25;
  return 26;                                                        // unlabelled, or a desktop webcam
}

/** The camera to open: the remembered one if it is still there and not to be
 * avoided, else the best ranked. `avoid` holds labels of known ultra-wides;
 * if every camera is to be avoided, the best of them is still better than none. */
export function chooseCamera(devices: readonly CameraInfo[], remembered: CameraInfo | null, avoid: ReadonlySet<string> = new Set()): CameraInfo | null {
  const rear = rearCameras(devices);
  if (rear.length === 0) return null;
  if (remembered && !avoid.has(remembered.label)) {
    const same = rear.find((d) => d.deviceId && d.deviceId === remembered.deviceId)
      ?? rear.find((d) => d.label && d.label === remembered.label);
    if (same) return same;
  }
  const ranked = [...rear].sort((a, b) => (Number(avoid.has(a.label)) - Number(avoid.has(b.label))) || cameraRank(a.label) - cameraRank(b.label));
  return ranked[0]!;
}

/** Whether the device list has labels yet (they come with permission). */
export function labelled(devices: readonly CameraInfo[]): boolean {
  return devices.some((d) => d.label.trim() !== "");
}
