import { describe, expect, it } from "vitest";
import { cameraRank, chooseCamera, isFrontLabel, labelled, rearCameras, type CameraInfo } from "./choose";
import { focal35FromPx, focalPxFrom35, fovRatio, frame35, frameFocalPx } from "./focal";

const cam = (label: string, deviceId = label) => ({ deviceId, label });
// The cameras the phones listed (S3, 2026-11-17).
const IPHONE: CameraInfo[] = ["Back Dual Wide Camera", "Back Triple Camera", "Back Ultra Wide Camera", "Back Telephoto Camera",
  "Back Camera", "Back Dual Camera", "Front Camera"].map((l) => cam(l));
const SAMSUNG = [cam("camera 2, facing back"), cam("camera 0, facing back"), cam("camera 1, facing front")];
const FIREFOX = [cam("Facing front:1"), cam("Facing back:0")];

describe("choosing the camera", () => {
  it("prefers the physical main camera over virtual, ultra-wide and front ones", () => {
    expect(chooseCamera(IPHONE, null)?.label).toBe("Back Camera");
    expect(chooseCamera(SAMSUNG, null)?.label).toBe("camera 0, facing back");
    expect(chooseCamera(FIREFOX, null)?.label).toBe("Facing back:0");
    expect(chooseCamera([cam("camera 0, facing back")], null)?.label).toBe("camera 0, facing back");
    expect(cameraRank("Back Triple Camera")).toBeLessThan(cameraRank("Back Ultra Wide Camera"));
    expect(cameraRank("camera 0, facing back")).toBeLessThan(cameraRank("camera 2, facing back"));
  });

  it("keeps the remembered camera, unless it is gone or an ultra-wide", () => {
    expect(chooseCamera(IPHONE, cam("Back Triple Camera"))?.label).toBe("Back Triple Camera");
    expect(chooseCamera(IPHONE, { deviceId: "old id", label: "Back Telephoto Camera" })?.label).toBe("Back Telephoto Camera");
    expect(chooseCamera(IPHONE, cam("A camera unplugged"))?.label).toBe("Back Camera");
    const avoid = new Set(["camera 0, facing back"]);              // say a still showed it is an ultra-wide
    expect(chooseCamera(SAMSUNG, cam("camera 0, facing back"), avoid)?.label).toBe("camera 2, facing back");
    expect(chooseCamera([cam("camera 0, facing back")], null, avoid)?.label).toBe("camera 0, facing back");
  });

  it("knows front cameras and labels that are not there yet", () => {
    expect(isFrontLabel("Front TrueDepth Camera")).toBe(true);
    expect(isFrontLabel("camera 1, facing front")).toBe(true);
    expect(isFrontLabel("Facing front:1")).toBe(true);
    expect(isFrontLabel("Back Camera")).toBe(false);
    expect(rearCameras(SAMSUNG).map((c) => c.label)).toEqual(["camera 2, facing back", "camera 0, facing back"]);
    expect(labelled([cam("", "a"), cam("", "b")])).toBe(false);
    expect(chooseCamera([cam("", "a")], null)?.deviceId).toBe("a");
    expect(chooseCamera([cam("Front Camera")], null)).toBeNull();
  });
});

describe("focal lengths for frames", () => {
  it("convert as splinewire.camera does", () => {
    // 24 mm on a 4032×3024 still: 24 × 5040 / 43.267 px
    expect(focalPxFrom35(24, { width: 4032, height: 3024 })).toBeCloseTo(2795.689, 3);          // splinewire.camera.focal_px_from_35mm(24, (4032, 3024))
    expect(focal35FromPx(2795.689, { width: 4032, height: 3024 })).toBeCloseTo(24, 3);
  });

  it("scale a still's focal length to a frame by the long side", () => {
    const still = { width: 4032, height: 3024 };
    const f = focalPxFrom35(24, still);
    // The iPhone's full-sensor stream (S3): same field, portrait: the still's focal applies unchanged.
    expect(frameFocalPx(24, still, { width: 3024, height: 4032 })).toBeCloseTo(f, 6);
    // A 1080p crop at full width: a factor 1920/4032, not the diagonal's.
    expect(frameFocalPx(24, still, { width: 1920, height: 1080 })).toBeCloseTo((f * 1920) / 4032, 6);
    expect(frameFocalPx(24, null, { width: 1920, height: 1080 })).toBeCloseTo((f * 1920) / 4032, 6);
    // Saved as a photo, that frame is measured with a larger 35 mm figure (its diagonal is shorter).
    expect(frame35(24, still, { width: 1920, height: 1080 })).toBeCloseTo(24 * (1920 * 1.25) / Math.hypot(1920, 1080), 6);
    expect(frame35(24, still, { width: 3000, height: 4000 })).toBeCloseTo(24, 6);
  });

  it("compare the live view's scale with the still's", () => {
    const pins: [number, number][] = [[100, 100], [200, 100], [300, 120], [400, 160]];
    const frame = pins.map(([x, y]) => [x / 2, y / 2] as [number, number]);
    expect(fovRatio(frame, { width: 2016, height: 1512 }, pins, { width: 4032, height: 3024 })).toBeCloseTo(1, 6);
    // A frame cropped 10% tighter shows the chain 10% larger.
    const tight = pins.map(([x, y]) => [x * 0.55, y * 0.55] as [number, number]);
    expect(fovRatio(tight, { width: 2016, height: 1512 }, pins, { width: 4032, height: 3024 })).toBeCloseTo(1.1, 6);
    expect(fovRatio(frame.slice(0, 2), { width: 1, height: 1 }, pins, { width: 1, height: 1 })).toBeNull();
  });
});
