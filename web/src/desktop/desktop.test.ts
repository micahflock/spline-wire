import { describe, expect, it } from "vitest";
import fixture from "../test-data/measurement.json";
import type { Point } from "../relay/messages";
import { DOWNLOADS, downloadName, fileStem, renamedFile, stemOf } from "./names";
import { plotGeometry, plotSvg } from "./plot";
import { qrSvg } from "./qr";
import { HOLD_MS, Selection } from "./selection";
import { statusLine } from "./status";
import { UnseenTitle } from "./title";
import { exactTime, relativeTime } from "./when";
import { esc, readNote, readWarning } from "./wording";

describe("which curve is shown", () => {
  it("shows each new curve while the person has not picked one", () => {
    const s = new Selection();
    expect(s.arrived("a", 0)).toBe(true);
    expect(s.arrived("b", 1000)).toBe(true);
    expect(s.id).toBe("b");
  });

  it("keeps the person's pick for a minute, then follows new curves again", () => {
    const s = new Selection();
    s.arrived("a", 0);
    s.arrived("b", 0);
    s.pick("a", 10_000);
    expect(s.arrived("c", 10_000 + HOLD_MS - 1)).toBe(false);
    expect(s.id).toBe("a");
    expect(s.arrived("d", 10_000 + HOLD_MS)).toBe(true);
    expect(s.id).toBe("d");
  });

  it("shows the newest after a load and the next older one after a delete", () => {
    const s = new Selection();
    s.keepIn(["c", "b", "a"]);
    expect(s.id).toBe("c");
    s.pick("b", 0);
    s.keepIn(["c", "a"], 1);              // b, at index 1, was deleted
    expect(s.id).toBe("a");
    expect(s.arrived("d", 1)).toBe(true);  // the replacement was not the person's pick
    s.pick("a", 2);
    s.keepIn(["d", "c"], 2);              // the last one went: the one before it
    expect(s.id).toBe("c");
    s.keepIn([]);
    expect(s.id).toBeNull();
    expect(s.arrived("e", 3)).toBe(true);
  });

  it("an empty page shows a curve that arrives even right after a pick", () => {
    const s = new Selection();
    s.pick("a", 0);
    s.keepIn([], 0);
    expect(s.arrived("b", 1)).toBe(true);
  });
});

describe("the tab title", () => {
  it("counts curves that arrive while the tab is hidden, until it is seen", () => {
    const t = new UnseenTitle("Spline Wire");
    t.arrived(false);
    expect(t.title).toBe("Spline Wire");
    t.arrived(true);
    t.arrived(true);
    expect(t.title).toBe("(2) Spline Wire");
    t.seen();
    expect(t.title).toBe("Spline Wire");
  });
});

describe("relative time", () => {
  const now = Date.parse("2026-11-18T12:00:00Z");
  it("says how long ago, in words a person reads at a glance", () => {
    expect(relativeTime(now - 5_000, now)).toBe("just now");
    expect(relativeTime(now + 30_000, now)).toBe("just now");     // the phone's clock is ahead
    expect(relativeTime(now - 50_000, now)).toBe("1 min ago");
    expect(relativeTime(now - 2 * 60_000 - 10_000, now)).toBe("2 min ago");
    expect(relativeTime(now - 59 * 60_000, now)).toBe("59 min ago");
    expect(relativeTime(now - 3 * 3600_000 - 1, now)).toBe("3 h ago");
    expect(relativeTime(now - 30 * 3600_000, now)).toBe("yesterday");
    expect(relativeTime(now - 3 * 86400_000, now)).toBe("3 days ago");
    expect(relativeTime(Date.parse("2026-10-02T12:00:00Z"), now, "en-GB")).toBe("2 Oct");
    expect(relativeTime(Date.parse("2025-10-02T12:00:00Z"), now, "en-GB")).toBe("2 Oct 2025");
  });

  it("gives the exact time for the tooltip", () => {
    expect(exactTime(now, "en-GB")).toMatch(/18 Nov 2026/);
  });
});

describe("download names", () => {
  it("are session.OUTPUT_NAMES after the photo's stem, as bridge._stem makes it", () => {
    expect(downloadName("dxf", "IMG_0001.jpg")).toBe("IMG_0001-curve.dxf");
    expect(downloadName("json", "IMG_0001.jpg")).toBe("IMG_0001.json");
    expect(downloadName("csv", "IMG_0001.jpg")).toBe("IMG_0001-curve.csv");
    expect(downloadName("svg", "IMG_0001.jpg")).toBe("IMG_0001-curve.svg");
    expect(downloadName("fusion_csv", "IMG_0001.jpg")).toBe("IMG_0001-fusion-cm.csv");
    expect(downloadName("tsv", "IMG_0001.jpg")).toBe("IMG_0001-points.tsv");
    expect(DOWNLOADS.map((d) => d.kind).sort()).toEqual(["csv", "dxf", "fusion_csv", "json", "svg", "tsv"]);
  });

  it("follow a person's name for the curve, whole, with what file systems refuse replaced", () => {
    expect(downloadName("dxf", "IMG_0001.jpg", "Pipe 1.5in")).toBe("Pipe 1.5in-curve.dxf");     // a dot in a name is no extension
    expect(downloadName("json", "IMG_0001.jpg", "a/b: c?")).toBe("a-b- c-.json");
    expect(fileStem("Knob...")).toBe("Knob");
    expect(fileStem("??")).toBe("-");
    expect(fileStem(" . ")).toBe("curve");
    // The phone's files, named by the core after the photo, renamed after the curve.
    expect(renamedFile("IMG_0001-curve.dxf", "IMG_0001.jpg", "Door knob")).toBe("Door knob-curve.dxf");
    expect(renamedFile("IMG_0001.json", "IMG_0001.jpg", "")).toBe("IMG_0001.json");
  });

  // Each pair is Python 3.12's bridge._stem(name).
  it.each([
    ["IMG_0001.jpg", "IMG_0001"], ["a.tar.gz", "a.tar"], [".hidden", ".hidden"], ["foo.", "foo."],
    ["C:\\x\\y.JPG", "y"], ["dir/z.heic", "z"], ["noext", "noext"], ["", "photo"], ["..", ".."],
    ["a..b", "a."], ["x .jpg", "x "], [".", "photo"], ["...", "..."],
  ])("stem of %j is %j", (name, stem) => {
    expect(stemOf(name)).toBe(stem);
  });
});

describe("the curve drawing", () => {
  const curve = fixture.document.curve as unknown as { points: Point[]; pins: Point[]; spline: Point[] };

  it("covers everything with a margin, on whole major lines, at 1:1", () => {
    const g = plotGeometry(curve.points, curve.pins, curve.spline);
    for (const [x, y] of [...curve.points, ...curve.pins, ...curve.spline]) {
      expect(x).toBeGreaterThanOrEqual(g.x0 + 8 - 1e-9);
      expect(x).toBeLessThanOrEqual(g.x1 - 8 + 1e-9);
      expect(y).toBeGreaterThanOrEqual(g.y0 + 8 - 1e-9);
      expect(y).toBeLessThanOrEqual(g.y1 - 8 + 1e-9);
    }
    expect(g.major).toBe(10);
    expect(g.minor).toBe(2);
    for (const v of [g.x0, g.x1, g.y0, g.y1]) expect(Math.abs(v % 10)).toBe(0);
    expect(g.xs[0]!.at).toBe(g.x0);
    expect(g.xs.at(-1)!.at).toBe(g.x1);
    expect(g.xs.length).toBe((g.x1 - g.x0) / 2 + 1);
    expect(g.xs.filter((l) => l.major).length).toBe((g.x1 - g.x0) / 10 + 1);
    expect(g.bar.x2 - g.bar.x1).toBeCloseTo(10, 12);
    expect(g.bar.label).toBe("10 mm");
    // One SVG unit per mm on both axes: the viewBox is the extent itself.
    const svg = plotSvg(curve.points, curve.pins, curve.spline, "x");
    expect(svg).toContain(`viewBox="${g.x0} ${-g.y1} ${g.x1 - g.x0} ${g.y1 - g.y0}"`);
    expect(svg.match(/class="pt"/g)).toHaveLength(curve.points.length);
    expect(svg.match(/class="pin"/g)).toHaveLength(curve.pins.length);
    // With the chain it was measured with, the chain is drawn as that chain (drawing/chain-art.ts), y flipped as the plot's.
    const drawn = plotSvg(curve.points, curve.pins, curve.spline, "x", undefined,
      { pitch_mm: 10, half_width_mm: 4, fiducial: "dot", fiducial_mm: 5, ring_inner_mm: 0, contact: "edge" });
    expect(drawn).not.toContain('class="pin"');
    expect(drawn.match(/<g class="chain-art">.*?<\/g>/s)![0].match(/<circle/g)).toHaveLength(curve.pins.length);
    expect(drawn).toContain(`cy="${+(-curve.pins[0]![1]).toFixed(2)}"`);
  });

  it("coarsens the grid for a big curve", () => {
    const pts: Point[] = [[0, 0], [100, 40], [200, 0]];
    const g = plotGeometry(pts, pts, pts);
    expect([g.major, g.minor]).toEqual([50, 10]);
    expect(g.bar.label).toBe("50 mm");
    expect(g.x0).toBe(-50);
    expect(g.x1).toBe(250);
  });

  it("draws sharp corners: the pieces, a marker at each, the points they replaced faint, and a target per point", () => {
    const targets: Point[] = [[0, 0], [10, 0], [12, 2], [14, 10], [14, 20]];
    const pts: Point[] = [[0, 0], [10, 0], [14, 0], [14, 10], [14, 20]];
    const svg = plotSvg(pts, targets, [], "x", {
      pieces: [[[0, 0], [10, 0], [14, 0]], [[14, 0], [14, 10], [14, 20]]],
      corners: [{ at: [14, 0], source: 2 }], targets, replaced: [2],
    });
    expect(svg.match(/class="trace"/g)).toHaveLength(2);
    expect(svg.match(/class="corner"/g)).toHaveLength(1);
    expect(svg.match(/class="pt"/g)).toHaveLength(4);            // the corner is drawn as a corner
    expect(svg.match(/class="pt-was"/g)).toHaveLength(1);
    // Every point but the ends is a target, and so is the corner (its marked point's).
    expect([...svg.matchAll(/data-act="corner" data-i="(\d+)"/g)].map((m) => m[1])).toEqual(["1", "2", "3", "2"]);
  });

  it("escapes the label", () => {
    expect(plotSvg([[0, 0], [1, 1]], [], [[0, 0], [1, 1]], '<b>"x"</b>')).toContain('aria-label="&lt;b&gt;&quot;x&quot;&lt;/b&gt;"');
  });
});

describe("the pairing QR code", () => {
  it("is a square code with a quiet zone that changes with the link", () => {
    const a = qrSvg("https://example.org/spline-wire/phone/#pair=AAAAAAAAAAAAAAAAAAAAAA", "code");
    const b = qrSvg("https://example.org/spline-wire/phone/#pair=BBBBBBBBBBBBBBBBBBBBBB", "code");
    const size = Number(a.match(/viewBox="0 0 (\d+) \1"/)![1]);
    // A 66-character link needs version 5 (37 modules) at level M; plus 4 modules each side.
    expect(size).toBe(37 + 8);
    expect(a).not.toBe(b);
    expect(a).toBe(qrSvg("https://example.org/spline-wire/phone/#pair=AAAAAAAAAAAAAAAAAAAAAA", "code"));
  });
});

describe("plain words", () => {
  it("reads Python's warnings and notes", () => {
    expect(readWarning("3 pin(s) not detected; the curve bridges those gaps")).toMatchObject({ level: "warn", title: "3 pins not found" });
    expect(readWarning("found 11 pins on the chain, which is set to have 13; is the whole chain…")).toMatchObject({ title: "Found 11 pins, expected 13" });
    expect(readWarning("link lengths deviate from the pitch by up to 0.62 mm; check")).toMatchObject({ title: "A link is 0.62 mm off its length" });
    expect(readWarning("something new")).toMatchObject({ level: "warn", text: "something new" });
    expect(readNote("ignored 1 look-alike detection(s) not on the chain")).toMatchObject({ level: "info", title: "1 stray dot ignored" });
    expect(esc(`<a href="x">'&'</a>`)).toBe("&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
  });

  it("points a curve measured here back to its photo, not to a phone", () => {
    const warnings = ["2 pin(s) you added were left out", "3 pin(s) not detected; the curve bridges those gaps",
      "found 11 pins on the chain, which is set to have 13", "no focal length (none in the photo's EXIF, and none set)",
      "link lengths deviate from the pitch by up to 0.62 mm; check"];
    for (const w of warnings) {
      expect(readWarning(w).text).toMatch(/phone/);
      expect(readWarning(w, true).text).not.toMatch(/phone/);
      expect(readWarning(w, true).title).toBe(readWarning(w).title);
    }
  });
});

describe("the status line", () => {
  it("says what the link sees", () => {
    expect(statusLine({ state: "connecting", peers: 0 }, null).text).toBe("Connecting to the relay…");
    expect(statusLine({ state: "open", peers: 0 }, null)).toEqual({ lamp: "wait", text: "Connected · no phone open", retry: false });
    expect(statusLine({ state: "open", peers: 1 }, { device: "iPhone, Safari" }).text).toBe("Phone connected · iPhone, Safari");
    expect(statusLine({ state: "open", peers: 1 }, null).text).toBe("Phone connected");
    expect(statusLine({ state: "waiting", peers: 0, retryInMs: 7600 }, null)).toEqual(
      { lamp: "bad", text: "Can't reach the relay", countdown: "Trying again in 8 s", retry: true });
    expect(statusLine({ state: "waiting", peers: 0, retryInMs: 30_000, problem: "this pairing is already open in too many tabs" }, null, 1200))
      .toMatchObject({ text: "This pairing is already open in too many tabs", countdown: "Trying again in 2 s" });
  });
});
