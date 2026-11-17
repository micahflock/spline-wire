// Desktop Chromium check of the phone-capabilities spike page.
// Run from the repo root:  node web/spikes/phone-capabilities/check.mjs [--shots DIR]
//
// Generates test photos at run time (the repo's `splinewire synth` plus Pillow-made files),
// asks the Python code what it reads from each, and compares with what the page reports.
// The camera parts use Chromium's fake camera.
import { createServer } from "node:http";
import { readFile, mkdtemp, writeFile, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { tmpdir, networkInterfaces } from "node:os";
import { join, dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium, devices } = require("playwright");

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../..");
const shotsArg = process.argv.indexOf("--shots");
const SHOTS = shotsArg > 0 ? process.argv[shotsArg + 1] : null;

let failures = 0, passes = 0;
function check(cond, what, detail) {
  if (cond) { passes++; console.log("  ok   " + what); }
  else { failures++; console.log("  FAIL " + what + (detail !== undefined ? "  -> " + JSON.stringify(detail) : "")); }
}
const close = (a, b, tol = 1e-3) => a != null && b != null && Math.abs(a - b) <= tol;

// ---------- fixtures ----------
const FIXTURE_PY = String.raw`
import json, struct, sys
from pathlib import Path
from PIL import Image
from splinewire.camera import focal_px_from_exif, EXIF_IFD
from splinewire.process import photo_info

out = Path(sys.argv[1])

def rich_exif():
    ex = Image.Exif()
    ex[0x010F] = "TestMake"; ex[0x0110] = "Test Phone 1"; ex[0x0112] = 6; ex[0x0131] = "17.1"
    sub = ex.get_ifd(EXIF_IFD)
    from PIL.TiffImagePlugin import IFDRational
    sub[0x920A] = IFDRational(57, 10); sub[0xA405] = 26; sub[0xA434] = "back camera 5.7mm f/1.6"
    sub[0xA002] = 800; sub[0xA003] = 600; sub[0x9003] = "2026:11:17 10:20:30"
    sub[0x829A] = IFDRational(1, 120)
    gps = ex.get_ifd(0x8825); gps[1] = "N"; gps[2] = (IFDRational(52), IFDRational(31), IFDRational(0))
    return ex

# a big-endian ("MM", as iPhones write) TIFF/EXIF block built by hand
def tiff_mm(ifd0, exif, gps):
    TYPES = {"A": 2, "H": 3, "L": 4, "R": 5}
    def encode(t, v):
        if t == "A": b = v.encode() + b"\0"; return b, len(b)
        if t == "H": return struct.pack(">H", v), 1
        if t == "L": return struct.pack(">L", v), 1
        if t == "R": return struct.pack(">LL", *v), 1
    def ifd_size(d): return 2 + 12 * len(d) + 4
    def data_size(d): return sum(len(encode(t, v)[0]) for t, v in d.values() if len(encode(t, v)[0]) > 4)
    o0 = 8
    oe = o0 + ifd_size(ifd0) + 12 * 2 + data_size(ifd0)
    og = oe + ifd_size(exif) + data_size(exif)
    ifd0 = dict(ifd0); ifd0[0x8769] = ("L", oe); ifd0[0x8825] = ("L", og)
    oe = o0 + ifd_size(ifd0) + data_size(ifd0)
    og = oe + ifd_size(exif) + data_size(exif)
    ifd0[0x8769] = ("L", oe); ifd0[0x8825] = ("L", og)
    blob = bytearray(b"MM\0*" + struct.pack(">L", o0))
    for d, start in ((ifd0, o0), (exif, oe), (gps, og)):
        assert len(blob) == start
        data_at = start + ifd_size(d); entries = b""; data = b""
        for tag in sorted(d):
            t, v = d[tag]; b, count = encode(t, v)
            if t == "A": count = len(b)
            if len(b) <= 4: val = b.ljust(4, b"\0")
            else: val = struct.pack(">L", data_at + len(data)); data += b
            entries += struct.pack(">HHL", tag, TYPES[t], count) + val
        blob += struct.pack(">H", len(d)) + entries + b"\0\0\0\0" + data
    return bytes(blob)

MM = tiff_mm(
    {0x010F: ("A", "BigEndianMake"), 0x0110: ("A", "BE Phone"), 0x0112: ("H", 1)},
    {0x920A: ("R", (6860, 1000)), 0xA405: ("H", 24), 0xA434: ("A", "main camera 6.86mm"),
     0x829A: ("R", (1, 60)), 0x9003: ("A", "2026:11:17 11:00:00"), 0xA002: ("L", 640), 0xA003: ("L", 480)},
    {0x0001: ("A", "N")})

im = Image.new("RGB", (800, 600), (120, 160, 200))
im.save(out / "rich-le.jpg", quality=90, exif=rich_exif())
Image.new("RGB", (640, 480), (90, 90, 90)).save(out / "rich-be.jpg", quality=90, exif=b"Exif\0\0" + MM)
Image.new("RGB", (640, 480), (200, 100, 50)).save(out / "no-exif.jpg", quality=90)
Image.new("RGB", (64, 48), (0, 200, 0)).save(out / "image.png")
(out / "notes.txt").write_text("not a photo\n")
heic_ok = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    Image.new("RGB", (512, 384), (30, 60, 90)).save(out / "photo.heic", exif=rich_exif())
    heic_ok = True
except Exception as e:
    print("no HEIC fixture:", e, file=sys.stderr)

def expected(p):
    with Image.open(p) as im:
        ex = im.getexif(); sub = ex.get_ifd(EXIF_IFD); gps = ex.get_ifd(0x8825)
        g = lambda tag: sub.get(tag, ex.get(tag))
        num = lambda v: None if v is None else float(v)
        return {
            "size": f"{im.size[0]}x{im.size[1]}", "format": im.format,
            "focal_px": focal_px_from_exif(im), "info": photo_info(p),
            "Make": g(0x010F), "Model": g(0x0110), "Orientation": g(0x0112),
            "LensModel": g(0xA434), "FocalLength": num(g(0x920A)), "FocalLengthIn35mmFilm": num(g(0xA405)),
            "PixelXDimension": g(0xA002), "PixelYDimension": g(0xA003),
            "DateTimeOriginal": g(0x9003), "ExposureTime": num(g(0x829A)), "gps_present": len(gps) > 0,
            "has_exif": len(ex) > 0,
        }

names = ["synth.jpg", "rich-le.jpg", "rich-be.jpg", "no-exif.jpg", "image.png"] + (["photo.heic"] if heic_ok else [])
print(json.dumps({n: expected(out / n) for n in names}, default=str))
`;

async function makeFixtures(dir) {
  execFileSync("uv", ["run", "--quiet", "splinewire", "synth", "--shape", "pipe", "--size", "1600x1200", "--out", dir],
    { cwd: REPO, stdio: ["ignore", "ignore", "inherit"] });
  execFileSync("mv", [join(dir, "pipe.jpg"), join(dir, "synth.jpg")]);
  await writeFile(join(dir, "fixtures.py"), FIXTURE_PY);
  const out = execFileSync("uv", ["run", "--quiet", "python", join(dir, "fixtures.py"), dir],
    { cwd: REPO, stdio: ["ignore", "pipe", "inherit"] });
  return JSON.parse(out.toString());
}

// ---------- static server ----------
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".md": "text/plain" };
function serve() {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const file = join(HERE, path.endsWith("/") ? path + "index.html" : path);
    if (!file.startsWith(HERE)) { res.writeHead(403).end(); return; }
    try { const body = await readFile(file); res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" }).end(body); }
    catch { res.writeHead(404).end("not found"); }
  });
  return new Promise((r) => server.listen(0, "0.0.0.0", () => r(server)));
}

async function waitPhotos(page, n) {
  await page.waitForFunction((n) => window.__spike.report.s2_photos.length >= n, n, { timeout: 20000 });
  return page.evaluate(() => window.__spike.report.s2_photos);
}

// ---------- main ----------
const dir = await mkdtemp(join(tmpdir(), "phone-cap-"));
const server = await serve();
const port = server.address().port;
const URL0 = `http://localhost:${port}/`;
let browser;
try {
  console.log("fixtures in " + dir);
  const exp = await makeFixtures(dir);

  browser = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
  const ctx = await browser.newContext();
  await ctx.grantPermissions(["camera", "clipboard-read", "clipboard-write"], { origin: URL0.slice(0, -1) });
  const page = await ctx.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  await page.goto(URL0);

  console.log("S2: EXIF parsing vs the Python code");
  const order = Object.keys(exp);
  let n = 0;
  for (const name of order) {
    await page.setInputFiles("#in-library", join(dir, name));
    const photos = await waitPhotos(page, ++n);
    const got = photos[n - 1], e = exp[name];
    console.log(` ${name}`);
    check(got.route === "library", "route recorded", got.route);
    check(got.stored_size === e.size, `stored size ${e.size}`, got.stored_size);
    if (e.focal_px == null) {
      check(got.focal.focal_px == null, "no focal_px (as Python)", got.focal);
    } else {
      check(close(got.focal.focal_px, e.focal_px, 1e-3), `focal_px ${e.focal_px.toFixed(3)} matches Python`, got.focal.focal_px);
      check(close(got.focal.focal_35mm, e.info.focal_35mm), `focal_35mm ${e.info.focal_35mm} matches photo_info`, got.focal.focal_35mm);
    }
    if (e.has_exif) {
      // pillow-heif moves a HEIC's rotation into the HEIF container and reports Orientation 1;
      // the page reads the EXIF bytes, which still say 6
      const keys = ["Make", "Model", "Orientation", "LensModel", "PixelXDimension", "PixelYDimension", "DateTimeOriginal"]
        .filter((k) => !(k === "Orientation" && name.endsWith(".heic")));
      for (const k of keys) {
        const want = e[k] == null ? null : (typeof e[k] === "string" ? e[k].replace(/\0/g, "").trim() : e[k]);
        const have = got.exif[k];
        check(String(have) === String(want), `${k} = ${want}`, have);
      }
      for (const k of ["FocalLength", "FocalLengthIn35mmFilm", "ExposureTime"]) {
        const have = got.exif[k] == null ? null : parseFloat(String(got.exif[k]));
        check(e[k] == null ? have == null : close(have, e[k], 1e-4 * Math.max(1, e[k])), `${k} = ${e[k]}`, got.exif[k]);
      }
      check(got.gps_present === e.gps_present, `GPS present = ${e.gps_present}`, got.gps_present);
      check(!JSON.stringify(got).includes("52") || !e.gps_present || !JSON.stringify(got.exif).includes("52"), "GPS values not shown");
    } else {
      check(!got.exif, "no EXIF reported", got.exif_status);
    }
    if (name.endsWith(".heic")) check(/HEIC/.test(got.exif_status) && got.format.startsWith("HEIF"), "HEIC said clearly", got.exif_status);
    if (name === "no-exif.jpg") check(/no EXIF/.test(got.exif_status), "says JPEG with no EXIF", got.exif_status);
    if (name === "image.png") check(got.format === "PNG" && /no EXIF/.test(got.exif_status), "PNG, no EXIF", [got.format, got.exif_status]);
    if (name === "rich-be.jpg") check(/MM/.test(got.exif_byte_order), "big-endian EXIF parsed", got.exif_byte_order);
  }

  console.log(" other routes and a non-image file");
  await page.setInputFiles("#in-camera", join(dir, "synth.jpg"));
  let photos = await waitPhotos(page, ++n);
  check(photos[n - 1].route === "camera" && close(photos[n - 1].focal.focal_px, exp["synth.jpg"].focal_px), "camera input route", photos[n - 1].route);
  await page.setInputFiles("#in-files", join(dir, "notes.txt"));
  photos = await waitPhotos(page, ++n);
  check(photos[n - 1].route === "files" && photos[n - 1].format === "unknown" && /could not decode/.test(photos[n - 1].decode || ""),
    "text file: unknown format, decode failure recorded", [photos[n - 1].format, photos[n - 1].decode]);
  await page.setInputFiles("#in-jpeg", join(dir, "rich-le.jpg"));
  photos = await waitPhotos(page, ++n);
  check(photos[n - 1].route === "jpeg", "JPEG-only route", photos[n - 1].route);
  await page.locator(".result select").first().selectOption("Photo Library");
  check(await page.evaluate((i) => window.__spike.report.s2_photos[i].menu_choice, n - 1) === "Photo Library", "menu choice recorded");

  console.log("S3: fake camera");
  await page.click("#btn-start");
  await page.waitForFunction(() => /Streaming \d+x\d+/.test(document.getElementById("stream-status").textContent), null, { timeout: 15000 });
  const s3a = await page.evaluate(() => window.__spike.report.s3);
  check(Array.isArray(s3a.devices) && s3a.devices.length > 0 && s3a.devices[0].label !== "(no label)", "devices listed with labels", s3a.devices);
  check(s3a.capabilities && s3a.capabilities.width && s3a.capabilities.width.max > 0, "track capabilities recorded", s3a.capabilities && s3a.capabilities.width);
  check(s3a.settings && s3a.settings.width > 0, "track settings recorded", s3a.settings);

  await page.click("#btn-ladder");
  await page.waitForFunction(() => !document.getElementById("btn-ladder").disabled, null, { timeout: 90000 });
  const ladder = await page.evaluate(() => window.__spike.report.s3.ladder);
  console.log("   ladder: " + ladder.map((l) => `${l.requested} -> ${l.delivered || l.error} @ ${l.measured_fps}`).join("; "));
  check(ladder.length >= 6 && ladder.every((l) => /^\d+x\d+$/.test(l.delivered || "")), "every ladder step delivered a size", ladder);
  check(ladder.every((l) => l.measured_fps > 1), "frame rate measured on every step", ladder.map((l) => l.measured_fps));
  check(ladder.some((l) => l.delivered === "1920x1080"), "1920x1080 delivered when asked", ladder.map((l) => l.delivered));

  await page.click("#btn-photo-caps");
  await page.waitForFunction(() => window.__spike.report.s3.image_capture.photoCapabilities || window.__spike.report.s3.image_capture.photoCapabilities_error, null, { timeout: 15000 });
  const ic = await page.evaluate(() => window.__spike.report.s3.image_capture);
  check(ic.available === true && ic.photoCapabilities && ic.photoCapabilities.imageWidth, "getPhotoCapabilities recorded", ic);

  await page.click("#btn-takephoto");
  photos = await waitPhotos(page, ++n);
  const tp = photos[n - 1];
  check(tp.route === "imagecapture" && /^\d+x\d+$/.test(tp.decoded_size || "") && tp.bytes > 0, `takePhoto blob analysed (${tp.decoded_size}, ${tp.bytes} B, ${tp.exif_status})`, tp);

  await page.click("#btn-grab");
  await page.waitForFunction(() => window.__spike.report.s3.fov.frame, null, { timeout: 10000 });
  await page.click("#btn-still-ic");
  await page.waitForFunction(() => window.__spike.report.s3.fov.still, null, { timeout: 20000 });
  n++;
  await page.locator("#ov-scale").fill("80");
  await page.click("#btn-fov-record");
  const fov = await page.evaluate(() => window.__spike.report.s3.fov);
  check(/^\d+x\d+$/.test(fov.frame.size) && /^\d+x\d+$/.test(fov.still.size) && fov.match_percent === 80, `FOV frame ${fov.frame.size} vs still ${fov.still.size}, match recorded`, fov);

  await page.fill("#device-note", "desktop test");
  await page.click("#btn-copy");
  await page.waitForFunction(() => /Copied/.test(document.getElementById("copy-status").textContent), null, { timeout: 5000 });
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  check(clip.includes("Phone: desktop test") && clip.includes("----- full JSON -----"), "Copy report puts the report on the clipboard");
  const json = JSON.parse(clip.split("----- full JSON -----\n")[1]);
  check(json.s2_photos.length === n && json.environment.secureContext === true && json.environment.userAgent, "report JSON parses and holds every photo + environment", json.s2_photos.length);
  check(!pageErrors.length, "no uncaught page errors", pageErrors);
  await ctx.close();

  console.log("Failure paths");
  // ImageCapture missing (as on iOS Safari)
  const ctx2 = await browser.newContext();
  await ctx2.addInitScript(() => { delete window.ImageCapture; });
  const p2 = await ctx2.newPage();
  await p2.goto(URL0);
  await p2.click("#btn-takephoto");
  const ph2 = await waitPhotos(p2, 1);
  check(/ImageCapture is not available/.test(ph2[0].error || ""), "no ImageCapture: recorded as such", ph2[0]);
  await ctx2.close();

  // camera permission denied (headless Chromium cannot show a prompt, so the denial is simulated)
  const ctx3 = await browser.newContext();
  await ctx3.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Permission denied", "NotAllowedError"));
  });
  const p3 = await ctx3.newPage();
  await p3.goto(URL0);
  await p3.click("#btn-start");
  await p3.waitForFunction(() => window.__spike.report.errors.length > 0, null, { timeout: 20000 });
  const err3 = await p3.evaluate(() => window.__spike.report.errors);
  check(/NotAllowedError/.test(err3[0].error), `permission denied recorded (${err3[0].error})`, err3);
  check(/NotAllowedError/.test(await p3.innerHTML("#stream-status")), "denial shown on the page");
  await p3.click("#btn-takephoto");
  const ph3 = await waitPhotos(p3, 1);
  check(/camera could not be started/.test(ph3[0].error || ""), "takePhoto without camera: recorded", ph3[0]);
  await ctx3.close();

  // insecure context: plain http on a non-loopback address
  const ip = Object.values(networkInterfaces()).flat().find((i) => i && i.family === "IPv4" && !i.internal);
  if (ip) {
    const p4 = await (await browser.newContext()).newPage();
    await p4.goto(`http://${ip.address}:${port}/`);
    const banner = await p4.textContent("#secure-banner");
    await p4.click("#btn-start");
    await p4.waitForFunction(() => window.__spike.report.errors.length > 0, null, { timeout: 10000 });
    const err4 = await p4.evaluate(() => window.__spike.report.errors);
    check(/Not a secure context/.test(banner) && /secure context/.test(err4[0].error), "insecure context: banner + error recorded", err4);
  } else console.log("  skip insecure-context test (no non-loopback IPv4 address)");

  if (SHOTS) {
    const phone = await browser.newContext({ ...devices["iPhone 13"] });
    const p5 = await phone.newPage();
    await p5.goto(URL0);
    await p5.setInputFiles("#in-library", join(dir, "synth.jpg"));
    await waitPhotos(p5, 1);
    await p5.screenshot({ path: join(SHOTS, "phone-top.png") });
    await p5.screenshot({ path: join(SHOTS, "phone-full.png"), fullPage: true });
    const overflow = await p5.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    check(overflow <= 0, "no horizontal scroll at phone width", overflow);
    console.log("screenshots in " + SHOTS);
  }
} finally {
  if (browser) await browser.close();
  server.close();
  await rm(dir, { recursive: true, force: true });
}
console.log(`\n${passes} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
