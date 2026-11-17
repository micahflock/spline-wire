# Phone capabilities spike (S2, S3)

`index.html` is a single static page (no build, no external scripts) for spikes S2 and S3 in
`docs/web-build.md`. It must be served over **HTTPS** for the camera parts; the photo parts also work
over plain HTTP. Nothing leaves the phone: photos are read in the page and only the text report is
copied or shared.

Check in desktop Chromium (fake camera, test photos generated at run time), from the repo root:

    node web/spikes/phone-capabilities/check.mjs [--shots DIR]

## For the owner: what to do on the phone

Do this once on the iPhone in Safari and once on the Android phone in Chrome. Have the chain or any
object nearby to photograph; the photos themselves are not sent anywhere.

1. Open the page's HTTPS address. The green box at the top must say "Secure context".
2. **0. About this phone**: type the phone model and OS version.
3. **1. Photos** — one photo per button, in order. After each, the result card appears under the
   buttons; if the phone showed a menu, set **"What you tapped in the menu"** on that card.
   - **A. Camera**: take a photo. (iPhone: the camera should open directly.)
   - **B. Photo library**: iPhone: choose *Photo Library* and pick a photo taken with the normal
     Camera app; then tap B again and choose *Take Photo*. Android: pick a camera photo from the
     gallery/photo picker.
   - **C. Files**: iPhone: first save a camera photo to Files (Photos app, Share, *Save to Files*),
     then tap C, choose *Choose File*, and pick it. Android: choose *Files*/*Documents* (not the photo
     picker), and pick a camera photo from DCIM/Camera.
   - **D. Library, JPEG only**: pick the same photo as in B.
   - **E. Live camera still**: allow camera access. On iPhone this will say ImageCapture is not
     available; that is a result too.
   Each card says in its first line whether the focal length survived.
4. **2. Live camera stream**: tap **Start camera** (allow access), then **Try resolution ladder** and
   wait for it to finish (about 20 s; the preview restarts several times). On Android also tap
   **Photo capabilities** and **takePhoto at max size**. If the phone has several rear cameras, the
   dropdown lists them; you can rerun the ladder on another one.
5. **Field of view**: lean the phone against something so it can't move, pointed at a page of text or
   a ruler. Tap **Grab video frame**, then **Still via takePhoto** (Android) — or on iPhone, without
   moving the phone, **A. Camera** and then **Use latest photo**. Drag **Frame size** until the
   red-outlined frame lines up with the still underneath, tap **Record match**.
6. **3. Send the results back**: add any notes (menus seen, prompts, which camera opened), then tap
   **Share** or **Copy report** and paste the text into a message. If both fail, select all the text
   in the box and copy it.

## What the page records

- Per photo: route (button) and menu choice, file name, MIME type, bytes, format (JPEG/HEIC/PNG/…),
  stored and decoded pixel size, the JPEG segments (shows re-encoding: e.g. JFIF and no Exif), and the
  EXIF tags Make, Model, LensMake, LensModel, FocalLength, FocalLengthIn35mmFilm, Orientation,
  PixelX/YDimension, DateTimeOriginal, ExposureTime, FNumber, ISO, Software; GPS presence only.
- focal_px the way `splinewire.camera.focal_px_from_exif` computes it: FocalLengthIn35mmFilm (Exif IFD,
  else IFD0) × the stored image diagonal in px / 43.27 mm. FocalLength alone is not used by the
  pipeline (it needs the sensor size).
- Stream: devices and labels, track capabilities/settings, each ladder step's delivered size and
  measured frame rate, ImageCapture photo capabilities/settings and takePhoto results, frame vs still
  sizes and the recorded field-of-view match.
- Environment (user agent, screen, devicePixelRatio, secure context, which APIs exist) and every error
  with its text.

HEIC files: the EXIF and size are found by a byte scan (good enough for a spike), and browsers other
than Safari cannot decode them, so no preview.
