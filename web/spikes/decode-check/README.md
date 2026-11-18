# Decoding check (TypeScript port, D18)

`index.html` is a test page for a phone. It answers two questions the port cannot answer on a
desktop:

1. **Does this browser decode a phone's JPEG to the pixels Pillow decodes?** The pipeline was
   checked against Python with Pillow's decoder. Chromium's decoder (libjpeg-turbo, as Pillow's) gives
   the same pixels exactly (`e2e/decode.spec.ts`). Safari on iPhone decodes with Apple's own library,
   and may differ by a gray level here and there; the page says how many pixels differ and what that
   does to the pins.
2. **How fast is the TypeScript engine on this phone?** It measures the test photo and a photo the
   person picks, and shows the times.

The test photo (`photo.jpg`, 2000x1500, colour, 4:2:0, stored on its side with EXIF orientation 6),
Pillow's gray image of it (`pillow-gray.png`) and Python's pins (`python.json`) are written by
`uv run python web/spikes/decode-check/make.py`. `e2e/decode-check.spec.ts` checks the page in
Chromium.

## For the owner

On the iPhone (Safari) and an Android phone (Chrome), open `spikes/decode-check/` on the site:

1. Tap **Run the check**. It takes a few seconds.
2. Tap **Choose a photo** and pick a photo of the chain taken with the phone's camera app (a normal
   12-megapixel photo). Measuring it can take several seconds; the page does not respond meanwhile.
3. Tap **Copy the report** and paste it into a message.
