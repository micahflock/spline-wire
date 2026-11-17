# Tools: retaking the screenshots

The screenshots in `../screenshots/` are produced by a script so they can be retaken after a UI change.

```bash
# from the repo root
uv sync
uv run python "for UI review/tools/make_sample_photos.py" /tmp/sw-photos        # six sample photos: five simulated, one plain noise (~10 MB; not committed)

# start the app on an empty scratch workspace and scratch settings
APPDATA=/tmp/sw-appdata uv run splinewire-app --no-browser --port 8765 --workspace /tmp/sw-ws &

# take them (Playwright must be resolvable: require('playwright'))
SW_PHOTOS=/tmp/sw-photos node "for UI review/tools/capture_screenshots.mjs"
```

Notes:

- Start with an **empty** workspace each time, or the photo list doubles. Stop the app between runs
  (the page's Quit button, or `POST /api/quit` with the token).
- `SW_URL` / `SW_PHONE` / `SW_OUT` override the addresses and output folder. The phone page is reached through a **non-loopback address of the same
  machine** (default `192.0.2.2`; set `SW_PHONE` to yours, see `hostname -I`) because the server treats any non-loopback client
  as "the phone". On loopback the phone page still renders, but uploads count as "Added here".
- `d15`–`d17` edit the app's `/api/state` response in the browser (Playwright route interception) to show the firewall
  and phone-connected notices, which the app only produces on Windows or after a real phone has connected.
- The script's selectors and function names (`viewer`, `state`, `chainBox`, `#opt-pick`, …) come from `desktop.html` /
  `phone.html`; a UI rewrite will break it and it will need to be updated.
- `make_sample_photos.py` uses seed 3 and fixed environments, so the photos (and detections) are repeatable for a given version of the simulator.
