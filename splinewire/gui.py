"""Desktop app: add chain photos, measure them, check the result.

Run with `splinewire-gui` (or the packaged SplineWire.exe). Photo paths
given on the command line, e.g. photos dropped onto the .exe, are added to
the list. `--selftest LOGFILE` runs a headless check of the packaged app.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from splinewire.chain import ChainSpec, default_chain_path, load_chain_spec
from splinewire.contact import spline_samples
from splinewire.process import PHOTO_SUFFIXES, PhotoResult, process_photo

APP_TITLE = "Spline Wire"
OUT_SUBDIR = "splinewire-out"
TRUTH_WARN_MM = 1.0   # the accuracy target; flag test-part photos that miss it

CHAIN_FIELDS = [
    ("pitch_mm", "Pitch (pin to pin)", "mm"),
    ("half_width_mm", "Half width (pin line to edge)", "mm"),
    ("ring_outer_mm", "Ring outer diameter", "mm"),
    ("ring_inner_mm", "Ring inner diameter", "mm"),
    ("n_pins", "Number of pins", ""),
]

STATUS_STYLE = {"ok": "#1a7f37", "warn": "#9a6700", "error": "#cf222e", "working": "#0969da", "pending": "#57606a"}


@dataclass
class PhotoItem:
    path: Path
    status: str = "pending"
    summary: str = "not processed"
    result: PhotoResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class Job:
    spec: ChainSpec
    side: str
    focal_35mm: float | None
    truth: Path | None
    out_dir: Path | None   # None: a folder next to each photo


# ---------------------------------------------------------------- settings

def settings_path() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) / "SplineWire" if base else Path.home() / ".config" / "splinewire"
    return root / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # settings are a convenience; never block the app on them


# ---------------------------------------------------------------- app

class App:
    def __init__(
        self,
        root: tk.Tk,
        photos: list[Path] | None = None,
        settings: dict | None = None,
        persist: bool = True,
        interactive: bool = True,
    ) -> None:
        self.root = root
        self.persist = persist
        self.interactive = interactive
        self.errors: list[str] = []   # unexpected callback errors, also written to the log file
        self.items: dict[str, PhotoItem] = {}
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._shown: str | None = None

        root.title(APP_TITLE)
        root.geometry("1280x820")
        root.minsize(980, 640)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.report_callback_exception = self._report_exception

        self._init_vars(load_settings() if settings is None else settings)
        self._build()
        self.add_photos(photos or [])
        root.after(100, self._poll)

    # -- state ---------------------------------------------------------

    def _init_vars(self, s: dict) -> None:
        try:
            default = load_chain_spec(default_chain_path())
            defaults = {k: getattr(default, k) for k, _, _ in CHAIN_FIELDS}
        except (OSError, KeyError, ValueError):
            defaults = {"pitch_mm": 10.0, "half_width_mm": 4.0, "ring_outer_mm": 5.0,
                        "ring_inner_mm": 2.0, "n_pins": 13}
        chain = {**defaults, **s.get("chain", {})}
        self.chain_vars = {k: tk.StringVar(value=str(chain[k])) for k, _, _ in CHAIN_FIELDS}
        self.side_var = tk.StringVar(value=s.get("side", "inside"))
        self.focal_var = tk.StringVar(value=s.get("focal_35mm", ""))
        self.truth_var = tk.StringVar(value=s.get("truth", ""))
        self.out_mode_var = tk.StringVar(value=s.get("out_mode", "beside"))
        self.out_dir_var = tk.StringVar(value=s.get("out_dir", ""))
        self.last_dir = s.get("last_dir", str(Path.home()))
        self.status_var = tk.StringVar(value="Add photos of the chain to begin.")

    def _settings(self) -> dict:
        return {
            "chain": {k: v.get() for k, v in self.chain_vars.items()},
            "side": self.side_var.get(),
            "focal_35mm": self.focal_var.get(),
            "truth": self.truth_var.get(),
            "out_mode": self.out_mode_var.get(),
            "out_dir": self.out_dir_var.get(),
            "last_dir": self.last_dir,
        }

    def _save_settings(self) -> None:
        if self.persist:
            save_settings(self._settings())

    def on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    # -- layout --------------------------------------------------------

    def _build(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=(0, 8, 8, 8))
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_photos(left)
        self._build_chain(left)
        self._build_options(left)
        self._build_output(left)
        self._build_actions(left)
        self._build_views(right)

        ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 3),
                  relief="sunken").pack(fill="x", side="bottom")

    def _build_photos(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Photos", padding=6)
        box.pack(fill="both", expand=True)
        frame = ttk.Frame(box)
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, columns=("photo", "result"), show="headings",
                                 height=9, selectmode="extended")
        self.tree.heading("photo", text="Photo")
        self.tree.heading("result", text="Result")
        self.tree.column("photo", width=170, stretch=True)
        self.tree.column("result", width=190, stretch=True)
        for status, color in STATUS_STYLE.items():
            self.tree.tag_configure(status, foreground=color)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.show_selected())

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Add photos…", command=self.browse_photos).pack(side="left")
        ttk.Button(row, text="Remove", command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(row, text="Clear", command=self.clear_photos).pack(side="left")

    def _build_chain(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Chain", padding=6)
        box.pack(fill="x", pady=(8, 0))
        for r, (key, label, unit) in enumerate(CHAIN_FIELDS):
            ttk.Label(box, text=label).grid(row=r, column=0, sticky="w", pady=1)
            ttk.Entry(box, textvariable=self.chain_vars[key], width=9, justify="right").grid(
                row=r, column=1, padx=4, pady=1)
            ttk.Label(box, text=unit).grid(row=r, column=2, sticky="w")
        ttk.Button(box, text="Load chain file…", command=self.browse_chain).grid(
            row=len(CHAIN_FIELDS), column=0, columnspan=3, sticky="w", pady=(4, 0))
        box.columnconfigure(0, weight=1)

    def _build_options(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Options", padding=6)
        box.pack(fill="x", pady=(8, 0))
        ttk.Label(box, text="The object was on the…").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(box, text="inside of the bend (chain wrapped around it)",
                        variable=self.side_var, value="inside").grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(box, text="outside of the bend (chain pressed into a hollow)",
                        variable=self.side_var, value="outside").grid(row=2, column=0, columnspan=3, sticky="w")

        ttk.Label(box, text="Focal length override").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(box, textvariable=self.focal_var, width=7, justify="right").grid(
            row=3, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(box, text="mm (35 mm equiv.)").grid(row=3, column=2, sticky="w", pady=(6, 0))
        ttk.Label(box, text="Leave blank to use the photo's EXIF data.", foreground="#57606a").grid(
            row=4, column=0, columnspan=3, sticky="w")

        ttk.Label(box, text="Compare to test-part truth file (optional)").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        truth_row = ttk.Frame(box)
        truth_row.grid(row=6, column=0, columnspan=3, sticky="ew")
        ttk.Entry(truth_row, textvariable=self.truth_var).pack(side="left", fill="x", expand=True)
        ttk.Button(truth_row, text="…", width=3, command=self.browse_truth).pack(side="left", padx=(4, 0))
        ttk.Button(truth_row, text="Clear", command=lambda: self.truth_var.set("")).pack(side="left")
        box.columnconfigure(2, weight=1)

    def _build_output(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Save results to", padding=6)
        box.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(box, text=f"a “{OUT_SUBDIR}” folder next to each photo",
                        variable=self.out_mode_var, value="beside").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="this folder:", variable=self.out_mode_var, value="folder").pack(side="left")
        ttk.Entry(row, textvariable=self.out_dir_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="…", width=3, command=self.browse_out_dir).pack(side="left")

    def _build_actions(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(10, 0))
        self.process_all_btn = ttk.Button(row, text="Process all", style="Accent.TButton",
                                          command=lambda: self.process(all_items=True))
        self.process_all_btn.pack(side="left")
        self.process_sel_btn = ttk.Button(row, text="Process selected",
                                          command=lambda: self.process(all_items=False))
        self.process_sel_btn.pack(side="left", padx=4)
        ttk.Button(row, text="Open output folder", command=self.open_output_folder).pack(side="right")

    def _build_views(self, parent: ttk.Frame) -> None:
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill="both", expand=True)
        self.photo_canvas = tk.Canvas(self.tabs, background="#2b2b2b", highlightthickness=0)
        self.curve_canvas = tk.Canvas(self.tabs, background="#ffffff", highlightthickness=0)
        details = ttk.Frame(self.tabs)
        self.details = tk.Text(details, wrap="word", font=("Consolas", 10), padx=10, pady=8,
                               relief="flat", state="disabled")
        dscroll = ttk.Scrollbar(details, orient="vertical", command=self.details.yview)
        self.details.configure(yscrollcommand=dscroll.set)
        self.details.pack(side="left", fill="both", expand=True)
        dscroll.pack(side="right", fill="y")
        self.tabs.add(self.photo_canvas, text="Detections")
        self.tabs.add(self.curve_canvas, text="Curve")
        self.tabs.add(details, text="Details")
        self.photo_canvas.bind("<Configure>", lambda _e: self._draw_photo())
        self.curve_canvas.bind("<Configure>", lambda _e: self._draw_curve())
        self._draw_photo()
        self._draw_curve()

    # -- photo list ----------------------------------------------------

    def browse_photos(self) -> None:
        pattern = " ".join(f"*{s}" for s in PHOTO_SUFFIXES)
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Choose chain photos", initialdir=self.last_dir,
            filetypes=[("Photos", pattern), ("All files", "*.*")])
        if paths:
            self.last_dir = str(Path(paths[0]).parent)
            self.add_photos([Path(p) for p in paths])

    def add_photos(self, paths: list[Path]) -> None:
        known = {item.path.resolve() for item in self.items.values()}
        added = None
        for p in paths:
            p = Path(p)
            if not p.is_file() or p.suffix.lower() not in PHOTO_SUFFIXES or p.resolve() in known:
                continue
            item = PhotoItem(p)
            iid = self.tree.insert("", "end", values=(p.name, item.summary), tags=(item.status,))
            self.items[iid] = item
            known.add(p.resolve())
            added = added or iid
        if added:
            self.tree.selection_set(added)
            self.status_var.set(f"{len(self.items)} photo(s). Press “Process all”.")

    def remove_selected(self) -> None:
        for iid in self.tree.selection():
            if self.items[iid].status != "working":
                self.tree.delete(iid)
                del self.items[iid]
        self.show_selected()

    def clear_photos(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.tree.delete(*self.tree.get_children())
        self.items.clear()
        self.show_selected()

    def _refresh_row(self, iid: str) -> None:
        item = self.items.get(iid)
        if item and self.tree.exists(iid):
            self.tree.item(iid, values=(item.path.name, item.summary), tags=(item.status,))

    # -- processing ----------------------------------------------------

    def _job(self) -> Job | None:
        try:
            values = {k: v.get().strip() for k, v in self.chain_vars.items()}
            spec = ChainSpec(
                pitch_mm=float(values["pitch_mm"]),
                half_width_mm=float(values["half_width_mm"]),
                ring_outer_mm=float(values["ring_outer_mm"]),
                ring_inner_mm=float(values["ring_inner_mm"]),
                n_pins=int(float(values["n_pins"])),
            )
        except ValueError as err:
            messagebox.showerror(APP_TITLE, f"Check the chain settings:\n{err}", parent=self.root)
            return None
        focal_text = self.focal_var.get().strip()
        try:
            focal = float(focal_text) if focal_text else None
            if focal is not None and focal <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(APP_TITLE, "The focal length override must be a positive number "
                                 "(or blank).", parent=self.root)
            return None
        truth = Path(self.truth_var.get()) if self.truth_var.get().strip() else None
        if truth is not None and not truth.is_file():
            messagebox.showerror(APP_TITLE, f"Truth file not found:\n{truth}", parent=self.root)
            return None
        out_dir = None
        if self.out_mode_var.get() == "folder":
            if not self.out_dir_var.get().strip():
                messagebox.showerror(APP_TITLE, "Choose an output folder.", parent=self.root)
                return None
            out_dir = Path(self.out_dir_var.get())
        return Job(spec, self.side_var.get(), focal, truth, out_dir)

    def process(self, all_items: bool) -> None:
        if self.worker and self.worker.is_alive():
            return
        iids = list(self.tree.get_children()) if all_items else list(self.tree.selection())
        if not iids:
            self.status_var.set("Nothing to process: add or select photos first.")
            return
        job = self._job()
        if job is None:
            return
        self._save_settings()
        for iid in iids:
            item = self.items[iid]
            item.status, item.summary = "pending", "queued"
            self._refresh_row(iid)
        self._set_busy(True)
        work = [(iid, self.items[iid].path) for iid in iids]
        self.worker = threading.Thread(target=self._run, args=(work, job), daemon=True)
        self.worker.start()

    def _run(self, work: list[tuple[str, Path]], job: Job) -> None:
        for n, (iid, path) in enumerate(work, 1):
            self.events.put(("start", iid, f"Processing {path.name} ({n}/{len(work)})…"))
            try:
                out_dir = job.out_dir or path.parent / OUT_SUBDIR
                result = process_photo(path, job.spec, out_dir, focal_35mm=job.focal_35mm,
                                       side=job.side, truth_path=job.truth)
                self.events.put(("done", iid, result))
            except Exception as err:  # report every failure in the list, keep going
                detail = "".join(traceback.format_exception_only(type(err), err)).strip()
                self.events.put(("error", iid, detail))
        self.events.put(("finished", None, len(work)))

    def _poll(self) -> None:
        try:
            self._drain_events()
        finally:  # an error while showing one result must not stop delivery of the rest
            self.root.after(100, self._poll)

    def _drain_events(self) -> None:
        try:
            while True:
                kind, iid, payload = self.events.get_nowait()
                item = self.items.get(iid) if iid else None
                if kind == "start" and item:
                    item.status, item.summary = "working", "processing…"
                    self.status_var.set(payload)
                elif kind == "done" and item:
                    item.result, item.error = payload, None
                    item.status, item.summary = _summarize(payload)
                elif kind == "error" and item:
                    item.result, item.error = None, payload
                    item.status, item.summary = "error", payload.splitlines()[-1][:80]
                elif kind == "finished":
                    self._set_busy(False)
                    failed = sum(1 for i in self.items.values() if i.status == "error")
                    self.status_var.set(f"Done: {payload} photo(s) processed"
                                        + (f", {failed} failed." if failed else "."))
                if iid:
                    self._refresh_row(iid)
                    if iid in self.tree.selection():
                        self.show_selected()
        except queue.Empty:
            pass

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.process_all_btn.configure(state=state)
        self.process_sel_btn.configure(state=state)
        self.root.configure(cursor="watch" if busy else "")

    # -- result views --------------------------------------------------

    def _selected_item(self) -> PhotoItem | None:
        sel = self.tree.selection()
        return self.items.get(sel[0]) if sel else None

    def show_selected(self) -> None:
        item = self._selected_item()
        self._draw_photo()
        self._draw_curve()
        self._write_details(item)

    def _draw_photo(self) -> None:
        c = self.photo_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        item = self._selected_item()
        if item is None:
            _canvas_message(c, w, h, "Add photos, then select one to see its detections.", "#dddddd")
            return
        try:
            if item.result is not None:
                img = Image.fromarray(_crop_to_rings(item.result)[:, :, ::-1].copy())
            else:
                from splinewire.pipeline import load_photo
                img = Image.fromarray(load_photo(item.path)[0])
        except Exception as err:
            _canvas_message(c, w, h, f"Cannot show this photo:\n{err}", "#ffb4b4")
            return
        img.thumbnail((w - 8, h - 8), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(img)
        c.create_image(w // 2, h // 2, image=self._preview_photo)
        if item.result is None:
            msg = item.error or "Not processed yet: detections appear here after processing."
            c.create_text(10, 10, anchor="nw", text=msg, fill="#ffb4b4" if item.error else "#dddddd",
                          width=w - 20, font=("Segoe UI", 10))

    def _draw_curve(self) -> None:
        c = self.curve_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        item = self._selected_item()
        if item is None or item.result is None:
            _canvas_message(c, w, h, "The measured curve appears here after processing.", "#57606a")
            return
        m = item.result.measurement
        curve = spline_samples(m.contacts_mm)
        pins, contacts = m.pins_mm, m.contacts_mm
        pts = np.vstack([curve, pins])
        lo, hi = pts.min(axis=0) - 6.0, pts.max(axis=0) + 6.0
        margin = 40
        scale = min((w - 2 * margin) / (hi[0] - lo[0]), (h - 2 * margin) / (hi[1] - lo[1]))
        ox = (w - scale * (hi[0] - lo[0])) / 2
        oy = (h - scale * (hi[1] - lo[1])) / 2

        def xy(p) -> tuple[float, float]:
            return ox + (p[0] - lo[0]) * scale, h - oy - (p[1] - lo[1]) * scale

        step = 10.0 if scale * 10 >= 25 else 50.0
        for gx in np.arange(np.ceil(lo[0] / step) * step, hi[0], step):
            x = xy((gx, 0))[0]
            c.create_line(x, 0, x, h, fill="#eef1f4")
        for gy in np.arange(np.ceil(lo[1] / step) * step, hi[1], step):
            y = xy((0, gy))[1]
            c.create_line(0, y, w, y, fill="#eef1f4")

        r = max(2.0, min(8.0, 2.5 * scale))  # nominal 5 mm ring
        c.create_line(*[v for p in pins for v in xy(p)], fill="#c8ced6", width=max(1, int(scale * 1.5)))
        for p in pins:
            x, y = xy(p)
            c.create_oval(x - r, y - r, x + r, y + r, outline="#8c959f")
        c.create_line(*[v for p in curve for v in xy(p)], fill="#1f2328", width=2, smooth=False)
        for p in contacts:
            x, y = xy(p)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#cf222e", outline="")

        x0, y0 = 16, h - 18
        c.create_line(x0, y0, x0 + step * scale, y0, width=3, fill="#1f2328")
        c.create_text(x0 + step * scale + 6, y0, anchor="w", text=f"{step:g} mm")
        c.create_text(12, 10, anchor="nw", fill="#57606a", font=("Segoe UI", 9), text=(
            "Seen from above, first pin at the origin. Grid "
            f"{step:g} mm.\nRed dots: curve points (target surface). Grey circles: chain pins. "
            "Black line: fitted curve."))

    def _write_details(self, item: PhotoItem | None) -> None:
        t = self.details
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", _details_text(item))
        t.configure(state="disabled")

    # -- misc ----------------------------------------------------------

    def browse_chain(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Chain file",
                                          filetypes=[("Chain YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        try:
            spec = load_chain_spec(Path(path))
        except Exception as err:
            messagebox.showerror(APP_TITLE, f"Could not load chain file:\n{err}", parent=self.root)
            return
        for key, var in self.chain_vars.items():
            var.set(str(getattr(spec, key)))

    def browse_truth(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Truth file",
                                          filetypes=[("Truth JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.truth_var.set(path)

    def browse_out_dir(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="Output folder")
        if path:
            self.out_dir_var.set(path)
            self.out_mode_var.set("folder")

    def open_output_folder(self) -> None:
        item = self._selected_item()
        if item is not None and item.result is not None:
            folder = item.result.outputs["json"].parent
        elif self.out_mode_var.get() == "folder" and self.out_dir_var.get().strip():
            folder = Path(self.out_dir_var.get())
        elif item is not None:
            folder = item.path.parent / OUT_SUBDIR
        else:
            return
        if not folder.is_dir():
            self.status_var.set(f"{folder} doesn't exist yet: process a photo first.")
            return
        _open_in_file_manager(folder)

    def _report_exception(self, exc, val, tb) -> None:
        detail = "".join(traceback.format_exception(exc, val, tb))
        self.errors.append(detail)
        log = _append_error_log(detail)
        if self.interactive:
            where = f"\n\nDetails were saved to {log}" if log else ""
            messagebox.showerror(APP_TITLE, f"Unexpected error:\n\n{detail[-1500:]}{where}",
                                 parent=self.root)


# ---------------------------------------------------------------- helpers

def _summarize(result: PhotoResult) -> tuple[str, str]:
    m = result.measurement
    if result.truth_comparison:
        t = result.truth_comparison
        text = f"max error {t['max_error_mm']:.2f} mm ({t['max_error_scaled_mm']:.2f} scale-fit)"
    else:
        text = f"{len(m.order.indices)} pins, tilt {m.rectification.tilt_deg:.0f} deg"
    if m.warnings:
        return "warn", f"{text}, {len(m.warnings)} warning(s)"
    if result.truth_comparison and result.truth_comparison["max_error_mm"] > TRUTH_WARN_MM:
        return "warn", f"{text} (over {TRUTH_WARN_MM:g} mm)"
    return "ok", text


def _details_text(item: PhotoItem | None) -> str:
    if item is None:
        return "Select a photo."
    lines = [str(item.path), ""]
    if item.error:
        return "\n".join(lines + ["Processing failed:", item.error, "",
                                  "Check that the whole chain is in the photo, in focus, "
                                  "and that the chain settings match your chain."])
    if item.result is None:
        return "\n".join(lines + ["Not processed yet."])
    res = item.result
    m, r = res.measurement, res.measurement.rectification
    focal_src = "estimated from the chain" if r.focal_estimated else "from EXIF or override"
    lines += [
        f"Pins on the chain      {len(m.order.indices)}",
        f"Missing pins (gaps)    {len(m.order.gaps)}",
        f"Stray rings ignored    {len(m.order.rejected)}",
        f"Camera tilt            {r.tilt_deg:.1f} deg",
        f"Focal length           {r.focal_px:.0f} px ({focal_src})",
        f"Link length error      rms {r.residual_rms_mm:.3f} mm, max {r.residual_max_mm:.3f} mm",
        f"Object side            {m.object_side} of the bend",
        f"Curve points           {len(m.contacts_mm)}",
    ]
    if res.truth_comparison:
        t = res.truth_comparison
        lines += ["", "Compared to truth file",
                  f"  max pin error        {t['max_error_mm']:.3f} mm",
                  f"  rms pin error        {t['rms_error_mm']:.3f} mm",
                  f"  after scale fit      {t['max_error_scaled_mm']:.3f} mm "
                  f"(part measures {100 * (t['scale'] - 1):+.2f}% vs design)",
                  "  A scale far from 0% means the test part printed off-size "
                  "(check its 50 mm bar) or the chain pitch setting is wrong."]
    lines += ["", "Warnings" if m.warnings else "No warnings."]
    lines += [f"  - {w}" for w in m.warnings]
    lines += ["", "Saved files"] + [f"  {p}" for p in res.outputs.values()]
    lines += ["", "In Fusion: fit a spline through the points in the -curve.csv file, "
                  "or insert the -curve.svg (1:1 scale, mm)."]
    return "\n".join(lines)


def _crop_to_rings(result: PhotoResult, margin: float = 0.25) -> np.ndarray:
    """The preview cropped to the detected rings plus a margin, so the chain
    fills the view even when it is small in the photo."""
    preview = result.preview
    rings = result.measurement.rings
    if not rings:
        return preview
    pts = np.array([r.center_px for r in rings])
    size = max(r.outer_axes_px[0] for r in rings)
    lo, hi = pts.min(axis=0) - size, pts.max(axis=0) + size
    pad = margin * (hi - lo).max()
    h, w = preview.shape[:2]
    x0, y0 = int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad))
    x1, y1 = int(min(w, hi[0] + pad)), int(min(h, hi[1] + pad))
    return preview[y0:y1, x0:x1]


def _append_error_log(detail: str) -> Path | None:
    try:
        path = settings_path().with_name("errors.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(detail + "\n")
        return path
    except OSError:
        return None


def _canvas_message(c: tk.Canvas, w: int, h: int, text: str, color: str) -> None:
    c.create_text(w // 2, h // 2, text=text, fill=color, width=w - 40,
                  justify="center", font=("Segoe UI", 11))


def _open_in_file_manager(folder: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError:
        pass


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--selftest"]:
        from splinewire.selftest import run_selftest
        return run_selftest(Path(args[1]) if len(args) > 1 else None)
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    App(root, photos=[Path(a) for a in args])
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
