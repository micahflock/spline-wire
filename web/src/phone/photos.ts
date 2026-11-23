// The phone's photos and what was measured in them, in IndexedDB.
//
// Photos never leave the phone (D1, D5). They are kept here, keyed by the id
// the page gives CoreClient, so that a reload keeps them and the core worker,
// which closes photos to save memory, can open one again from its bytes.
//
// Two object stores: "photos" holds one small record per photo (its settings,
// edits, status and last measurement document); "blobs" holds the files
// themselves under "<id>:<kind>". Listing the photos then never reads a photo.
// Each photo has the original (what the core measures), a preview downscaled
// for the editor, and a thumbnail for the list; the page shows at most one
// preview at a time, so it never holds several decoded photos (S1's memory).
// Once measured it also has its output files ("outputs", a few KB), so that
// the photo itself can be deleted to save space and the curve kept: copied,
// downloaded and sent as before, but no longer measured again or edited.
//
// The IDBFactory is passed in, so tests run against fake-indexeddb.

import type { Failure } from "../core/protocol";
import type { Edits, MeasurementDocument, Point, Side } from "../core/types";
import type { Exif } from "./exif";
import type { Corners, Name, Star } from "../relay/sync";
import type { CurveSync } from "./send";
import type { Shade } from "./settings";

export type Status = "queued" | "loading" | "measuring" | "done" | "error";
export type BlobKind = "original" | "preview" | "thumb" | "outputs";
/** The photo's own files, deleted to save space; the thumbnail and the outputs stay. */
export const PHOTO_FILES = ["original", "preview"] as const;
const ALL_KINDS = ["original", "preview", "thumb", "outputs"] as const;

/** A curve sent from this photo before revisions (D26): each send a curve of its own. Read by send.ts syncOf. */
export interface SentCurve {
  id: string;
  /** curveKey of what was sent, to tell whether the curve changed since. */
  key: string;
  at: string;
  delivered: boolean;
}

export interface PhotoRecord {
  /** Also the core's photo id. */
  id: string;
  name: string;
  type: string;
  size: number;
  /** ms since 1970. */
  added: number;
  /** Where it came from: the system camera (the file input), the library, or
   * the page's own camera (S2 wants to know which keep EXIF). */
  via: "take" | "choose" | "camera";
  /** Chain preset key and pin count the photo is measured with. */
  chain: string;
  pins: number;
  /** Its chain's colours (settings.Shade); absent on photos from before they could be chosen: light. */
  shade?: Shade;
  side: Side | Point;
  edits: Edits;
  status: Status;
  doc: MeasurementDocument | null;
  /** A failure that left no document (not a photo, a bug). */
  failure: Failure | null;
  /** The photo's size in px, upright, as the core sees it; null when this browser could not decode it. */
  pixels: { width: number; height: number } | null;
  /** The preview's size in px; null without one. */
  preview: { width: number; height: number } | null;
  /** Before D26; records since have `sync`. */
  sent?: SentCurve[];
  /** Its curve on the paired computer (send.ts). */
  sync?: CurveSync | null;
  /** Added while paired: its curve goes to the computer by itself once measured. */
  autoSend?: boolean;
  /** Starred, kept in step with the computer (relay/sync.ts). */
  star?: Star;
  /** The person's name for it, in place of `name` (the file's), kept in step the same way. */
  title?: Name;
  /** Points marked as sharp corners on the computer's drawing; the phone only keeps them in step. */
  corners?: Corners;
  /** The focal length settings the document was measured with (main.ts focalKey):
   * a photo measured before they changed is measured again when opened. */
  measuredWith: string | null;
  /** The tags the page read (exif.ts); missing in records from before D17. */
  exif?: Exif | null;
  /** A focal length this photo is measured with although its file does not
   * say so: a takePhoto still without the 35 mm tag (its lens's remembered
   * value), or a video frame (the frame's). Settings' override still wins. */
  focal?: PhotoFocal | null;
  /** For a photo from the page's camera. */
  camera?: CameraNote | null;
  /** When its photo was deleted to save space (ms since 1970): the curve stays, from its outputs. */
  photoDeleted?: number;
}

export interface PhotoFocal {
  focal35: number;
  why: "lens" | "frame";
  /** The lens it was learned from; null for the saved setting. */
  lens: string | null;
}

export interface CameraNote {
  label: string;
  still: "photo" | "frame";
  /** The overlay's last chain and the video size, to compare fields of view (focal.fovRatio). */
  frameChain: Point[];
  frameSize: { width: number; height: number };
  /** The live view's scale against the still's, once measured: 1 when they agree. */
  fov: number | null;
  ms: number;
  /** For a frame: why takePhoto gave no photo, if it was tried. */
  why?: string;
}

/** Photos kept by default (Settings, Photos kept): past it the oldest lose their
 * photo and keep their curve. A 12-megapixel photo is 3-6 MB; a curve a few tens of KB. */
export const KEEP_PHOTOS = 10;
/** Curves kept. Past it the oldest go altogether. */
export const KEEP_CURVES = 200;
const DB_VERSION = 1;

export class PhotoStore {
  private constructor(private readonly db: IDBDatabase) {}

  static open(factory: IDBFactory, name = "splinewire-phone"): Promise<PhotoStore> {
    return new Promise((resolve, reject) => {
      const req = factory.open(name, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains("photos")) db.createObjectStore("photos", { keyPath: "id" });
        if (!db.objectStoreNames.contains("blobs")) db.createObjectStore("blobs");
      };
      req.onsuccess = () => resolve(new PhotoStore(req.result));
      req.onerror = () => reject(req.error ?? new Error("could not open the photo store"));
      req.onblocked = () => reject(new Error("the photo store is open in an older tab: close it and reload"));
    });
  }

  /** Every photo, newest first. */
  async list(): Promise<PhotoRecord[]> {
    const all = await this.request<PhotoRecord[]>("photos", "readonly", (s) => s.getAll());
    return all.sort((a, b) => b.added - a.added);
  }

  get(id: string): Promise<PhotoRecord | undefined> {
    return this.request<PhotoRecord | undefined>("photos", "readonly", (s) => s.get(id));
  }

  /** Store a new photo with its files, in one transaction. */
  async add(record: PhotoRecord, blobs: Partial<Record<BlobKind, Blob>>): Promise<void> {
    await this.transaction(["photos", "blobs"], "readwrite", (tx) => {
      tx.objectStore("photos").put(record);
      for (const [kind, blob] of Object.entries(blobs)) {
        if (blob) tx.objectStore("blobs").put(blob, blobKey(record.id, kind as BlobKind));
      }
    });
  }

  /** Replace a photo's record (not its files). */
  async put(record: PhotoRecord): Promise<void> {
    await this.request("photos", "readwrite", (s) => s.put(record));
  }

  blob(id: string, kind: BlobKind): Promise<Blob | undefined> {
    return this.request<Blob | undefined>("blobs", "readonly", (s) => s.get(blobKey(id, kind)));
  }

  /** Store a blob of a photo's (its outputs) without touching its record. */
  async putBlob(id: string, kind: BlobKind, blob: Blob): Promise<void> {
    await this.request("blobs", "readwrite", (s) => s.put(blob, blobKey(id, kind)));
  }

  async delete(id: string): Promise<void> {
    await this.transaction(["photos", "blobs"], "readwrite", (tx) => {
      tx.objectStore("photos").delete(id);
      for (const kind of ALL_KINDS) tx.objectStore("blobs").delete(blobKey(id, kind));
    });
  }

  /** Delete a photo's own files and save its record (marked `photoDeleted`), in one transaction. */
  async deleteFiles(record: PhotoRecord): Promise<void> {
    await this.transaction(["photos", "blobs"], "readwrite", (tx) => {
      tx.objectStore("photos").put(record);
      for (const kind of PHOTO_FILES) tx.objectStore("blobs").delete(blobKey(record.id, kind));
    });
  }

  /** Everything: every photo, curve and file. */
  async clear(): Promise<void> {
    await this.transaction(["photos", "blobs"], "readwrite", (tx) => {
      tx.objectStore("photos").clear();
      tx.objectStore("blobs").clear();
    });
  }

  close(): void {
    this.db.close();
  }

  private request<T>(store: string, mode: IDBTransactionMode, make: (s: IDBObjectStore) => IDBRequest): Promise<T> {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(store, mode);
      const req = make(tx.objectStore(store));
      // Resolve when the transaction commits, not when the request succeeds:
      // a write is only safe once its transaction is done.
      tx.oncomplete = () => resolve(req.result as T);
      tx.onerror = () => reject(tx.error ?? req.error ?? new Error("photo store error"));
      tx.onabort = () => reject(tx.error ?? new Error("photo store write aborted (is the phone's storage full?)"));
    });
  }

  private transaction(stores: string[], mode: IDBTransactionMode, work: (tx: IDBTransaction) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(stores, mode);
      work(tx);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("photo store error"));
      tx.onabort = () => reject(tx.error ?? new Error("photo store write aborted (is the phone's storage full?)"));
    });
  }
}

/** What to delete to keep the phone's storage small. */
export interface Cleanup {
  /** Photos to delete, keeping their curves. */
  photos: string[];
  /** Photos to delete altogether: ones with no curve to keep, and the oldest curves. */
  whole: string[];
}

/** Past the newest `keepPhotos` photos (records newest first; 0 for every one), a
 * photo with a curve loses its photo and keeps the curve, and one without a curve
 * (it failed) goes. Past the newest `keepCurves` records, everything goes. Photos
 * not finished measuring, and those in `spare` (the one on screen), are left alone. */
export function cleanupPlan(records: readonly PhotoRecord[], keepPhotos: number, keepCurves = KEEP_CURVES,
  spare: ReadonlySet<string> = new Set()): Cleanup {
  const out: Cleanup = { photos: [], whole: [] };
  let withPhoto = 0;
  for (const [i, r] of records.entries()) {
    const finished = r.status === "done" || r.status === "error";
    if (!finished || spare.has(r.id)) {
      if (!r.photoDeleted) withPhoto++;
      continue;
    }
    if (i >= keepCurves) out.whole.push(r.id);
    else if (r.photoDeleted) continue;
    else if (withPhoto < keepPhotos) withPhoto++;
    else if (hasCurve(r)) out.photos.push(r.id);
    else out.whole.push(r.id);
  }
  return out;
}

/** A measurement that gave a curve, which outlives the photo. */
export function hasCurve(r: PhotoRecord): boolean {
  return !!r.doc?.ok && !!r.doc.curve;
}

function blobKey(id: string, kind: BlobKind): string {
  return `${id}:${kind}`;
}

/** A new photo id: time-ordered, unique on this phone, safe in a URL and as a Python dict key. */
export function newPhotoId(now: number = Date.now()): string {
  const rand = crypto.getRandomValues(new Uint8Array(4));
  return "p" + now.toString(36).padStart(9, "0") + Array.from(rand, (b) => b.toString(16).padStart(2, "0")).join("");
}
