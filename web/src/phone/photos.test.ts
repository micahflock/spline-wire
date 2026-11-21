import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vitest";
import { KEEP_PHOTOS, PhotoStore, cleanupPlan, newPhotoId, type PhotoRecord } from "./photos";

function record(id: string, added: number): PhotoRecord {
  return {
    id, name: `${id}.jpg`, type: "image/jpeg", size: 3, added, via: "take", chain: "snap", pins: 13, side: "inside",
    edits: { add: [], remove: [] }, status: "queued", doc: null, failure: null, pixels: { width: 40, height: 30 },
    preview: { width: 40, height: 30 }, sent: [], measuredWith: null,
  };
}

const blob = (text: string) => new Blob([text], { type: "image/jpeg" });

describe("PhotoStore", () => {
  it("keeps photos and their files across opening the database again (a reload)", async () => {
    const factory = new IDBFactory();
    const a = await PhotoStore.open(factory);
    await a.add(record("one", 1), { original: blob("abc"), thumb: blob("t") });
    await a.add(record("two", 2), { original: blob("defg") });
    a.close();

    const b = await PhotoStore.open(factory);
    expect((await b.list()).map((r) => r.id)).toEqual(["two", "one"]);      // newest first
    expect(await (await b.blob("one", "original"))!.text()).toBe("abc");
    expect(await (await b.blob("one", "thumb"))!.text()).toBe("t");
    expect(await b.blob("two", "preview")).toBeUndefined();
    const r = (await b.get("one"))!;
    r.status = "done";
    r.edits = { add: [[1, 2]], remove: [] };
    await b.put(r);
    expect((await b.get("one"))!.edits.add).toEqual([[1, 2]]);
    expect(await (await b.blob("one", "original"))!.text()).toBe("abc");   // put leaves the files alone
  });

  it("deletes a photo with its files", async () => {
    const s = await PhotoStore.open(new IDBFactory());
    await s.add(record("one", 1), { original: blob("abc"), preview: blob("p"), thumb: blob("t") });
    await s.delete("one");
    expect(await s.list()).toEqual([]);
    for (const kind of ["original", "preview", "thumb"] as const) expect(await s.blob("one", kind)).toBeUndefined();
  });

  it("deletes a photo's own files and keeps its curve's", async () => {
    const factory = new IDBFactory();
    const s = await PhotoStore.open(factory);
    await s.add(record("one", 1), { original: blob("abc"), preview: blob("p"), thumb: blob("t") });
    await s.putBlob("one", "outputs", blob("{}"));
    const r = { ...(await s.get("one"))!, photoDeleted: 5 };
    await s.deleteFiles(r);
    s.close();
    const b = await PhotoStore.open(factory);
    expect((await b.get("one"))!.photoDeleted).toBe(5);
    expect(await b.blob("one", "original")).toBeUndefined();
    expect(await b.blob("one", "preview")).toBeUndefined();
    expect(await (await b.blob("one", "thumb"))!.text()).toBe("t");
    expect(await (await b.blob("one", "outputs"))!.text()).toBe("{}");
    await b.delete("one");
    expect(await b.blob("one", "outputs")).toBeUndefined();
  });

  it("clears everything", async () => {
    const s = await PhotoStore.open(new IDBFactory());
    for (let i = 0; i < 3; i++) await s.add(record(`p${i}`, i), { original: blob(String(i)), thumb: blob("t") });
    await s.clear();
    expect(await s.list()).toEqual([]);
    expect(await s.blob("p1", "thumb")).toBeUndefined();
  });

  it("makes ids that sort by time and differ", () => {
    const a = newPhotoId(1000), b = newPhotoId(1000), c = newPhotoId(2000);
    expect(a).not.toBe(b);
    expect(a.slice(0, 3)).toBe(b.slice(0, 3));
    expect(/^p[0-9a-z]+$/.test(c)).toBe(true);
    expect(c > a).toBe(true);
  });
});

describe("cleanupPlan", () => {
  const measured = (id: string, ok = true, extra: Partial<PhotoRecord> = {}): PhotoRecord => ({
    ...record(id, 0), status: ok ? "done" : "error",
    doc: { ok, curve: ok ? { points: [[0, 0], [1, 1]], pins: [] } : null } as unknown as PhotoRecord["doc"], ...extra,
  });

  it("keeps the newest photos and the curves of older ones", () => {
    const rs = ["a", "b", "c", "d"].map((id) => measured(id));
    expect(cleanupPlan(rs, 2)).toEqual({ photos: ["c", "d"], whole: [] });
    expect(cleanupPlan(rs, 0)).toEqual({ photos: ["a", "b", "c", "d"], whole: [] });
    expect(cleanupPlan(rs, 10)).toEqual({ photos: [], whole: [] });
  });

  it("deletes older photos that gave no curve altogether", () => {
    const rs = [measured("a"), measured("b", false), measured("c", false)];
    expect(cleanupPlan(rs, 2)).toEqual({ photos: [], whole: ["c"] });
  });

  it("counts only photos still kept, and spares those being measured and the one on screen", () => {
    const rs = [measured("a", true, { photoDeleted: 1 }), { ...measured("b"), status: "measuring" as const }, measured("c"), measured("d"), measured("e")];
    // a has lost its photo already; b (measuring) and c (spared) keep theirs and fill the two places.
    expect(cleanupPlan(rs, 2, 200, new Set(["c"]))).toEqual({ photos: ["d", "e"], whole: [] });
    expect(cleanupPlan(rs, 3, 200, new Set(["c"]))).toEqual({ photos: ["e"], whole: [] });
  });

  it("drops the oldest curves past the curve limit", () => {
    const rs = ["a", "b", "c"].map((id) => measured(id, true, id === "a" ? {} : { photoDeleted: 1 }));
    expect(cleanupPlan(rs, 1, 2)).toEqual({ photos: [], whole: ["c"] });
  });

  it("keeps ten photos unless told otherwise", () => {
    expect(KEEP_PHOTOS).toBe(10);
  });
});
