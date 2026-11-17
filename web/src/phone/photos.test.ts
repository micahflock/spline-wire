import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vitest";
import { PhotoStore, newPhotoId, type PhotoRecord } from "./photos";

function record(id: string, added: number): PhotoRecord {
  return {
    id, name: `${id}.jpg`, type: "image/jpeg", size: 3, added, via: "take", chain: "default", pins: 13, side: "inside",
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

  it("prunes all but the newest", async () => {
    const s = await PhotoStore.open(new IDBFactory());
    for (let i = 0; i < 5; i++) await s.add(record(`p${i}`, i), { original: blob(String(i)) });
    expect(await s.prune(3)).toEqual(["p1", "p0"]);
    expect((await s.list()).map((r) => r.id)).toEqual(["p4", "p3", "p2"]);
    expect(await s.blob("p0", "original")).toBeUndefined();
  });

  it("makes ids that sort by time and differ", () => {
    const a = newPhotoId(1000), b = newPhotoId(1000), c = newPhotoId(2000);
    expect(a).not.toBe(b);
    expect(a.slice(0, 3)).toBe(b.slice(0, 3));
    expect(/^p[0-9a-z]+$/.test(c)).toBe(true);
    expect(c > a).toBe(true);
  });
});
