// The tests and their photos, kept on the phone (IndexedDB) until the person
// clears them: the photos are needed to score them again with Python.

import type { TestRecord } from "../../src/testbench/bench";

const DB = "splinewire-bench";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore("tests", { keyPath: "id" });
      req.result.createObjectStore("photos");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(stores: string[], mode: IDBTransactionMode, fn: (t: IDBTransaction) => IDBRequest<T> | void): Promise<T | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(stores, mode);
    const req = fn(t);
    t.oncomplete = () => resolve(req ? req.result : undefined);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export async function saveTest(test: TestRecord, photo?: Blob): Promise<void> {
  await tx(["tests", "photos"], "readwrite", (t) => {
    t.objectStore("tests").put(test);
    if (photo) t.objectStore("photos").put(photo, test.id);
  });
}

export async function allTests(): Promise<TestRecord[]> {
  const tests = (await tx<TestRecord[]>(["tests"], "readonly", (t) => t.objectStore("tests").getAll())) ?? [];
  return tests.sort((a, b) => a.created.localeCompare(b.created));
}

export async function photoOf(id: string): Promise<Blob | undefined> {
  return tx<Blob>(["photos"], "readonly", (t) => t.objectStore("photos").get(id));
}

export async function deleteTest(id: string): Promise<void> {
  await tx(["tests", "photos"], "readwrite", (t) => {
    t.objectStore("tests").delete(id);
    t.objectStore("photos").delete(id);
  });
}

export async function clearAll(): Promise<void> {
  await tx(["tests", "photos"], "readwrite", (t) => {
    t.objectStore("tests").clear();
    t.objectStore("photos").clear();
  });
}
