// Where the pairing, the phone's outbox and the desktop's received curves are
// kept between visits. A small synchronous key-value interface, so the logic
// can be tested with a Map and backed by localStorage in the pages (a curve is
// 1-3 KB; fifty of them are far below localStorage's limits).

export interface KeyValueStore {
  get(key: string): string | null;
  set(key: string, value: string): void;
  remove(key: string): void;
}

export class MemoryStore implements KeyValueStore {
  private readonly map = new Map<string, string>();
  get(key: string): string | null {
    return this.map.get(key) ?? null;
  }
  set(key: string, value: string): void {
    this.map.set(key, value);
  }
  remove(key: string): void {
    this.map.delete(key);
  }
}

/** localStorage, tolerating private modes where it throws: then nothing is
 * remembered, and the page still works for the visit. */
export class LocalStore implements KeyValueStore {
  constructor(private readonly prefix = "splinewire:") {}
  get(key: string): string | null {
    try {
      return localStorage.getItem(this.prefix + key);
    } catch {
      return null;
    }
  }
  set(key: string, value: string): void {
    try {
      localStorage.setItem(this.prefix + key, value);
    } catch {
      /* full or forbidden */
    }
  }
  remove(key: string): void {
    try {
      localStorage.removeItem(this.prefix + key);
    } catch {
      /* forbidden */
    }
  }
}

export function readJson<T>(store: KeyValueStore, key: string, fallback: T): T {
  const text = store.get(key);
  if (text === null) return fallback;
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}
