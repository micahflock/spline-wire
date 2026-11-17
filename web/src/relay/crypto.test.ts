import { describe, expect, it } from "vitest";
import { deriveKeys, isSecret, newSecret, open, seal } from "./crypto";
import { ROOM_RE } from "../../relay/rules";

describe("pairing crypto", () => {
  it("derives the same room and key from the same secret, and a routable room id", async () => {
    const secret = newSecret();
    expect(isSecret(secret)).toBe(true);
    const a = await deriveKeys(secret);
    const b = await deriveKeys(secret);
    expect(a.room).toBe(b.room);
    expect(a.room).toMatch(ROOM_RE);
    expect(a.room).not.toBe(secret);
    expect(await open(b, await seal(a, "phone", { hello: [1, 2.5, "é"] }))).toEqual({
      sender: "phone",
      body: { hello: [1, 2.5, "é"] },
    });
  });

  it("is stable: a known secret gives a known room (a change here breaks existing pairings)", async () => {
    expect((await deriveKeys("AAAAAAAAAAAAAAAAAAAAAA")).room).toBe("sqwKiXBRZAWPHjtU2lraHA");
  });

  it("refuses frames from another pairing, tampered frames and a forged sender", async () => {
    const mine = await deriveKeys(newSecret());
    const other = await deriveKeys(newSecret());
    const frame = await seal(mine, "phone", { n: 1 });
    expect(await open(other, frame)).toBeNull();
    const f = JSON.parse(frame);
    expect(await open(mine, JSON.stringify({ ...f, s: "desktop" }))).toBeNull();
    expect(await open(mine, JSON.stringify({ ...f, c: f.c.slice(0, -2) + "AA" }))).toBeNull();
    expect(await open(mine, JSON.stringify({ ...f, v: 2 }))).toBeNull();
    for (const junk of ["", "{", "null", "[]", '{"v":1}', '{"v":1,"s":"phone","n":"!!","c":"!!"}']) {
      expect(await open(mine, junk)).toBeNull();
    }
  });

  it("refuses a malformed secret", async () => {
    await expect(deriveKeys("short")).rejects.toThrow("not a pairing secret");
  });
});
