import { describe, expect, it } from "vitest";
import { pairingUrl } from "../relay/pairing";
import { codeMeaning } from "./qr";

const HERE = "https://example.github.io/spline-wire/phone/";
const SECRET = "UkypDe-UM7GtIrxg48MQHw";

describe("codeMeaning", () => {
  it("takes the secret from this page's pairing code", () => {
    expect(codeMeaning(pairingUrl(HERE, SECRET), HERE)).toEqual({ kind: "pair", secret: SECRET });
    // The page open with a relay setting or a pairing already in its address.
    expect(codeMeaning(pairingUrl(HERE, SECRET), `${HERE}?relay=wss%3A%2F%2Fr.example#x`)).toEqual({ kind: "pair", secret: SECRET });
  });

  it("does not use a code made by another copy of the site", () => {
    expect(codeMeaning(pairingUrl("https://evil.example/spline-wire/phone/", SECRET), HERE))
      .toEqual({ kind: "elsewhere", where: "evil.example/spline-wire/phone/" });
    expect(codeMeaning(pairingUrl("https://example.github.io/other/phone/", SECRET), HERE).kind).toBe("elsewhere");
  });

  it("ignores codes that are not pairing codes", () => {
    for (const text of ["hello", "", HERE, `${HERE}#pair=short`, "https://example.com/#pair=", "WIFI:S:x;;"]) {
      expect(codeMeaning(text, HERE), text).toEqual({ kind: "other" });
    }
  });
});
