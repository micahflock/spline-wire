import { describe, expect, it } from "vitest";
import { phonePageUrl, relayAddress } from "./config";
import { deviceName } from "./device";

describe("where the relay is", () => {
  const base = { search: "", hostname: "192.168.1.20", dev: false };
  it("takes ?relay= first, then the build's address, then the dev relay on this host", () => {
    expect(relayAddress({ ...base, search: "?relay=ws%3A%2F%2F127.0.0.1%3A9000%2F", built: "wss://r.example" })).toBe("ws://127.0.0.1:9000");
    expect(relayAddress({ ...base, built: "wss://r.example/" })).toBe("wss://r.example");
    expect(relayAddress({ ...base, dev: true })).toBe("ws://192.168.1.20:8787");
    expect(relayAddress({ ...base, hostname: "[::1]", dev: true })).toBe("ws://[::1]:8787");
  });

  it("has none in a production build made without one, and ignores what is not a relay address", () => {
    expect(relayAddress(base)).toBeNull();
    expect(relayAddress({ ...base, search: "?relay=https://evil.example", built: "" })).toBeNull();
    expect(relayAddress({ ...base, built: "javascript:alert(1)", dev: true })).toBe("ws://192.168.1.20:8787");
  });

  it("points the pairing link at phone/ under the site, carrying an override along", () => {
    expect(phonePageUrl("https://u.github.io/spline-wire/", "")).toBe("https://u.github.io/spline-wire/phone/");
    expect(phonePageUrl("http://localhost:5173/", "?relay=ws://127.0.0.1:9000&x=1")).toBe(
      "http://localhost:5173/phone/?relay=ws%3A%2F%2F127.0.0.1%3A9000",
    );
    expect(phonePageUrl("http://localhost:5173/", "?relay=http://x")).toBe("http://localhost:5173/phone/");
  });
});

describe("device names for the hello message", () => {
  it.each([
    ["Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1", "iPhone, Safari"],
    ["Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/131.0 Mobile/15E148 Safari/604.1", "iPhone, Chrome"],
    ["Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36", "Android, Chrome"],
    ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0", "Windows, Edge"],
    ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) Gecko/20100101 Firefox/132.0", "Mac, Firefox"],
    ["", "a browser"],
  ])("%s", (ua, name) => {
    expect(deviceName(ua)).toBe(name);
  });
});
