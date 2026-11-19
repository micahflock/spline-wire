import { describe, expect, it } from "vitest";
import { browserInfo, uaData } from "./browser";

const UA = {
  safari18: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
  // iOS 26: the OS in the string stays at 18_6; Safari's version follows the release.
  safari26: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1",
  inApp: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
  chromeIos: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/142.0.7444.46 Mobile/15E148 Safari/604.1",
  chromeAndroid: "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36",
  samsung: "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/28.0 Chrome/130.0.0.0 Mobile Safari/537.36",
  firefoxAndroid: "Mozilla/5.0 (Android 16; Mobile; rv:146.0) Gecko/146.0 Firefox/146.0",
  macSafari: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
  windowsEdge: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.3595.53",
};

describe("browser and OS for the camera view's diagnostics", () => {
  it("names Safari on an iPhone, taking iOS 26 and later from Safari's version", () => {
    expect(browserInfo(UA.safari18)).toBe("Safari 18.1 · iOS 18.1");
    expect(browserInfo(UA.safari26)).toBe("Safari 26.1 · iOS 26.1");
  });

  it("names other iOS browsers, and an app's web view as such", () => {
    expect(browserInfo(UA.chromeIos)).toBe("Chrome (iOS) 142 · iOS 18.6");
    expect(browserInfo(UA.inApp)).toBe("WebKit (in-app browser) · iOS 18.6");
  });

  it("takes Chrome on Android's real version and model from userAgentData, not its reduced string", () => {
    expect(browserInfo(UA.chromeAndroid)).toBe("Chrome 142 · Android");          // "Android 10; K" is not the phone's
    expect(browserInfo(UA.chromeAndroid, {
      platform: "Android", platformVersion: "16.0.0", model: "Pixel 7 Pro",
      fullVersionList: [{ brand: "Not_A Brand", version: "99.0.0.0" }, { brand: "Google Chrome", version: "142.0.7444.59" }],
    })).toBe("Chrome 142.0.7444.59 · Android 16.0.0 · Pixel 7 Pro");
    expect(browserInfo(UA.samsung, { platform: "Android", platformVersion: "15.0.0", model: "SM-S918B" })).toBe("Samsung Internet 28 · Android 15.0.0 · SM-S918B");
  });

  it("names Firefox on Android, whose string keeps the Android version", () => {
    expect(browserInfo(UA.firefoxAndroid)).toBe("Firefox 146 · Android 16");
    // A phone emulated on a computer: userAgentData describes the computer, so the string's version stands.
    const emulated = "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36";
    expect(browserInfo(emulated, { platform: "Linux", platformVersion: "6.18.0", model: "" })).toBe("Chrome 142 · Android 14");
  });

  it("names desktop browsers", () => {
    expect(browserInfo(UA.macSafari)).toBe("Safari 26.1 · macOS");
    expect(browserInfo(UA.windowsEdge)).toBe("Edge 142 · Windows");
    expect(browserInfo("")).toBe("browser unknown");
  });

  it("asks Chromium for the high-entropy values, and gives null elsewhere or on refusal", async () => {
    let asked: string[] = [];
    const nav = { userAgentData: { getHighEntropyValues: async (h: string[]) => ((asked = h), { model: "Pixel 7", platformVersion: "16.0.0" }) } };
    expect(await uaData(nav)).toEqual({ model: "Pixel 7", platformVersion: "16.0.0" });
    expect(asked).toEqual(["platformVersion", "model", "fullVersionList"]);
    expect(await uaData({})).toBeNull();
    expect(await uaData({ userAgentData: { getHighEntropyValues: async () => { throw new Error("no"); } } })).toBeNull();
  });
});
