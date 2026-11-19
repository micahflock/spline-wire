// The browser, its version, the OS and its version, and the phone's model where
// the browser tells it: one line for the camera view's diagnostics, so that a
// screenshot sent by someone else says what it came from.
//
// The user-agent string is not enough everywhere:
//   - Chrome on Android sends a reduced one, "Android 10; K" whatever the phone
//     and the release; navigator.userAgentData gives the real Android version
//     and the model when asked (getHighEntropyValues), which camera-view.ts does
//     once and passes here.
//   - Safari on iOS 26 and later freezes the OS in the string at 18_6; its own
//     "Version/" follows the iOS release (Safari 26 ships with iOS 26), so that
//     is used for the OS too.
//   - Other iOS browsers (Chrome, Firefox, Edge, apps like Quiche) are WebKit
//     inside; the string may name them (CriOS, FxiOS, EdgiOS) or not.
// relay/device.ts deviceName is the pairing label, without versions; this is
// diagnostics only, shown to the person and nowhere else.

/** What navigator.userAgentData gives (Chromium only), high-entropy values included when asked. */
export interface UAData {
  brands?: { brand: string; version: string }[];
  fullVersionList?: { brand: string; version: string }[];
  platform?: string;
  platformVersion?: string;
  model?: string;
  mobile?: boolean;
}

export function browserInfo(ua: string, data?: UAData | null): string {
  const model = data?.platform === "Android" ? data.model?.trim() ?? "" : "";
  return [browserOf(ua, data), osOf(ua, data), model].filter(Boolean).join(" · ") || "browser unknown";
}

function browserOf(ua: string, data?: UAData | null): string {
  const v = (re: RegExp) => ua.match(re)?.[1] ?? "";
  const major = (s: string) => s.split(".")[0] ?? s;
  // Order matters: Edge, Opera and Samsung say "Chrome" too, and every iOS browser says "Safari".
  const named: [RegExp, string][] = [
    [/EdgiOS\/([\d.]+)/, "Edge (iOS)"], [/CriOS\/([\d.]+)/, "Chrome (iOS)"], [/FxiOS\/([\d.]+)/, "Firefox (iOS)"],
    [/EdgA?\/([\d.]+)/, "Edge"], [/OPR\/([\d.]+)/, "Opera"], [/SamsungBrowser\/([\d.]+)/, "Samsung Internet"],
    [/Firefox\/([\d.]+)/, "Firefox"],
  ];
  for (const [re, name] of named) if (re.test(ua)) return `${name} ${major(v(re))}`;
  if (/Chrome\/[\d.]+/.test(ua)) {
    // The reduced string says Chrome/142.0.0.0; the full version list says 142.0.7444.59.
    const full = data?.fullVersionList?.find((b) => /Google Chrome|Chromium/.test(b.brand))?.version;
    return `Chrome ${full ?? major(v(/Chrome\/([\d.]+)/))}`;
  }
  if (/iPhone|iPad|iPod/.test(ua)) {
    const version = v(/Version\/([\d.]+)/);
    return version && /Safari\//.test(ua) ? `Safari ${version}` : "WebKit (in-app browser)";
  }
  if (/Safari\//.test(ua) && /Version\/([\d.]+)/.test(ua)) return `Safari ${v(/Version\/([\d.]+)/)}`;
  return "";
}

function osOf(ua: string, data?: UAData | null): string {
  if (/iPhone|iPad|iPod/.test(ua)) {
    const fromUa = (ua.match(/OS (\d+(?:_\d+)*) like Mac OS X/)?.[1] ?? "").replace(/_/g, ".");
    // Safari's version follows the iOS release; the OS in the string stops at 18.6 (iOS 26 and later).
    const safari = /Safari\//.test(ua) && !/CriOS|FxiOS|EdgiOS/.test(ua) ? ua.match(/Version\/([\d.]+)/)?.[1] : undefined;
    const version = safari && Number(safari.split(".")[0]) >= 26 ? safari : fromUa;
    return `${/iPad/.test(ua) ? "iPadOS" : "iOS"}${version ? ` ${version}` : ""}`;
  }
  if (/Android/.test(ua)) {
    // Chrome's reduced string always says "Android 10; K": take the real version from userAgentData.
    const reduced = /Android 10; K\b/.test(ua);
    // (Only when userAgentData says Android: an emulated phone's describes the computer.)
    const fromData = data?.platform === "Android" ? data.platformVersion : undefined;
    const version = fromData || (reduced ? "" : ua.match(/Android ([\d.]+)/)?.[1] ?? "");
    return `Android${version ? ` ${version}` : ""}`;
  }
  if (/Windows NT/.test(ua)) return "Windows";
  if (/CrOS/.test(ua)) return "ChromeOS";
  if (/Mac OS X|Macintosh/.test(ua)) return "macOS";
  if (/Linux/.test(ua)) return "Linux";
  return data?.platform ?? "";
}

/** The high-entropy values Chromium gives when asked (the real Android version, the
 * model, the full browser version); null elsewhere or if it will not say. */
export async function uaData(nav: { userAgentData?: { getHighEntropyValues?(hints: string[]): Promise<UAData> } } = navigator as never): Promise<UAData | null> {
  try {
    return (await nav.userAgentData?.getHighEntropyValues?.(["platformVersion", "model", "fullVersionList"])) ?? null;
  } catch {
    return null;
  }
}
