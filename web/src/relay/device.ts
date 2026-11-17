// A short name for this device, sent in the hello message so the other page
// can say who is there ("Phone connected: iPhone, Safari"). Only for people to
// read: nothing depends on it.

export function deviceName(userAgent: string): string {
  const ua = userAgent;
  const os =
    /iPhone/.test(ua) ? "iPhone"
    : /iPad/.test(ua) ? "iPad"
    : /Android/.test(ua) ? "Android"       // the model is hidden by today's browsers ("K")
    : /Windows/.test(ua) ? "Windows"
    : /Mac OS X|Macintosh/.test(ua) ? "Mac"
    : /CrOS/.test(ua) ? "ChromeOS"
    : /Linux/.test(ua) ? "Linux"
    : "";
  // Order matters: Edge and Opera say "Chrome" too, and every iOS browser says "Safari".
  const browser =
    /EdgA?\/|Edg\//.test(ua) ? "Edge"
    : /OPR\//.test(ua) ? "Opera"
    : /SamsungBrowser\//.test(ua) ? "Samsung Internet"
    : /Firefox\/|FxiOS\//.test(ua) ? "Firefox"
    : /CriOS\/|Chrome\//.test(ua) ? "Chrome"
    : /Safari\//.test(ua) ? "Safari"
    : "";
  const name = [os, browser].filter(Boolean).join(", ");
  return (name || "a browser").slice(0, 80);
}
