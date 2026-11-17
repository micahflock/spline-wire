// "2 min ago" for the list, with the exact time in a tooltip. The time is the
// phone's clock at measuring; a phone a little ahead of this computer must not
// say "in 1 min", so anything in the future is "just now".

const MIN = 60_000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

export function relativeTime(then: number, now: number, locale?: string): string {
  const ago = now - then;
  if (ago < 45_000) return "just now";
  if (ago < HOUR) return `${Math.max(1, Math.round(ago / MIN))} min ago`;
  if (ago < DAY) return `${Math.floor(ago / HOUR)} h ago`;
  const days = Math.floor(ago / DAY);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  const d = new Date(then);
  const sameYear = d.getFullYear() === new Date(now).getFullYear();
  return d.toLocaleDateString(locale, sameYear ? { day: "numeric", month: "short" } : { day: "numeric", month: "short", year: "numeric" });
}

export function exactTime(then: number, locale?: string): string {
  return new Date(then).toLocaleString(locale, { dateStyle: "medium", timeStyle: "medium" });
}
