// The status line in the top bar: what the relay link sees, in words.

import type { LinkStatus } from "../relay/link";

export interface PhoneSeen {
  /** From the phone's hello; empty when a phone sent a curve without one. */
  device: string;
  app?: string;
}

export interface StatusLine {
  /** The lamp's colour: ok (a phone is there), wait (connected, no phone), bad (no relay), or off. */
  lamp: "ok" | "wait" | "bad" | "";
  text: string;
  /** "Trying again in 8 s": counts down every second, so the page keeps it out of the announced status. */
  countdown?: string;
  /** Offer "Retry now". */
  retry: boolean;
}

/** `waitLeftMs`: what is left of the link's retry wait, counted down by the page. */
export function statusLine(s: LinkStatus, phone: PhoneSeen | null, waitLeftMs = s.retryInMs ?? 0): StatusLine {
  switch (s.state) {
    case "open":
      if (s.peers < 1) return { lamp: "wait", text: "Connected · no phone open", retry: false };
      // A peer that has not said hello may be an older phone build, or a second
      // tab of this page; "phone" is right far more often than not.
      return { lamp: "ok", text: phone?.device ? `Phone connected · ${phone.device}` : "Phone connected", retry: false };
    case "waiting": {
      const secs = Math.max(1, Math.ceil(waitLeftMs / 1000));
      const why = s.problem ? s.problem[0]!.toUpperCase() + s.problem.slice(1) : "Can't reach the relay";
      return { lamp: "bad", text: why, countdown: `Trying again in ${secs} s`, retry: true };
    }
    case "connecting":
      return { lamp: "", text: "Connecting to the relay…", retry: false };
    default:
      return { lamp: "", text: "Not connected", retry: false };
  }
}
