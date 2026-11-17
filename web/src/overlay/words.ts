// What the camera view says, in a few words a person can act on while holding
// the phone. One line at a time: the coach puts the most important problem
// first, and fixing it often fixes the rest.

import type { CoachState, Problem } from "./types";

export interface Say {
  /** Large, a few words. */
  title: string;
  /** Smaller, optional: why, or how. */
  detail: string;
}

export const PROBLEM_WORDS: Record<Problem, Say> = {
  "no-chain": { title: "Point the camera at the chain", detail: "Lay it flat, dots facing up." },
  "pins-missing": { title: "Part of the chain is out of view", detail: "Or glare or a shadow hides some dots." },
  "at-edge": { title: "Keep the chain away from the edges", detail: "Move back a little, or center it." },
  "too-far": { title: "Move closer", detail: "Let the chain fill more of the view." },
  "tilted": { title: "Tilt the phone flatter over the chain", detail: "Look straight down on it." },
  "moving": { title: "Hold still", detail: "" },
  "blurred": { title: "Out of focus", detail: "Move back a little, or wait for it to focus." },
};

export interface SayContext {
  /** The measuring engine has loaded: before that nothing is analysed. */
  engineReady: boolean;
}

/** The message for a coach state. */
export function sayFor(state: CoachState, ctx: SayContext): Say {
  if (!ctx.engineReady) return { title: "Getting the measuring engine ready…", detail: "You can take the photo with the button meanwhile." };
  switch (state.phase) {
    case "capture":
    case "captured":
      return { title: "Taking the photo…", detail: "Keep still for a moment." };
    case "steady":
      return { title: "Hold still…", detail: state.straighter ? "Straighter over the chain is better still." : "" };
    case "searching":
      return PROBLEM_WORDS["no-chain"];
    case "adjusting":
      // A problem is shown only once it has lasted (coach.ts PROBLEM_MS): until then, nothing new to say.
      return state.problems[0] ? PROBLEM_WORDS[state.problems[0]] : { title: "Hold still…", detail: "" };
  }
}

/** "11 of 13 pins", or nothing before a chain is seen. */
export function pinsText(state: CoachState): string {
  if (state.phase === "searching" || state.pinsFound === 0) return "";
  return `${state.pinsFound} of ${state.pinsExpected} pins`;
}
