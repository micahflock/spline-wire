// Chain presets as data/chain-snap.yaml and data/chain.yaml give them (public/chains.json), for the drawings' tests.

import type { ChainLook } from "./chain-art";

export const SNAP: ChainLook = { pitch_mm: 13, half_width_mm: 4.39, fiducial: "ring", fiducial_mm: 8.79, ring_inner_mm: 7.07, contact: "round" };
export const ORING: ChainLook = { pitch_mm: 10, half_width_mm: 4, fiducial: "dot", fiducial_mm: 5, ring_inner_mm: 0, contact: "edge" };
