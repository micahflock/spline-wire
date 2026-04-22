# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Curve Capture

## Overview

Capture arbitrary real-world curves — pipe profiles, molding, doorknob silhouettes, anything awkward for a ruler or calipers — and drop them into CAD as reference geometry. The user wraps a stiff, self-holding planar chain around the target curve, removes it (the chain retains its shape), photographs it with a phone, and the software stack converts the photo into XY points (and ideally an approximating curve) inside CAD.

Primary CAD target: **Autodesk Fusion**. Minimize user interaction between "measure with chain" and "points in CAD." Scope boundaries, deferred work, and explicit non-goals live in `docs/architecture.md` and `docs/next-steps.md`.

## Hardware concept

Think of a bike chain, roughly a hand span long (~5 in / ~125 mm), with enough friction at every pin that each link holds its angle when posed by hand. Wrap it around a curve, lift it off, carry it to the camera — the shape stays. Each link carries a fiducial; at least one end carries a distinct **skew fiducial** so the vision step can rectify the photo.

## Repo layout

- `CLAUDE.md` — this file. Project overview and pointers.
- `docs/architecture.md` — first-pass system architecture (hardware → capture → CV → CAD).
- `docs/open-questions.md` — tracked gaps and unresolved design decisions.
- `docs/next-steps.md` — concrete, near-term tasks to validate feasibility before building the full stack, in risk order. Also the de facto status doc — items get checked off as they're validated.

## Priority

The computer vision step is the riskiest. Building a stiff chain is mechanical work with known solutions; moving points into CAD is an integration problem with known APIs. Fiducial design + LLM vision accuracy is the unknown that gates everything else. **Validate it first** — see `docs/next-steps.md`, item 1.
