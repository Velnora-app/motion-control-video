# TheoreticallyPose V4 — Quick Guide

Turn any video clip into **control videos** (pose skeleton, depth map, silhouettes) for
video-to-video AI generation — Seedance, Kling, Runway, whatever you're driving.
Everything runs **in your browser, on your machine**. Your video never gets uploaded
anywhere.

## What you need

- **Chrome or Edge** on a reasonably modern computer (the depth engines use your GPU
  via WebGPU — Safari and Firefox may only run the skeleton/silhouette parts)
- A video clip (a few seconds is the sweet spot; mp4 works great)
- That's it. No account, no install, no subscription. It's one HTML file.

## Quick start (60 seconds)

1. **Double-click `theoreticallypose.html`** — it opens in your browser
2. **Drag your clip** onto the left panel (or click to browse)
3. Hit **Track** — it finds the person and builds the skeleton (~30s for a short clip)
4. Hit **Bake** in the Depth panel — it computes a depth map for every frame
5. **Export MP4** (or PNG sequence) — that's your control video. Feed it to your
   AI video tool as the driving/reference video.

First time you Bake, the depth model downloads automatically (45–95 MB depending on
engine, with a progress bar). It's cached after that — future bakes start instantly.

## The layers (mix and match)

Think of it as three stackable passes — each with its own on/off and opacity:

- **Skeleton** — the pose stick figure. "OpenPose" colors are what most AI models
  were trained on; use those unless you have a reason not to.
- **Silhouette** — solid body shapes from the video. "Tint" style colors each person.
- **Depth** — grayscale depth of the whole scene (bright = near, dark = far).

**Presets** are one-click mixes: *Stick figure*, *OpenPose*, **Full Stack** (all three
layers — the flagship look), *Silhouette*, *Depth only*. Watch the status bar at the
bottom — every preset click tells you exactly what it switched.

## The two depth engines

In the Depth panel there's an **engine** dropdown:

- **Fast (per-frame)** — quick, but each frame is computed independently, so the
  depth can shimmer/flicker over time. The *stabilize* and *smooth* controls clean
  most of that up.
- **Consistent (VDA)** — a temporal model that looks at 32 frames at once. Depth
  comes out **stable straight away** — no flicker to fix — and it's actually *faster*
  than Fast mode on most GPUs. Needs WebGPU (Chrome/Edge). If your browser can't run
  it, the app tells you and Fast mode still works.

If your machine supports it, **use Consistent.** It's the whole reason V4 exists.

## Multiple people

1. Tick **Multiple characters** and pick how many (2–5)
2. Hit **Track** again — changing the character count needs a re-track
   (re-tracking replaces the previous track completely)
3. Each character gets a color — you'll see numbered skeletons in the overlay.
   **Click a character's color swatch to pick your own** — it updates the skeleton
   tints, tinted silhouettes, and colored depth everywhere at once
4. Bonus: in the Depth panel, tick **color chars** to tint each person's region of
   the depth map with their character color (the depth shading is preserved —
   near people stay bright, far people stay dark)

**Tip:** multi-person tracking works best when people are reasonably large in frame
(roughly a third of the frame height or more). Tiny background figures may not track.

## Working with a range

Set **In** / **Out** points (I / O keys) to work on just a section — tracking,
baking, and exports all respect the range. **Full** resets to the whole clip.

## Exporting

- **Export MP4** — the composited control video, ready to drop into your AI tool
- **PNG seq** — a zip of lossless frames, for compositing or picky pipelines
- **JSON** — the raw pose keypoints, if you want to do something custom

## If something's weird

- **"Consistent engine needs WebGPU"** — use Chrome or Edge, current version. On an
  old GPU, use the Fast engine instead.
- **Depth model won't download** — check your internet; the models come from
  HuggingFace on first use. After that it works offline.
- **A person isn't tracking** — they're probably too small in frame, or heavily
  cut off. Try a tighter crop of the clip.
- **Depth looks inverted** (near is dark) — tick **invert** in the Depth panel.
- **Changed character count and things look stale** — just hit Track again.

---

*TheoreticallyPose by [Theoretically Media](https://www.youtube.com/@TheoreticallyMedia).
Depth models: Depth Anything V2 + Video Depth Anything (Small, Apache-2.0) by ByteDance,
served from HuggingFace. Pose: MediaPipe. Everything processes locally in your browser.*
