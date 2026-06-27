# 10-Day Plan (quick reference)

Full version with per-person checklists: see the shared `10-Day-Build-Plan.docx`.
This file is the fast in-repo lookup.

## Roles
| ID | Role | Owns |
|---|---|---|
| A | ML / CLIP Lead | Zero-shot inference, embeddings, accuracy/F1 |
| B | XAI / Grad-CAM Lead | Grad-CAM (+ optional LRP), heatmaps |
| C | Data Engineer | Dataset sourcing, logo insertion, annotation |
| D | Quant Analysis Lead | Spurious-feature scoring, accuracy-delta, stats |
| E | Dashboard Lead | Plotly Dash app, Vercel deploy |
| F | PM / Integration / Docs | Timeline, pipeline integration, report |

## Day-by-day

- **Day 1** — Setup. **Critical: Person B resolves ViT vs RN50 Grad-CAM feasibility by EOD** (`src/xai/gradcam_feasibility_spike.py`). Person C downloads raw data. Person A smoke-tests CLIP.
- **Day 2** — Data cleaning + logo-insertion script (C). Full CLIP inference on original set (A). Grad-CAM pipeline built (B). Dashboard skeleton (E). Metrics framework (D).
- **Day 3** — CLIP inference on logo variant (A). Grad-CAM on original set (B). Heatmap energy-ratio scorer (D). Tab 1 wired live (E). Logo bounding-box annotation (C).
- **Day 4** — Grad-CAM on logo variant (B). Accuracy-delta analysis (D). Tab 2: image browser (E). **Midpoint go/no-go on the hypothesis** (A, D, F).
- **Day 5** — Buffer / optional LRP stretch (B). Tab 3: comparison view (E). Methodology draft (F). Cross-class ranking (D).
- **Day 6** — Visual polish (B, E). Pipeline integration script (A, F). Vercel deploy config (E). Stats sanity checks (D).
- **Day 7** — Bug bash, full pipeline run by a non-builder (All). Dashboard copy finalized (E, F). Results draft (D, F).
- **Day 8** — Bug fixes + polish (E, B). Full report draft (F). Demo script/recording (F, E).
- **Day 9** — Rehearsal (All). Production deploy check (E). Claims-vs-results QA pass (D, A).
- **Day 10** — Final packaging and submission (All).

## Non-negotiable scope guardrails
- CLIP zero-shot + Grad-CAM + Dash dashboard. Not BiLRP, not a paper reproduction.
- SimCLR/BarlowTwins comparison: cut by default, revisit only if ahead of schedule on Day 4.
- See `docs/scope_note.md` for why the `jacobkauffmann/unsupervised-ch` repo is not our template.
