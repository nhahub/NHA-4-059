# Scope note: why we are not reusing jacobkauffmann/unsupervised-ch

During planning we found a public repo (`jacobkauffmann/unsupervised-ch`,
tag `v1.0.0`) that looked related at first glance — it also uses CLIP, the
ImageNet truck classes, and a garbage-truck logo finding. It implements the
paper **"The Clever Hans Effect in Unsupervised Learning"** (Kauffmann et
al., now published in *Nature Machine Intelligence*, 2025).

It does **not** match our project's scope:

| | This project | That paper's repo |
|---|---|---|
| Core question | Does CLIP zero-shot classification use real semantics or shortcuts? | Does CLIP's *representation* encode logos in a way that contaminates any downstream classifier? |
| CLIP's role | Doing the actual zero-shot classification we evaluate | Feeding embeddings into a separately trained linear classifier; CLIP is never doing zero-shot classification in their experiments |
| Primary XAI method | Grad-CAM | BiLRP (similarity-decomposition LRP) |
| Truck classes | 8 classes incl. **beer truck** | 8 classes incl. **police van** (no beer truck) |
| Deliverable | Interactive dashboard | Paper reproduction code, no dashboard |

We're keeping this note so nobody mid-sprint goes "let's just copy their
code" and quietly drifts the whole project into a different research
question with a different deliverable. If you want to cite their finding
(CLIP relying on garbage-truck logos) as related work / motivation in the
report, that's fair and useful — just don't import their method as our
method.
