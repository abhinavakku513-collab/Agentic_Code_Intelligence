# Official FAQ (provided verbatim by the owner, 2026-09-20)

> The PPT mentions a Javascript codebase but the dataset is in Python. Will there be an additional dataset shared?

A. No. We would be using the APPS dataset (Python) as mentioned in the theme guideline document throughout the hackathon. Please ignore the Javascript codebase mentioned in the PPT.

> Can I use XYZ method during retrieval/pre/post processing?

A. We have no restrictions on specific methods to be used during any stage. However, please keep in mind that as the PPT and theme guidelines document indicates: the resource usage (running time, GPU requirement, model size, etc.) would be factored in during the evaluation of a submission.

---
Consequences (see CLAUDE.md): Python/APPS only, JS/TS out of scope (D14); any method allowed, so a search-level model object is a legitimate method (D8); running time, GPU requirement and model size are scored, weights unpublished, so every release candidate ships a resource scorecard and encoder choice follows the "smallest within 1.0 NDCG@10 point" rule (D2, D4).
