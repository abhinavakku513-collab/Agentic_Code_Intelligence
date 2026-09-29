ACIS — Agentic Code Intelligence

Natural-language query in, ranked code snippets out, with the exact source as evidence.

ACIS (Agentic Code Intelligence) is a retrieval system for large codebases. Given a natural-language query and a library of code, it ranks code snippets by relevance so that an agent or developer can reach the relevant implementation faster.

Built for Samsung PRISM GenAI Hackathon — Theme 1: Agentic Code Intelligence.

1. Project Scope

The Theme 1 problem is fundamentally code retrieval: given a library of code and a natural-language query, provide a ranking of code snippets in order of relevance.

Goal

Scope

P0 — Retrieval Accuracy

Retrieve relevant code snippets for a natural-language query.

P1 — Retrieval Across Versions

Retrieve from different versions of a changing codebase and rebuild/update indexes efficiently.

Bonus — Evolutionary Retrieval

Retrieve across all versions while handling near-identical code revisions and their lineage.

Out of scope

Generation, explanation by an LLM, or answering the user's question with generated code is not part of the scored retrieval path.

2. System Overview

Natural-language query
        │
        ▼
Query preprocessing / routing
        │
        ├──────────────► Dense retrieval
        │
        ├──────────────► BM25 / lexical retrieval
        │
        └──────────────► Query/code features
                            │
                            ▼
                    Fusion / learned ranking
                            │
                            ▼
                  Ranked code snippets
                            │
                            ▼
                  Exact source evidence

Version-aware retrieval:

Repository
   │
   ├── Version v1 ──► indexed units
   ├── Version v2 ──► changed / reused units
   ├── Version v3 ──► changed / reused units
   │
   └── Lineage ─────► evolution across versions

3. Implemented Scope

P0 — Retrieval Engine

Implemented components include:

Query preprocessing and route selection

Model-specific query formatting

Dense embeddings

Dense similarity retrieval

BM25 / lexical retrieval

Retrieval features

Candidate union

Ranking / fusion

Tail handling

Confidence information

Evaluation and run ledgers

UI/API access to the same retrieval engine

The selected real encoder is:

Alibaba-NLP/gte-modernbert-base

The project records model metadata and pinning information used by the evaluation workflow.

P1 — Retrieval Across Versions

Implemented:

Immutable snapshots

Content-addressed reuse

Version isolation

Incremental updates

Changed/added/deleted/renamed unit detection

Reuse of unchanged embeddings

Embedding only new/changed units

Snapshot validation

Atomic activation

Rollback

Crash-safe temporary snapshot creation

Manifest/hash validation

Bonus — Evolutionary Retrieval

Implemented:

Lineage recovery

Grouping related revisions

Revision timelines

Best-revision information

Grouped retrieval

Flat all-version retrieval

Duplicate analysis across versions

Measured lineage evidence includes pairwise precision/recall/F1 of 1.0 on the evaluated lineage set.

The current grouped-vs-flat Evolution-NDCG result is approximately tied:

Grouped Evolution-NDCG@10: 0.80146
Flat Evolution-NDCG@10:    0.80225
Delta:                      -0.00079 (~ -0.08 percentage points)

Therefore this README does not claim that lineage grouping improves the ranking metric. Its demonstrated benefit is lineage recovery and duplicate suppression.

4. Repository Architecture

acis/
├── core/
├── data/
├── prep/
├── embed/
├── lexical/
├── features/
├── rank/
├── engine/
├── store/
├── ingest/
├── lineage/
├── agent/
├── api/
├── cli/
├── eval/
├── mteb_adapter/
└── obs/
    └── sec/

configs/
├── dev.yaml
├── official.yaml
└── models/

docs/
├── spec/
├── PHASE2_REPORT.md
└── ...

runs/
├── ledger.jsonl
├── model_radar.json
└── ...

scripts/
└── bench/

5. Requirements

Reference development environment:

Linux / WSL2

Python 3.12

Git

GNU Make

uv

CPU execution supported

GPU optional

The scored query path does not depend on an external LLM.

6. Clean-Machine Setup — Every Command

This section records the complete setup workflow, including the environment/setup issues encountered during preparation.

6.1 Enter the repository

After cloning:

cd ~/Agentic_Code_Intelligence
pwd
git status
git remote -v
git branch -a

If the repository has not yet been cloned:

git clone <REPOSITORY_URL> Agentic_Code_Intelligence
cd Agentic_Code_Intelligence

6.2 Install Git and Make

Ubuntu/WSL:

sudo apt update
sudo apt install git make

Verify:

git --version
make --version

6.3 Initialize/check Git

If working with an existing cloned repository, do not run git init again. Check it instead:

git status

If starting a repository that genuinely has no .git directory:

git init

6.4 Install uv

The project uses uv for Python environment/dependency management.

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version

If uv is still not found, reopen the terminal and run:

uv --version

6.5 Set up Python dependencies

make setup

Verify the project CLI:

uv run acis --help

6.6 Hardware/environment check

make doctor

This records the detected hardware profile under:

runs/hardware.json

6.7 Fetch APPS/CoIR assets

make fetch

This is the dataset preparation/network step used by the evaluation workflow.

6.8 Fetch the real model

The selected real encoder is:

Alibaba-NLP/gte-modernbert-base

Intended command:

make fetch-models MODELS=gte-modernbert-base

The model is pinned using its revision and file hashes.

7. Important Clean-Clone Issue: Missing Model Card

A clean checkout currently needs the pinned model card at:

configs/models/gte-modernbert-base.yaml

If it is missing, the model registry intentionally refuses to guess the model and the following command fails:

make fetch-models MODELS=gte-modernbert-base

with an error of the form:

acis: invalid_input: no model card for 'gte-modernbert-base'

make demo or search then fails with:

NotReady: no model card for 'gte-modernbert-base'

This was an actual clean-machine setup issue encountered during repository preparation.

Do not replace the real encoder with acis/hashing-4096 to hide this problem. The hashing encoder was a development/infrastructure stand-in and is not the final real encoder.

Before a clean-clone judge/demo release, the correct pinned GTE model card must be committed under configs/models/ and must match the model metadata used by the recorded evaluation.

To diagnose the repository:

ls configs/models/
git log --all --oneline -- configs/models
git ls-tree -r HEAD --name-only | grep 'configs/models'

8. Basic Search

After model setup:

uv run acis search "find the shortest path in a weighted graph"

Another example:

uv run acis search "reverse a linked list"

The results are ranked code units/snippets with retrieval evidence.

9. Demo

Run:

make demo

The demo covers:

P0 retrieval

P1 versioned retrieval

Bonus evolutionary retrieval

Local web UI

UI:

http://127.0.0.1:8000/

For a clean CPU machine, the demo workflow can use the supplied release artifact:

dist/acis-demo-index.zip

This is a demonstration convenience only. The official evaluation is cold and must not depend on a prebuilt demo cache.

10. API

Start the server:

uv run acis serve

Check the UI/API resources:

for p in / /app.css /app.js /favicon.ico; do
  curl -s -o /dev/null -w "$p %{http_code}\n" http://127.0.0.1:8000$p
done

Expected response: HTTP 200 for the available resources.

Search

curl -s -X POST localhost:8000/v1/search \
  -H 'content-type: application/json' \
  -d '{"query":"reverse a linked list","top_k":3,"explain":true}'

Version update demonstration

curl -s -X POST localhost:8000/v1/repos/apps-history/commit \
  -H 'content-type: application/json' \
  -d '{"edits":3}'

Evolution

curl -s -X POST localhost:8000/v1/evolve \
  -H 'content-type: application/json' \
  -d '{"query":"sort an array","repo_id":"apps-history","top_k":5}'

11. CLI Operations

Implemented project workflow commands include:

aci index --repo <repo> --from <git|dir|zip|jsonl> --rev <revision>
aci search "<query>"
aci versions
aci activate <version>
aci rollback
aci validate
aci gc
aci serve

Evaluation commands:

aci eval p0
aci eval p1
aci eval bonus

The repository's current installed entry point is invoked in the documented setup examples as uv run acis ....

12. P0 Evaluation

The official P0 evaluation is based on CoIR AppsRetrieval through MTEB.

Primary metrics:

NDCG@10

MRR

The development workflow evaluates the development split before any held-out test evaluation.

Official configuration:

configs/official.yaml

Reproduction:

make reproduce

Verify a run:

uv run acis eval verify-submission --run-dir runs/<run_id>

The official run is intended to be cold so that stale cached embeddings cannot silently replace the evaluation work.

13. Recorded P0 Evidence

The model-selection bake-off was performed over the full 5,000-query development set.

Recorded GTE result:

Model:       Alibaba-NLP/gte-modernbert-base
Dev queries: 5,000
NDCG@10:     0.7103456795
MRR@10:      0.6745552381
R@100:       0.9378
Parameters:  149,014,272
Run ID:      gate-055152620da6

A clean-tokenization G5 result was also recorded:

NDCG@10:     0.723515
MRR@10:      0.689393
R@100:       0.9276
Run ID:      gate-bffed3d40ab2

These are development measurements, not held-out test claims.

14. P0 Retrieval Pipeline

The retrieval pipeline contains:

route
  ↓
query preparation
  ↓
encode
  ↓
dense retrieval
  ↓
BM25 / lexical retrieval
  ↓
candidate union
  ↓
features
  ↓
ranking / fusion
  ↓
tail handling
  ↓
final composition
  ↓
confidence

Query preparation supports:

Query normalization

Model-specific task formatting

Statement-like vs generic routing

Multiple query views

Dense truncation configuration

Lexical normalization

The full document remains available to BM25/features while dense representation limits apply to the embedding path.

15. Model Selection

The encoder was selected using a full development-set bake-off rather than a single hand-picked query.

Selected real model:

Alibaba-NLP/gte-modernbert-base

16. P1 — Versioned Retrieval

P1 supports changing repositories through immutable versioned snapshots.

For a new version the system detects:

unchanged units

changed units

added units

deleted units

renamed units

Unchanged representations can be reused. New/changed units are embedded as required.

Snapshot creation is validated before activation and activation is atomic. Rollback returns the active state to a previous valid version.

Temporary snapshot directories use a .tmp-<token> workflow so incomplete builds are not exposed as valid snapshots.

17. Bonus — Evolutionary Retrieval

The Bonus supports retrieval across versions.

Flat

Every matching revision is represented separately.

Grouped by lineage

Related revisions are grouped so that a logical unit is represented once, with its timeline and best revision available.

Measured lineage evidence:

Pairwise precision: 1.0
Pairwise recall:    1.0
Pairwise F1:        1.0

Duplicate-rate@10 evaluation:

Grouped: 0.00
Flat:    0.77

The current grouped Evolution-NDCG result does not establish a ranking-quality improvement, so the project does not claim one.

18. Testing and Evaluation Discipline

The project contains tests for retrieval contracts, model registry behavior, indexing, snapshots, version isolation, incremental updates, rollback, crash recovery, lineage, evaluation, and UI/API integration.

Development evaluation follows:

Implement change
      ↓
Tests
      ↓
DEV evaluation
      ↓
Ledger entry
      ↓
Inspect metrics/failures
      ↓
Only then consider held-out TEST

Do not tune against held-out test labels.

Do not fake, estimate, or manually invent accuracy numbers. If a result has not been measured, report it as Not measured.

19. Clean-Machine Troubleshooting

make setup fails because uv is missing

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
make setup

APPS assets are missing

make fetch

make fetch-models MODELS=gte-modernbert-base fails

Error:

acis: invalid_input: no model card for 'gte-modernbert-base'

Check:

ls configs/models/

Required:

configs/models/gte-modernbert-base.yaml

Do not silently substitute another model.

make demo fails with NotReady

If the preceding message says the GTE model card is missing, fix the model-card/repository configuration first and rerun:

make demo

Search fails before model setup

Example:

uv run acis search "find the shortest path in a weighted graph"

If the error is the missing GTE model card, complete the model setup first.

20. Reproducibility

A reproducible run records:

Model name and revision

Model file fingerprints

Preparation/configuration fingerprint

Dataset and split

Code/config revision

NDCG@10

MRR

Query count

Runtime where measured

Run ID

Ledger entry

Hardware profile

Run artifacts are kept under runs/ and accuracy claims are tied to ledgered measurements.

21. Judge / Demo Checklist

On a fully configured machine:

make doctor
make fetch
make fetch-models MODELS=gte-modernbert-base
uv run acis search "find the shortest path in a weighted graph"
make demo

Open:

http://127.0.0.1:8000/

Demonstrate:

Natural-language query → ranked code snippets

Retrieval evidence and stage information

P1 version selection

Incremental version update

Rollback

Bonus grouped lineage retrieval

Flat all-version retrieval

Duplicate suppression / lineage evidence

ACIS — Agentic Code Intelligence

Samsung PRISM GenAI Hackathon — Theme 1

Core objective: high-quality code retrieval from natural-language queries, with version-aware and lineage-aware retrieval extensions.