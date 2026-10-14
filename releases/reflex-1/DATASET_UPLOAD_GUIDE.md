# Package the data, upload privately, and link Reflex-1

The easiest useful dataset release is the **781 admitted phone training examples**, with the general training sources linked separately. Suggested name (not yet chosen): `cbarkinozer/reflex-phone-routing-v1`. This avoids presenting our phone data as the entire combined training mixture or relicensing BANKING77/CLINC150 as Apache-2.0.

No new synthetic generation is needed for the model we already trained. Upload the exact admitted training rows, not the original 900-row generator output, a fresh 80/20 reshuffle, or the Argilla UI snapshot. There were 1,080 reviewed original rows, but the comparison-ready pipeline kept 781 train /90 validation /90 calibration /88 historical test after quarantining conflicts and fixing validation IDs. The combined model used **781 phone plus 781 sampled general** rows. Scenario categories are metadata, not prediction labels.

## 1. Recover exact executed training inputs

On the Windows laptop the verified backup is under:

`C:/Users/barkin/Documents/GitHub/pocket-agents-lab/runs/decision-followups-vast-20261013/`

Archive: `reflex-decision-artifacts-20261013.tar.gz`; expected SHA-256:

`2f4fd5a651880e8504013ac525f6f204f63690de5e04cc42702d397cda0f73be`

List its members before extracting, then recover the executed `data/prepared/phone-routing-human-reviewed-v2-1080-fixedval-20261013/`, `data/prepared/general-routing-v1-balanced/`, executed preparation/training scripts, and `runs/combined-balanced-scratch-120step-s3407/run-manifest.json`. Some archives have a leading `reflex-decision/` directory. Use the executed run manifest's SHA for the phone train file. The Mac's old companion training script predates laptop changes: do not regenerate the exact final split or mixture with that older script.

```sh
shasum -a 256 reflex-decision-artifacts-20261013.tar.gz
tar -tzf reflex-decision-artifacts-20261013.tar.gz
# Extract selected members into a private working directory after inspecting the list.
```

The current Mac has verified model artifacts from the private model repo, but not that full training-data archive. The Git review snapshot records annotations and is not automatically the exact training split. Do not label a reconstructed export as exact reproduction without matching its source SHA/IDs against the executed run.

## 2. Create a local train-only upload folder

From ReflexModels:

```sh
python3 tools/prepare_phone_dataset_release.py \
  --input-dir /private/path/phone-routing-human-reviewed-v2-1080-fixedval-20261013 \
  --expected-train-sha256 SHA_FROM_EXECUTED_RUN_MANIFEST \
  --output-dir data/prepared/hf-reflex-phone-routing-v1
cp releases/reflex-1/dataset/README.template.md \
  data/prepared/hf-reflex-phone-routing-v1/README.md
```

The packager checks exact source SHA, requires 781 rows and matching route schemas, rejects duplicate or protected IDs, and writes train JSONL plus a hash/count manifest. It excludes all non-train splits, reviewer identifiers and UI notes. Inspect the resulting text for secrets or accidentally real personal information; keep template provenance accurate. Choose your dataset name/license and update the card placeholders. For original synthetic phone examples, CC-BY-4.0 or Apache-2.0 are possible author choices if you have the rights; this guide does not assign one. Add LICENSE text and metadata only after deciding.

Package layout:

```
hf-reflex-phone-routing-v1/
  README.md
  LICENSE
  manifest.json
  data/train.jsonl
```

Do not include `.env`, Argilla credentials/snapshots, model weights, training archives, or any test JSONL. Keep the frozen evaluation sets separate. If you later deliberately publish an evaluation set, label it as an evaluation/regression benchmark and preserve a new unseen cohort for later versions.

## 3. Upload privately when you decide to do so

These commands are instructions for you; the benchmark work does not execute uploads. Hugging Face supports README dataset cards and explicit split file configuration. [Dataset-card documentation](https://huggingface.co/docs/hub/datasets-cards), [dataset uploading](https://huggingface.co/docs/datasets/upload_dataset).

Install `huggingface_hub`, then run `hf auth login` in your terminal so no token appears in a script or Git. Use a write-enabled token. Create a **private Dataset** repository in the web UI, or use:

```python
from huggingface_hub import HfApi
api = HfApi()  # uses your saved login
repo = "cbarkinozer/YOUR_FINAL_DATASET_NAME"
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
# Inspect an existing repo's visibility yourself before uploading: exist_ok does not change it.
api.upload_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path="data/prepared/hf-reflex-phone-routing-v1",
    commit_message="Add reviewed phone training dataset and provenance",
)
```

Confirm the repo is private, the viewer shows 781 train rows and all nine routes, and the card/manifest are correct. Verify loading it with your authenticated account:

```python
from datasets import load_dataset
rows = load_dataset("cbarkinozer/YOUR_FINAL_DATASET_NAME", split="train", token=True)
assert len(rows) == 781
```

## 4. Link model and dataset both ways

In `cbarkinozer/Reflex-1` model README YAML add the **actual existing dataset ID**, keeping existing metadata:

```yaml
datasets:
- cbarkinozer/YOUR_FINAL_DATASET_NAME
- LocalLLaMA/typed-decisions
- PolyAI/banking77
- clinc/clinc_oos
```

In prose specify the first is the 781-row phone component, the other three are general sources, and list exact revisions/counts and supervised in-domain evaluation limits. Link to `https://huggingface.co/datasets/cbarkinozer/YOUR_FINAL_DATASET_NAME`. Put the model link `https://huggingface.co/cbarkinozer/Reflex-1` in the dataset card. Hugging Face documents model-card `datasets` metadata [here](https://huggingface.co/docs/hub/model-cards). A Markdown link alone is useful but does not replace structured dataset metadata.

## 5. Public release later

Once you review the measured results/cards and decide to release, change each repo's visibility in Settings. Check links while logged out, and ensure license, source attribution, schema and example API request are present. Making one repo public does not make the other public. We have not changed either visibility or uploaded new files during these benchmarks.

If you want the **exact combined training mixture** public too, create a separately documented config/repo from the executed mix IDs and pinned upstream sources. Preserve the 391/195/195 general source sampling and 781 phone exposure, per-source licenses/attribution, and exact hashes. An independently resampled 781-row general pool is not exact reproduction. Do not upload benchmark questions as training examples or imply these three general benchmarks were zero-shot tests.
