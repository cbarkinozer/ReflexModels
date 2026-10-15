# Package the data, upload privately, and link Reflex-1

The easiest useful dataset release is the **781 admitted phone training examples**, with the general training sources linked separately. Requested name: `cbarkinozer/MOBILE_ROUTING9`. This avoids presenting our phone data as the entire combined training mixture or relicensing BANKING77/CLINC150 as Apache-2.0.

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
  --output-dir data/prepared/hf-mobile-routing9
cp releases/reflex-1/dataset/README.template.md \
  data/prepared/hf-mobile-routing9/README.md
```

The packager checks exact source SHA, requires 781 rows and matching route schemas, rejects duplicate or protected IDs, and writes train JSONL plus a hash/count manifest. It excludes all non-train splits, reviewer identifiers and UI notes. Inspect the resulting text for secrets or accidentally real personal information; keep template provenance accurate. Choose your dataset name/license and update the card placeholders. For original synthetic phone examples, CC-BY-4.0 or Apache-2.0 are possible author choices if you have the rights; this guide does not assign one. Add LICENSE text and metadata only after deciding.

Package layout:

```
hf-mobile-routing9/
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
repo = "cbarkinozer/MOBILE_ROUTING9"
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
# Inspect an existing repo's visibility yourself before uploading: exist_ok does not change it.
api.upload_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path="data/prepared/hf-mobile-routing9",
    commit_message="Add reviewed phone training dataset and provenance",
)
```

Confirm the repo is private, the viewer shows 781 train rows and all nine routes, and the card/manifest are correct. Verify loading it with your authenticated account:

```python
from datasets import load_dataset
rows = load_dataset("cbarkinozer/MOBILE_ROUTING9", split="train", token=True)
assert len(rows) == 781
```

You can also upload the same prepared folder through the dataset repo’s **Files → Add file → Upload files** page instead of the Python upload call. Keep the `data/` subdirectory and check the rendered card/viewer afterwards. On Windows, `Get-FileHash -Algorithm SHA256 <archive>` is the PowerShell equivalent of the archive hash check above.

## 4. Link model and dataset both ways

In `cbarkinozer/Reflex-1` model README YAML add the **actual existing dataset ID**, keeping existing metadata:

```yaml
datasets:
- cbarkinozer/MOBILE_ROUTING9
- LocalLLaMA/typed-decisions
- PolyAI/banking77
- clinc/clinc_oos
```

In prose specify the first is the 781-row phone component, the other three are general sources, and list exact revisions/counts and supervised in-domain evaluation limits. Link to `https://huggingface.co/datasets/cbarkinozer/MOBILE_ROUTING9`. Put the model link `https://huggingface.co/cbarkinozer/Reflex-1` in the dataset card. Hugging Face documents model-card `datasets` metadata [here](https://huggingface.co/docs/hub/model-cards). A Markdown link alone is useful but does not replace structured dataset metadata.

## 5. Public release later

Once you review the measured results/cards and decide to release, change each repo's visibility in Settings. Check links while logged out, and ensure license, source attribution, schema and example API request are present. Making one repo public does not make the other public. The model card, aggregate benchmark evidence, generic schema/examples and efficiency figures have been updated in the existing private model repository. Neither visibility nor dataset publication has changed; see `hf-model-card-update-mobile-routing9-20261015.json` for the verified model-card update receipt.

If you want the **exact combined training mixture** public too, create a separately documented config/repo from the executed mix IDs and pinned upstream sources. Preserve the 391/195/195 general source sampling and 781 phone exposure, per-source licenses/attribution, and exact hashes. An independently resampled 781-row general pool is not exact reproduction. Do not upload benchmark questions as training examples or imply these three general benchmarks were zero-shot tests.

## Updating the model card before public release

The measured card source is `releases/reflex-1/HF_MODEL_CARD.md`, and the current card is already synced to the private model repo's **README.md**. After you publish the exact phone training dataset, add its final ID/link in both places. `releases/reflex-1/benchmark-summary.json` contains aggregate metrics, confusion matrices and calibration/resource summaries without held-out request text, reviewer IDs or device-property dumps. It can accompany the model card. Check `complete: true` for the revised four-job mobile comparison scope. `original_expanded_queue_complete` remains false while the ten beta/historical/general Android jobs are deferred; do not describe those suites as completed on the phone. Existing GPU general benchmark results must retain their hardware labels.

To update the existing private model repository yourself after review:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="releases/reflex-1/HF_MODEL_CARD.md",
    path_in_repo="README.md", repo_id="cbarkinozer/Reflex-1",
    commit_message="Document measured mobile benchmarks and training data",
)
api.upload_file(
    path_or_fileobj="releases/reflex-1/benchmark-summary.json",
    path_in_repo="benchmark-summary.json", repo_id="cbarkinozer/Reflex-1",
    commit_message="Add aggregate benchmark evidence",
)
```

These writes do not themselves change an existing repository's privacy. Do not upload raw evaluation folders or the entire Git working tree as a model or dataset release. Keep the selected adapter/head/tokenizer/configs and GGUF files already present in the model repo; updating README is not a model retrain.

Also copy `releases/reflex-1/phone-routing-schema.json` and `releases/reflex-1/decision-request.example.json` into the model repository root when updating the card. They contain the generic decision schema and an ordinary example request, not test questions or gold labels. Upload each with `api.upload_file(path_or_fileobj=..., path_in_repo=the_filename, repo_id="cbarkinozer/Reflex-1")` or the web Files uploader. These generic example files are already uploaded and SHA-256 verified with the private model-card update.
