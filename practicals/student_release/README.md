# Student-facing MOA competition data

This release contains a compound-level train/test split for multiclass MOA prediction.

- training compounds: 212
- test compounds: 54
- training samples: 1272
- test samples: 324
- classes: 8
- doses per compound: 6

## Files

- `train.csv`: sample metadata, dose, compound ID, and SMILES with the training target.
- `test.csv`: the same feature fields, but **without the target column**.
- `expression_train.parquet`, `expression_test.parquet`: L1000 expression profiles keyed by `sample_id`.
- `image_manifest_train.csv`, `image_manifest_test.csv`: five-channel Cell Painting image references keyed by `sample_id`.
- `Multimodal_MOA_Submission_Template.ipynb`: a minimal expression-only baseline showing validation, prediction export, and model saving for leaderboard submission.

Image paths in the manifests are relative to the original package's `multimodal_moa_data/` directory. The image bytes are not duplicated by this split step.

## Rules

1. Keep the rows in `test.csv` in the same order when generating predictions.
2. Submit one predicted class per `sample_id`.
3. Do not use names, Broad IDs, source MOA labels, or any private metadata.
4. The six doses of one compound are kept in the same partition to prevent compound leakage.
5. The test target is held by the instructor and must not be reconstructed from the package.

Expected prediction format:

```csv
sample_id,predicted_class
S000043,DNA damage / replication
```
