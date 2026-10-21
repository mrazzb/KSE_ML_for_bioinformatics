# Practical 7 public materials

This folder contains the notebook and small public examples for image classification, YOLO detection, Cellpose instance segmentation, 3D microscopy inspection and VGG13D-style embeddings.

The full detection training annotations and trained model weights are intentionally kept outside the GitHub copy. One YOLO image/label pair (`604.jpg` and `604.txt`) is included so students can inspect the annotation format. The `detection/test` images are unlabeled examples for prediction and uncertainty review.

The notebook can still be read and executed through the inspection and demonstration sections. Training a detector requires placing the private labelled dataset under `detection/colonies_annotation` locally. Model files are also expected to remain local and are ignored by Git.

The 3D examples are multi-page 16-bit TIFF volumes. The notebook discovers all TIFF volumes under `3d`, provides a static preview, and opens Napari only when `RUN_NAPARI_VIEWER = True`. VGG13D embedding extraction is opt-in through `RUN_3D_EMBEDDINGS = True` and `RUN_EXPENSIVE = True`.
