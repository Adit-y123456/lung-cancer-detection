# Results

## Original research abstract

The earlier research abstract reported more than 90% overall classification accuracy and a peak accuracy of 94.12% for large cell carcinoma. It also described evaluation using confusion matrices and IoU scores.

Those statements refer to the earlier study. This repository contains a reconstructed implementation and has not yet reproduced those numbers. This implementation measures image classification metrics; IoU would require ground-truth region annotations and additional evaluation code.

## New experiments

No new training or evaluation has been run. After training, record actual test metrics from `results/cnn_metrics.json` and identify the dataset and split used.

## Needed for reproduction

- Dataset source and class counts
- Original patient-level train/test split, if available
- Model parameters, image preprocessing, and training settings
- Old confusion matrices and saved checkpoints, if recoverable
- Ground truth regions for any IoU evaluation
