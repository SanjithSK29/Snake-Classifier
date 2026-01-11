**Snake Classification — Notebook Summary**

This repository contains a Jupyter notebook for training and evaluating a binary snake image classifier (Venomous vs Non_Venomous) using a pretrained backbone from `timm`. The notebook, dataset layout, training pipeline, cross-validation, and error-analysis steps are documented below.

**Repository Structure**
- **Notebook:** [Class.ipynb](Class.ipynb) — main notebook with full pipeline and analysis.
- **Models:** `best_model_fold_0.pth` ... `best_model_fold_4.pth` — saved weights for each CV fold.
- **Dataset:** `Snake_Images/` — expected layout:
  - `Snake_Images/train/Venomous/` — images labeled Venomous
  - `Snake_Images/train/Non_Venomous/` — images labeled Non_Venomous
  - `Snake_Images/test/` — (optional) images for testing

**Project Overview**
- **Purpose:** Train a binary image classifier to distinguish venomous and non-venomous snakes using transfer learning (`resnet34` via `timm`).
- **Approach:** Load images with OpenCV, augment with `albumentations`, create a PyTorch `Dataset`/`DataLoader`, train with cross-validation (StratifiedKFold, 5 folds), save best models per fold, and run error analysis (confusion matrix, misclassified image inspection).

**Key Notebook Components (high-level walkthrough)**
- **Imports & Utilities:** Uses `torch`, `timm`, `albumentations`, `cv2`, `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `tqdm`.
- **Dataframe Creation:** Walks `Snake_Images/train/` to build a `DataFrame` with columns `file_name` and `label` (0/1 mapping to classes).
- **Visualization:** Sample images are read via OpenCV, converted to RGB, resized to `256x256`, and plotted.
- **Dataset Class:** `CustomDataset` wraps reading, `albumentations` transforms (default: `Resize(cfg.image_size, cfg.image_size)` + `ToTensorV2()`), and returns `(image, label)`. Images are normalized by dividing by `255` after `ToTensorV2()`.
- **Data Split:** Uses `train_test_split` and then sets up `StratifiedKFold` with `cfg.n_folds = 5`. A `kfold` column is added to `df`.
- **Transforms:** Soft augmentations named `transform_soft` include `Resize`, `Rotate`, `HorizontalFlip`, and `CoarseDropout`.
- **Model:** Created via `timm.create_model(cfg.backbone, pretrained=True, num_classes=cfg.n_classes)`. Default `cfg.backbone = 'resnet34'`, `cfg.n_classes = 2`.
- **Loss / Optimizer / Scheduler:** `CrossEntropyLoss`, `Adam` optimizer (`lr=1e-4` by default), and `CosineAnnealingLR` scheduler with `eta_min=cfg.lr_min`.
- **Training Loop:** `train_one_epoch` and `validate_one_epoch` functions compute loss, predictions, and metric (accuracy). `fit()` runs epochs, tracks best validation accuracy, and returns training history and the best model state.
- **Cross-Validation Loop:** For each fold the notebook creates dataloaders with `transform_soft` for training, trains the model with `fit()`, visualizes history, and saves the best model as `best_model_fold_{fold}.pth`.
- **Error Analysis:** Includes code to retrain (or reuse) a fold's model, gather predictions and probabilities on the validation set, build and plot a confusion matrix, print classification report, identify misclassified images, and display them grouped by false positives/negatives. It also produces summary statistics for misclassifications.

**Files of Interest**
- Notebook: [Class.ipynb](Class.ipynb)
- Trained weights (per fold): [best_model_fold_0.pth](best_model_fold_0.pth), [best_model_fold_1.pth](best_model_fold_1.pth), [best_model_fold_2.pth](best_model_fold_2.pth), [best_model_fold_3.pth](best_model_fold_3.pth), [best_model_fold_4.pth](best_model_fold_4.pth)

**Environment & Requirements**
- Tested with Python 3.8+ and the following packages (minimum):
  - `torch` (PyTorch) — GPU recommended when available
  - `timm`
  - `albumentations`
  - `opencv-python` (cv2)
  - `numpy`, `pandas`, `scikit-learn`
  - `matplotlib`, `seaborn`
  - `tqdm`

Install example (use a virtual environment):
```
pip install torch timm albumentations opencv-python numpy pandas scikit-learn matplotlib seaborn tqdm
```

**Configuration (cfg)**
- `cfg.root_dir` — root dataset path used by `CustomDataset` (default: `Snake_Images/train/`).
- `cfg.image_size` — image resize (default: `256`).
- `cfg.batch_size` — training batch size (default: `32`).
- `cfg.learning_rate`, `cfg.epochs`, `cfg.lr_min` — optimizer & scheduler params.
- `cfg.backbone` — backbone used by `timm` (default: `resnet34`).
- `cfg.n_folds` — number of cross-validation folds (default: `5`).

**How to run**
1. Prepare dataset in the layout shown above under `Snake_Images/train/`.
2. Open the notebook [Class.ipynb](Class.ipynb) in Jupyter or VS Code and run cells sequentially. The notebook is designed to be executed cell-by-cell (it builds objects incrementally).
3. To train full CV models, run the cross-validation loop near the end of the notebook. Models are saved as `best_model_fold_{fold}.pth`.

**Quick inference example**
```
import torch
import cv2
import timm
import numpy as np

cfg_backbone = 'resnet34'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = timm.create_model(cfg_backbone, pretrained=False, num_classes=2)
model.load_state_dict(torch.load('best_model_fold_0.pth', map_location=device))
model.to(device).eval()

img = cv2.imread('Snake_Images/test/some_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256,256)) / 255.0
img = np.transpose(img, (2,0,1)).astype('float32')
tensor = torch.tensor(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
pred = int(np.argmax(probs))
print('Predicted class:', pred, 'Probs:', probs)
```

**Notes, Tips & Caveats**
- The notebook normalizes images by dividing by `255` after `ToTensorV2()`; you may prefer using `albumentations.Normalize()` + `ToTensorV2()` instead.
- For reproducibility the notebook sets `cfg.seed` and applies deterministic PyTorch flags; this may impact performance on some CUDA setups.
- If training is slow or memory-bound, reduce `cfg.batch_size` or `cfg.image_size`.
- The current scheduler `T_max` uses an estimate based on dataset size; you may tune it for smoother schedules.
- The error analysis cells retrain a fold for analysis if you don't have saved models — to save time, load existing `best_model_fold_{i}.pth` files instead.

**Next steps / Suggestions**
- Add a `requirements.txt` or `environment.yml` for exact reproducibility.
- Provide a small sample of images under `Snake_Images/test/` for quick inference testing.
- Consider adding model evaluation metrics logging (e.g., TensorBoard or Weights & Biases) for better tracking.

**License & Contact**
- This repository contains research/experiment code. Add or change a license as appropriate for your use.
