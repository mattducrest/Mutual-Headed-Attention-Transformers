# Mutual-Headed Attention Transformers

PyTorch reference implementation of a mutual-headed attention architecture for
multimodal skin-lesion classification. The model combines dermoscopic images
with structured patient and lesion metadata, using the ISIC 2024 Challenge as
its example application.

The accompanying notebook explains the architecture step by step and is the
best place to start:

- [`mutual-headed-attention-notebook.ipynb`](mutual-headed-attention-notebook.ipynb)
- [Building a Mutual Attention Model from Scratch with PyTorch](https://medium.com/@mattducrest/building-a-mutual-attention-model-from-scratch-with-pytorch-7d0e07778032)

## Architecture

The model has four main components:

1. A pretrained Vision Transformer encodes dermoscopic images.
2. A fully connected network encodes structured metadata.
3. Mutual-headed attention lets each modality attend to features from the
   other modality.
4. A final classifier predicts the probability of a malignant lesion.

The attention block exchanges query information between the image and metadata
representations, then combines the attended features through residual
connections and concatenation.

## Repository layout

```text
.
├── main.py
├── mutual-headed-attention-notebook.ipynb
├── nn/
│   ├── data_preprocessing.py
│   └── model.py
├── train/
│   └── train.py
└── requirements.txt
```

- `main.py` assembles the preprocessing, model, and training pipeline.
- `nn/data_preprocessing.py` provides HDF5/JPEG data loaders and image
  augmentations.
- `nn/model.py` defines the image encoder, metadata encoder, mutual-attention
  block, and classifier.
- `train/train.py` contains the training and validation loops.

## Installation

Python 3.10 or a compatible environment is recommended.

```bash
git clone https://github.com/mattducrest/Mutual-Headed-Attention-Transformers.git
cd Mutual-Headed-Attention-Transformers
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The example uses the [ISIC 2024 Challenge dataset](https://www.kaggle.com/competitions/isic-2024-challenge/data),
which includes dermoscopic images, metadata, and binary lesion labels. The data
are not redistributed in this repository.

The training entry point is configured for Kaggle paths:

```text
/kaggle/input/isic-2024-challenge/train-metadata.csv
/kaggle/input/isic-2024-challenge/train-image.hdf5
```

To run elsewhere, update those paths in `main.py` and configure the Vision
Transformer checkpoint in `nn/model.py` for your environment.

## Running the example

Once the data and model paths are configured:

```bash
python main.py
```

The script selects CUDA when available, then Apple Metal Performance Shaders,
and otherwise falls back to CPU. It trains the multimodal classifier and records
loss, accuracy, and recall for the training and validation splits.

## Scope

This repository is an educational and research prototype. It has not been
validated for clinical use and must not be used to make medical decisions.

