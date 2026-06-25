# Multi-Fidelity Neural Networks for 3D RBC Reconstruction

> **Reconstruction of three-dimensional shapes of normal and disease-related erythrocytes from partial observations using multi-fidelity neural networks**  
> Haizhou Wen, He Li, Zhen Li  
> [arXiv:2511.14962](https://arxiv.org/abs/2511.14962)

This repository contains the code, processed data, and manuscript assets for reconstructing three-dimensional red blood cell (RBC) shapes from partial observations using multi-fidelity neural networks (MFNNs).

The work combines:

- simulation-based low-fidelity and high-fidelity RBC datasets generated from dissipative particle dynamics (DPD),
- experimental or image-derived partial cross-sections extracted from 2D RBC images, and
- TensorFlow MFNN models that reconstruct full 3D RBC surfaces from one, two, or three cross-sectional views.

## Repository At a Glance

At the top level, the repository currently has three main pieces:

```text
.
|-- DPDdata/
|-- ExpImages/                     # described in this README as ExpImages
`-- Multifidelity_neural_networks_for_three_dimensional_reconstruction_of_health_and_aged_erythrocyte.pdf
```

- `DPDdata/` contains the simulation-driven benchmark cases described in the manuscript.
- `ExpImages/` contains the image-based reconstruction workflows. In this README, it is referred to as `ExpImages/`, since that is the intended project-facing name.
- The PDF is the manuscript describing the scientific motivation, model design, and evaluation.

## Scientific Scope

The repository covers RBC shapes across several morphology classes:

- `D`: discocyte
- `S`: stomatocyte
- `E`: echinocyte

The manuscript positions the method as a way to recover full 3D RBC geometry from partial observations such as microscope images or orthogonal cross-sections. The MFNN architecture uses a low-fidelity branch to learn broad morphology and a high-fidelity branch to adapt to a smaller target dataset, with surface area and volume constraints used during training.

## Main Workflows

There are two complementary workflows in this repository.

### 1. Simulation Benchmark Workflow: `DPDdata/`

`DPDdata/` is the main simulation benchmark collection. Its own [README](./DPDdata/README.md) already documents the experimental families and naming conventions in detail.

The current case folders include:

- `D.D.(1)`
- `D.S`
- `E.E.(1)`
- `E.E.(3)`
- `E.E.(5)`
- `E.E.(7)`
- `E.E.(8)`
- `S.S.(2)`

Each case contains MFNN runs under different observation configurations, typically:

- `rbc*2CS_yz+xz`
- `rbc*2CS_xz+xy`
- `rbc*2CS_yz+xy`
- `rbc*3CS`

Within each subcase, the recurring file pattern is:

```text
<subcase>/
|-- data/                    # LF/HF .npz datasets
|-- result/                  # predictions and diagnostic plots
|-- datapre.py               # preprocessing and dataset loading
|-- mf_rbcnn_main1.py        # training entry point
|-- NN.py                    # MFNN model definition
`-- output_record_mfS_1.txt  # training log
```

This part of the repository is best understood as the controlled simulation study reported in the manuscript.

### 2. Image-Based Reconstruction Workflow: `ExpImages/` (`ExpImages/`)

`ExpImages/` is the image-driven side of the project. In documentation terms, this folder is better read as `ExpImages/`.

It currently contains four morphology-specific cases:

- `Disco`
- `Stomato`
- `Echino1`
- `Echino2`

Each case is organized into two stages:

```text
<Case>/
|-- ImageExtraction/
`-- MFNN/
```

#### `ImageExtraction/`

This stage converts an RBC image into MFNN-ready partial section data. The current structure is:

```text
ImageExtraction/
|-- In_image/                # input RBC image
|-- In_DPDref/               # DPD reference partial sections
|-- Out_graph/               # extraction diagnostics and plots
|-- Out_data4MFNN/           # generated .npz files for MFNN input
`-- ImageExtraction_*.py     # extraction/alignment script
```

From the current scripts, this stage does the following:

- segments the RBC from the image,
- extracts one or more characteristic cross-sectional curves,
- aligns those extracted curves to DPD reference sections,
- packages the aligned partial sections into `.npz` files,
- writes summary plots showing the extracted loops and alignment quality.

Representative outputs include files such as:

- `D_exp_partial_yz.npz`
- `D_exp_partial_xz.npz`
- `D_exp_partial_xz+yz.npz`
- `S_exp_partial_yz.npz`
- `E1_exp_partial_yz.npz`
- `E2_exp_partial_yz.npz`

#### `MFNN/`

This stage consumes the extracted partial sections and reconstructs a full 3D RBC surface using the trained MFNN setup.

The current `datapre.py` files show the pattern clearly:

- low-fidelity reference data are loaded from `data/*.npz`,
- full high-fidelity reference data are loaded from `data/*.npz`,
- image-derived partial sections from `ImageExtraction/Out_data4MFNN/` are copied into the local `data/` folder and used as the high-fidelity training input.

The recurring files are:

```text
MFNN/<run>/
|-- data/
|-- result/
|-- datapre.py
|-- mf_rbcnn_main1.py
|-- NN.py
|-- loss_plot.py
|-- loss_plot_single.py
`-- output_record_mfS_1.txt
```

The `mf_rbcnn_main1.py` scripts currently implement the same two-stage training pattern described in the manuscript:

1. Adam optimization
2. SGD with Nesterov momentum

and save predictions to:

- `result/mfS_predict.npz`

with reconstruction plots in:

- `result/x-y.png`
- `result/x-z.png`
- `result/y-z.png`

## Current Case Coverage in `ExpImages/`

The image-based reconstruction side currently contains:

- `Disco`
  - image extraction plus MFNN runs for `yz`, `xz`, and combined `xz+yz` partial sections
- `Stomato`
  - image extraction plus MFNN runs for `yz`, `xz`, and combined `xz+yz` partial sections
- `Echino1`
  - image extraction plus an MFNN run for a `yz`-based partial section
- `Echino2`
  - image extraction plus an MFNN run for a `yz`-based partial section

So in practice, `DPDdata/` is the broader simulation benchmark suite, while `ExpImages/` (`ExpImages/`) is the more focused image-to-reconstruction workflow.

## How the Pieces Fit Together

The repository currently supports the following end-to-end logic:

1. Start from simulated RBC datasets in `DPDdata/` or from an image case in `ExpImages/`.
2. For image-based cases, run `ImageExtraction_*.py` to convert the image into MFNN-ready partial cross-sections.
3. Place or use the generated `.npz` partial-section files in the corresponding `MFNN/<run>/data/` folder.
4. Run `mf_rbcnn_main1.py` to reconstruct the full 3D RBC surface.
5. Inspect `result/` plots and `mfS_predict.npz` for reconstruction quality.

## Notes on Naming

- This README uses `ExpImages` as the descriptive name for `ExpImages/`, per the intended rename.
- `DPDdata/README.md` remains the most detailed reference for the simulation benchmark naming system.
- Local folders such as `__pycache__/` and `.claude/` are development artifacts rather than core scientific outputs.

## Recommended Starting Points

If you are new to this repository, the best reading order is:

1. the manuscript PDF at the repository root,
2. [DPDdata/README.md](./DPDdata/README.md),
3. one image-based case under `ExpImages/` such as:
   - `ExpImages/Disco/ImageExtraction/`
   - `ExpImages/Disco/MFNN/`

That sequence gives the clearest picture of the project: simulation benchmark first, then image extraction, then MFNN-based 3D reconstruction.

## Requirements

- Python 3.12.x or higher version 
- TensorFlow 2.18.0 or higher version  
- cuDNN 8.6 (NVIDIA RTX4090) or depending on local GPU
- ...

## Citation

```bibtex
@article{wen2025mfnn,
  title   = {Reconstruction of three-dimensional shapes of normal and disease-related erythrocytes from partial observations using multi-fidelity neural networks},
  author  = {Wen, Haizhou and Li, He and Li, Zhen},
  journal = {arXiv preprint arXiv:2511.14962},
  year    = {2025}
}
```
