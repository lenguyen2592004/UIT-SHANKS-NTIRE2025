# UIT-SHANKS-NTIRE2025

This is the repository of team UIT-SHANKS for the NTIRE 2025 First Challenge on Day and Night Raindrop Removal for Dual-Focused Images.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Inference Pipeline](#inference-pipeline)
- [Training](#training)
- [Results](#results)
- [Installation](#installation)

## Introduction

This repository contains the code for our solution to the NTIRE 2025 Challenge on Day and Night Raindrop Removal. Our approach combines multiple models in a sequential pipeline to achieve high-quality raindrop removal. The pipeline consists of four main components:

1. **Raindrop-Pipeline**: A GAN-based approach for initial raindrop removal
2. **ResFusion**: A diffusion-based model for restoration
3. **Ensemble**: Combines outputs from multiple models
4. **Restormer**: Applied as a final denoising step

## Repository Structure

```
UIT-SHANKS-NTIRE2025/
├── inference/
│   ├── raindrop-pipeline-inference.ipynb
│   ├── resfusion-inference.ipynb
│   ├── ensemble.ipynb
│   ├── restormer-inference.ipynb
│   └── README.md (Detailed inference instructions)
├── train/
│   ├── raindrop-pipeline-train.ipynb
│   ├── raindrop-pipeline-train-with-pseudo-code.ipynb
│   ├── resfusion.ipynb
│   └── resfusion-with-pseudo-code.ipynb
├── model/ (Directory to store downloaded models)
│   ├── GAN_model/
│   ├── resfusion_model/
│   └── restormer_model/
└── private_dataset/ (Directory for test images)
```

## Models

Our pipeline uses three main models:

1. **GAN-based Model**: Adapted from Raindrop-Removal architecture
2. **ResFusion**: A diffusion-based model built on WeatherDiffusion framework
3. **Restormer**: A Transformer-based restoration model for final denoising

The pretrained models can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1bw8q4M2PM2yh9-KUF0yRvuY41ZE6Bxdy).

## Inference Pipeline

Our solution uses a pipeline approach where each model builds upon the results of the previous one:

1. First, the GAN-based model removes the majority of raindrops
2. Then, ResFusion enhances the results and handles more complex patterns
3. Next, outputs are combined using an ensemble approach
4. Finally, Restormer denoises the results for clean final images

For detailed inference instructions, please refer to the [inference README](inference/README.md).

### Quick Start for Inference

1. Download models from the [Google Drive link](https://drive.google.com/drive/folders/1bw8q4M2PM2yh9-KUF0yRvuY41ZE6Bxdy)
2. Create the proper folder structure for models and test data
3. Run the notebooks in sequence:
   - `raindrop-pipeline-inference.ipynb`
   - `resfusion-inference.ipynb`
   - `ensemble.ipynb`
   - `restormer-inference.ipynb`

## Training

Our training code is available in the `train` directory:

- `raindrop-pipeline-train.ipynb`: Training code for the GAN-based model
- `resfusion.ipynb`: Training code for the diffusion-based model

Notebooks with "-with-pseudo-code" suffix include detailed code explanations.

## Results

Our pipeline achieves high-quality raindrop removal on both day and night images with different focus settings. The sequential approach effectively removes raindrops while preserving image details and natural appearance.

## Installation

The following Python packages are required:

```bash
pip install torch torchvision opencv-python numpy pillow matplotlib tqdm albumentations pytorch_lightning tensorboard scikit-image
```

For GPU acceleration, ensure you have the appropriate CUDA drivers installed.

## Citation

If you use our code or models in your research, please cite:

```
@misc{uit-shanks-ntire2025,
  author = {UIT-SHANKS Team},
  title = {UIT-SHANKS Solution to NTIRE 2025 Challenge on Day and Night Raindrop Removal},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/UIT-SHANKS-NTIRE2025}}
}
```

## License

This project is licensed under the terms of the LICENSE file included in this repository.

## Acknowledgements

We thank the organizers of the NTIRE 2025 Challenge and acknowledge the contributions of the open-source projects that our solution builds upon.

- [Raindrop-Removal](https://github.com/Hyukju/Raindrop-Removal)
- [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion)
- [Restormer](https://github.com/swz30/Restormer)
