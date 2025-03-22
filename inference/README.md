# UIT-SHANKS-NTIRE2025 Inference Pipeline

This README provides step-by-step instructions for running the UIT-SHANKS raindrop removal inference pipeline.

## Overview

The pipeline consists of four main models:
1. **Raindrop-Pipeline**: Based on Raindrop-Removal GAN architecture
2. **ResFusion**: A diffusion-based approach for raindrop removal
3. **Ensemble**: Combines outputs from multiple models
4. **Restormer**: Applied as a final denoising step

## Prerequisites

The notebooks require the following Python packages:
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- PIL/Pillow
- matplotlib
- tqdm
- albumentations
- pytorch_lightning
- tensorboard
- scikit-image

You can install them with:

```bash
pip install torch torchvision opencv-python numpy pillow matplotlib tqdm albumentations pytorch_lightning tensorboard scikit-image
```

## Step 1: Download Models

1. Download the pre-trained models from Google Drive:
   - [Link to models](https://drive.google.com/drive/folders/1bw8q4M2PM2yh9-KUF0yRvuY41ZE6Bxdy)

2. Create the following folder structure for the models:
   ```
   UIT-SHANKS-NTIRE2025/
   ├── model/
   │   ├── GAN_model/
   │   │   └── model_epoch800.pth.pth  # GAN model
   │   ├── resfusion_model/
   │   │   └── lightning_logs/
   │   │       └── version_0/
   │   │           └── checkpoints/
   │   │               └── best-epoch=199-val_SSIM=0.855.ckpt  # ResFusion model 
   │   └── restormer_model/
   │       └── model_best.pth  # Restormer model
   ```

3. Download the test dataset from the same Google Drive link and place it in a folder named `private_dataset` at the root level.

## Step 2: Run Raindrop-Pipeline Inference

1. Open `raindrop-pipeline-inference.ipynb`
2. Make sure the model path points to your downloaded model:
   ```python
   model_path = './model/raindrop_model/model_epoch800.pth'
   ```
3. Ensure your test data path is set correctly:
   ```python
   test_dir = './private_dataset'
   ```
4. Run all cells in the notebook
5. The results will be saved in `./raindrop_output` directory

## Step 3: Run ResFusion Inference

1. Open `resfusion-inference.ipynb`
2. Make sure the model checkpoint path is correct:
   ```python
   parser.add_argument('--model_ckpt', default='./model/resfusion_model/lightning_logs/version_0/checkpoints/best-epoch=199-val_SSIM=0.855.ckpt', type=str)
   ```
3. Ensure your input data path is set correctly:
   ```python
   parser.add_argument('--data_dir', default='./private_dataset', type=str)
   ```
4. Run all cells in the notebook
5. The results will be saved in `./resfusion_inference` directory

## Step 4: Run Ensemble

1. Open `ensemble.ipynb`
2. Verify the input paths for both models:
   ```python
   raindrop_dir = "./raindrop_output"
   resfusion_dir = "./resfusion_inference"
   ```
3. Set the output directory:
   ```python
   output_dir = "./ensemble_output"
   ```
4. Run all cells in the notebook
5. The ensemble results will be saved in the `./ensemble_output` directory

## Step 5: Run Restormer Inference (Final Denoising)

1. Open `restormer-inference.ipynb`
2. Set the correct model path:
   ```python
   model_restoration = model.Restormer()
   model_restoration.load_state_dict(torch.load('./model/restormer_model/model_best.pth'))
   ```
3. Set input path to the ensemble output:
   ```python
   input_folder = "./ensemble_output"
   ```
4. Set the output directory:
   ```python
   result_dir = "./final_output"
   ```
5. Run all cells in the notebook
6. The final results will be saved in the `./final_output` directory

## Important Notes

1. **GPU Memory**: These models require significant GPU memory. If you encounter out-of-memory errors, try processing images in smaller batches.

2. **Model Paths**: Double-check that all model paths are correctly set according to where you downloaded them.

3. **Execution Order**: The notebooks must be run in the order specified above, as each step depends on the outputs of the previous step.

4. **Folder Structure**: Make sure to create any missing directories if the notebooks raise errors about non-existent folders.

5. **Clone Repositories**: Some models may require cloning their original repositories. The notebooks include cells for this, but ensure they execute properly.

## Troubleshooting

- If models fail to load, verify that the checkpoint files are in the correct format and location
- If an error occurs during inference, check the model's original repository for specific requirements
- For CUDA out-of-memory errors, reduce batch size or use CPU inference
- If tensorboard errors occur, ensure the tensorboard package is installed correctly
- For visualization issues, make sure matplotlib is installed properly

## Results

After running the complete pipeline, you'll have results from each stage:
1. Raindrop-Removal GAN results in `./raindrop_output/`
2. ResFusion results in `./resfusion_inference/`
3. Ensemble results in `./ensemble_output/`
4. Final denoised results in `./final_output/`

The final output directory contains the highest quality raindrop-removed images that can be used for your application. 