# Trained Models

This project includes multiple trained deep learning models for surface crack detection and segmentation.

Due to the large file sizes, the trained model weights are hosted on Google Drive instead of being stored directly in this GitHub repository.

## Available Models

| Model File | Description |
|---|---|
| `best_unet_crack_model.pth` | Trained U-Net model for pixel-level crack segmentation. |
| `custom_cnn_crack_model.pth` | Custom CNN model for crack classification. |
| `mobilenet_v2_balanced_finetuned.pth` | Fine-tuned MobileNetV2 model trained using a balanced dataset. |
| `mobilenet_v2_crack_finetunes.pth` | Fine-tuned MobileNetV2 model for crack detection. |
| `mobilenet_v2_crack_model.pth` | Trained MobileNetV2 model for crack classification. |

## Download Trained Models

**[Click here to access and download all trained model weights from Google Drive](https://drive.google.com/drive/folders/1fV2mVHoNUVBZW9pEsjOXcFILVVhlLznJ?usp=sharing)**

1. Open the Google Drive folder using the link above.
2. Select the model weights you want to use.
3. Click Download.
4. Place the downloaded `.pth` file in the `models/` directory of your project.
5. Update the model path in the notebook if required.

## Important Notes

- All model weights are saved in PyTorch `.pth` format.
- The models can be loaded using PyTorch.
- Use the corresponding model architecture when loading a checkpoint.
- Model weights are not included directly in this GitHub repository because of their large file sizes.

Refer to the main project notebook for training, evaluation, and inference instructions.
