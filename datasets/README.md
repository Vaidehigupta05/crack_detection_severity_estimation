# Dataset Setup

This project uses three publicly available datasets hosted on Kaggle.

1. [Cracked and Non-Cracked Surface Dataset](https://www.kaggle.com/datasets/geadalfa/cracked-non-cracked-surface-datasets)
2. [Crack Segmentation Dataset](https://www.kaggle.com/datasets/aravindnagarajan/crack-segmentation-dataset)
3. [Surface Crack Detection Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

## Dataset Extraction

The datasets are downloaded and extracted directly in Google Colab using the Kaggle API.

### Requirements
- A Kaggle account.
- Kaggle API credentials (`kaggle.json`).

### Setup
1. Create a Kaggle account or sign in.
2. Go to your Kaggle account settings and generate an API token.
3. Upload the downloaded `kaggle.json` file to Google Colab when prompted.
4. Run the dataset extraction cells in the notebook.

The notebook handles downloading and extracting the datasets.

**Note:** Never upload your `kaggle.json` file or API credentials to GitHub.
