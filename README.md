# NeuroScan AI

### Explainable Brain MRI Classification Using Deep Learning

NeuroScan AI is an educational AI application that classifies
brain MRI images into four categories using a fine-tuned ResNet18
convolutional neural network.

The application provides temperature-scaled class probabilities,
confidence-based review recommendations, and Grad-CAM explanations.

> **Disclaimer:** This project is an educational research prototype.
> It is not validated for medical diagnosis, screening, or patient care.

## Features

- Four-class brain MRI classification
- Fine-tuned ResNet18 with transfer learning
- Temperature scaling for probability calibration
- Experimental confidence-based review threshold
- Grad-CAM visualization
- Interactive Streamlit gallery
- CPU-compatible inference

## Classification Categories

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

## Machine Learning Pipeline

1. MRI dataset preparation and cleaning
2. Image quality and duplicate screening
3. Data augmentation and preprocessing
4. ResNet18 transfer learning
5. Hyperparameter optimization using Optuna
6. External MRI data integration
7. Version 2 model fine-tuning
8. Temperature scaling
9. Grad-CAM explainability
10. Streamlit application deployment

## Model Performance

| Metric | Original ResNet18 | Version 2 |
|---|---:|---:|
| Historical accuracy | 90.20% | 91.40% |
| Historical macro F1 | 90.04% | 91.27% |
| Tumor-to-no-tumor errors | 93 | 63 |

These results are from a previously examined historical test set
and should be considered exploratory, not independent clinical
validation.

Version 2 was selected for the educational demonstration because
it reduced historical tumor-to-no-tumor errors.

## Model Configuration

- Architecture: ResNet18
- Initial checkpoint: Optuna Trial 1
- Fine-tuned checkpoint: Version 2, epoch 4
- Input size: 224 x 224
- Calibration temperature: approximately 1.5373
- Experimental review threshold: 97%
- Explainability: Grad-CAM

## Technology Stack

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- Pillow and Matplotlib
- Streamlit
- Optuna (model development)

## Running Locally

Create and activate a Python virtual environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The application provides a preloaded gallery of demonstration MRI
images. Arbitrary user image uploads are disabled in this demo.

## Data Sources and Attribution

**Original training dataset:**

Masoud Nickparvar, Brain Tumor MRI Dataset, Kaggle.

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**Additional glioma and meningioma data:**

Qadri et al., Brain Tumor MR Image Data Set for Machine Vision
Approach for Brain Tumor Classification Using Multi Features Dataset.

Zenodo, record 7047164.

https://doi.org/10.5281/zenodo.7047164

Source dataset permissions and attribution requirements apply.
Redistribution rights for individual images and derived assets
should be checked before publishing them.

## Limitations

- No patient-independent clinical validation
- Potential dataset-source bias
- Historical evaluation data previously examined during development
- No guaranteed tumor localization from Grad-CAM
- Confidence scores are not clinical diagnostic probabilities
- Not suitable for medical decisions

## Project Status

- Model development: Complete for the V2 prototype
- Streamlit demo: Functional
- Four-class demo verification: Passed
- Public deployment: In preparation

## Purpose

This project demonstrates applied deep learning, model evaluation,
dataset-bias investigation, explainability, calibration, and
end-to-end AI application development.

## Demo Image Provenance

The 12 demo MRI images originate from the Brain Tumor MRI
Dataset curated by Masoud Nickparvar on Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

All 12 demo PNGs were matched to original JPEG source
images using exact decoded RGB pixel comparison.

The file `data/demo_samples/demo_manifest.csv` records
each original source filename and verification status.

The Kaggle dataset is listed under CC BY 4.0 and combines
images from figshare, SARTAJ, and Br35H. Original dataset
creators and applicable source permissions should be respected.

These images are educational demonstration examples, not
evidence of clinical validity or patient-independent testing.
