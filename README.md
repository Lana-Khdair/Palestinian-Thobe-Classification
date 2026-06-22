# Palestinian Thobe Classification using Deep Learning

## Overview

Palestinian embroidery (**Tatreez**) is an important part of Palestinian cultural heritage. This project leverages deep learning and transfer learning techniques to classify Palestinian thobes according to their regional origin:

* Nablus
* Jaffa
* Bethlehem

The model learns subtle visual differences in embroidery patterns and aims to support the preservation and digital documentation of Palestinian heritage through artificial intelligence.

---

## Demo

🌐 **Live Streamlit Application** Try the deployed application here:
https://thobe-classifier.streamlit.app/

Upload an image of a Palestinian thobe and receive a prediction of its regional origin along with model confidence scores and visual explanations.

🎥 **Video Demonstration** Watch a short demonstration of the application: 
https://drive.google.com/drive/my-drive

---

## Dataset

Since no publicly available dataset exists for this task, we built our own dataset by collecting Palestinian thobe images from publicly available online sources and organizing them according to their regional embroidery styles.

📂 **Kaggle Dataset:**  
https://www.kaggle.com/datasets/zeinadawod/palestinian-traditional-thobe-image-dataset

### Classes

| Class     | Description                                       |
| --------- | ------------------------------------------------- |
| Nablus    | Traditional embroidery patterns from Nablus       |
| Jaffa     | Distinctive embroidery styles from Jaffa          |
| Bethlehem | Characteristic embroidery patterns from Bethlehem |

---

## Methodology

### Data Preprocessing

* Image resizing and normalization
* Data augmentation
* Dataset cleaning and organization
* Exploratory Data Analysis (EDA)

### Deep Learning Models

We evaluated multiple transfer learning architectures:

* EfficientNetB0
* MobileNet

### Fine-Tuning

Several fine-tuning strategies were explored to optimize feature extraction and classification performance.

### Model Explainability

To better understand model predictions, **Grad-CAM** was used to visualize the image regions that most influenced the classification decision.

---

## Results

🏆 **Best Model:** EfficientNetB0 (Strategy B)

| Metric        | Value      |
| ------------- | ---------- |
| Test Accuracy | **92.86%** |

These results demonstrate the effectiveness of transfer learning for fine-grained embroidery pattern recognition.

---

## Technologies Used

* Python
* PyTorch
* EfficientNetB0
* MobileNet
* Grad-CAM
* Streamlit
* NumPy
* Pandas
* Matplotlib
* OpenCV

---

## Repository Structure

```text
```text
Palestinian-Thobe-Classification/
│
├── Dataset/                          # Original dataset
├── Dataset_cropped/                  # Cropped embroidery regions
├── data_split_cropped/               # Train/Validation/Test splits
├── RealTest_candidates/              # Real-world test samples
├── RealTest_candidates_cropped/      # Cropped real-world samples
│
├── app.py                            # Streamlit web application
├── thobe_classifier.py               # Classification pipeline
├── crop_dataset.py                   # Dataset cropping utility
├── run_gradcam.py                    # Grad-CAM visualization script
│
├── thobe_efficientnet_best.pt        # Best EfficientNetB0 model
├── thobe_mobilenet_best.pt           # Best MobileNetV2 model
├── eff_strategy_b.pt                 # Best EfficientNet strategy
├── mob_strategy_b.pt                 # Best MobileNet strategy
├── *.pt                              # Additional trained model checkpoints
│
├── gradcam_eff.png                   # EfficientNet Grad-CAM results
├── gradcam_mob.png                   # MobileNet Grad-CAM results
├── cm_efficientnetb0_best.png        # Confusion matrix
├── cm_mobilenetv2_best.png
├── roc_efficientnetb0_best.png       # ROC curves
├── roc_mobilenetv2_best.png
├── training_curves_*.png             # Training performance plots
├── lr_*.png                          # Learning rate analysis
├── strategies_*.png                  # Strategy comparisons
├── final_comparison.png              # Final model comparison
│
├── requirements.txt
└── README.md
```

```

---

## Running Locally

```bash
git clone https://github.com/Lana-Khdair/Palestinian-Thobe-Classification.git

cd Palestinian-Thobe-Classification

pip install -r requirements.txt

streamlit run app.py
```

---

## Future Work

* Expand the dataset with additional Palestinian cities.
* Explore more advanced vision architectures.
* Improve robustness to varying image quality and backgrounds.
* Develop a mobile-friendly deployment.

---

## Team

* Lana Khdair
* Aya Hijjawi

