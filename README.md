# Revealing Hidden Decision Patterns in Machine Learning Models

## Detecting Clever Hans Effects in CLIP ResNet-50

This project investigates whether the CLIP ResNet-50 model truly learns meaningful semantic patterns 
from image data, or whether it exploits irrelevant features (logos, text, watermarks) that accidentally 
correlate with the correct output — the so-called **Clever Hans (CH) effect**.

## Key Results

| Experiment | Accuracy |
|---|---|
| CLIP Original | 84.00% |
| CLIP + Logo Inserted | 81.25% |
| CH Mitigation (Clean) | 83.25% |
| CH Mitigation (+ Logo) | 82.50% |

## Pipeline

1. **Dataset**: ImageNet Truck subset (8 classes, 10,133 train / 400 test)
2. **Model**: CLIP ResNet-50 (1024-dim embeddings)
3. **Classifier**: Logistic Regression on CLIP features
4. **CH Detection**: Logo insertion → accuracy drop confirms shortcut reliance
5. **CH Mitigation**: Masking discriminative filters in `encoder.relu3`
6. **XAI Visualization**: Grad-CAM + BiLRP analysis

## Files

- `CH_Detection_Pipeline.ipynb` — Main notebook with full pipeline
- `results/` — All generated visualizations and charts
- `System_Analysis_Design_Document.pdf` — SA&D document

## Tech Stack

- Python 3.10+, PyTorch, OpenAI CLIP, scikit-learn
- Google Colab (T4 GPU)
- Hugging Face Datasets (ImageNet-1k streaming)
- Matplotlib for visualizations

## How to Run

1. Open `CH_Detection_Pipeline.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run all cells — data loads from Hugging Face, results save to Google Drive
