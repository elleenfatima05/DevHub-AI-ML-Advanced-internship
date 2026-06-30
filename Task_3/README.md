# Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

## Objective
Predict housing prices using both structured (tabular) data and house images.

## Dataset
- **Source:** Housing Sales Dataset + Custom Image Dataset

## Methodology / Approach
1. Used a CNN to extract features from house images
2. Combined extracted image features with tabular data
3. Trained a regression model using both modalities (feature fusion)
4. Evaluated performance using MAE and RMSE

## Tools & Technologies
`Python`, `TensorFlow/Keras` or `PyTorch`, `pandas`, `scikit-learn`, CNNs

## Key Results / Observations
| Metric | Score |
|--------|-------|
| MAE | [fill in] |
| RMSE | [fill in] |

[Add 1–2 sentences — e.g., whether combining image + tabular features improved performance over tabular-only baseline.]

## Repository Structure
```
├── notebook.ipynb       # CNN feature extraction, fusion, training, evaluation
├── README.md
├── requirements.txt
└── outputs/              # Saved model, plots
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

> Note: Trained model weights/checkpoints are not pushed to GitHub due to file size — hosted externally (link in `outputs/model_card.md`).

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
