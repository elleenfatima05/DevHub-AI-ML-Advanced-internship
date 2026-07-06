# Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

## Objective
Predict housing prices using both structured tabular data and house images by combining CNN-extracted image features with traditional tabular features through a feature fusion approach.

## Dataset
- **Tabular:** California Housing Dataset (sklearn built-in) — 8 features, 5,000 samples used
- **Images:** Synthetic house images generated from tabular features (64×64 RGB), resized to 224×224 for CNN input
- **Target:** Median house value (in $100,000 units)

## Methodology / Approach
1. Loaded California Housing tabular dataset and generated synthetic house images encoding key features (income, room count, age, population) as visual properties
2. Scaled tabular features using `StandardScaler`
3. Built a **dual-branch multimodal model** in PyTorch:
   - **CNN Branch:** Pretrained MobileNetV2 (ImageNet weights, last 20 layers fine-tuned) extracts 1280-dim image features
   - **Tabular Branch:** Small MLP (64→32 units with ReLU + Dropout) processes 8 tabular features
   - **Fusion Layer:** Concatenated features passed through a regression head (256→64→1)
4. Trained for 15 epochs using MSE loss, Adam optimizer, and StepLR scheduler
5. Evaluated using MAE and RMSE against a tabular-only Ridge Regression baseline

## Tools & Technologies
`Python`, `PyTorch`, `torchvision`, `MobileNetV2`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

## Key Results / Observations

### Model Performance
| Model | MAE | RMSE |
|-------|-----|------|
| Tabular-Only Baseline (Ridge Regression) | 0.5220 | 0.7080 |
| Multimodal (CNN + Tabular) | **0.4285** | **0.6060** |

The multimodal model reduced MAE by **17.9%** and RMSE by **14.4%** compared to the tabular-only baseline, confirming that image features contributed meaningful signal to the prediction task.

### Loss Curve Observations
- Both train and validation loss dropped sharply in the first 3 epochs, from ~0.9 down to ~0.45, indicating fast initial learning from pretrained CNN weights
- Validation loss showed minor jumps in early epochs (epochs 2–3) before stabilizing — expected behavior during fine-tuning of a pretrained network
- From epoch 6 onward, both losses converged and tracked closely together (both ending around 0.37–0.38), indicating no significant overfitting

### Predicted vs Actual Plot Observations
- Predictions follow the red diagonal line well across the mid-price range (1–4 units), showing good model calibration in the most common price range
- Noticeable scatter at higher price values (above 4.5 units), which is expected — the California Housing dataset caps values at 5.0, creating a clustering effect at the upper end that the model struggles to predict precisely
- Overall point distribution around the diagonal is strong, consistent with the low MAE of 0.4285

## Repository Structure
```
Task_3/
├── task3_multimodal_housing_price.ipynb    # Full notebook: data prep, model, training, evaluation
├── eda_plots.png                            # Price distribution + correlation heatmap
├── sample_images.png                        # Sample synthetic house images
├── loss_curve.png                           # Training vs validation loss
├── predicted_vs_actual.png                  # Predicted vs actual scatter plot
├── README.md
└── requirements.txt
```

## How to Run
Open `task3_multimodal_housing_price.ipynb` in Google Colab with T4 GPU runtime:
`Runtime > Change runtime type > Hardware accelerator > T4 GPU`

Run all cells top to bottom. Training takes approximately 10–15 minutes on a free T4 GPU.

> Note: Model weights (.pth file) are not included in this repository due to file size. All training outputs, metrics, and visualizations are saved as cell outputs in the notebook.

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
