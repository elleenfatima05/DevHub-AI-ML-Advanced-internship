# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn.

## Dataset
- **Source:** Telco Churn Dataset

## Methodology / Approach
1. Implemented data preprocessing steps (scaling, encoding) using `Pipeline`
2. Trained models including Logistic Regression and Random Forest
3. Used `GridSearchCV` for hyperparameter tuning
4. Exported the complete pipeline using `joblib`

## Tools & Technologies
`Python`, `scikit-learn`, `pandas`, `joblib`

## Key Results / Observations
| Model | Accuracy | Best Params |
|-------|----------|-------------|
| Logistic Regression | [fill in] | [fill in] |
| Random Forest | [fill in] | [fill in] |

[Add 1–2 sentences on which model performed best and why.]

## Repository Structure
```
├── notebook.ipynb       # Preprocessing, training, tuning, evaluation
├── README.md
├── requirements.txt
└── outputs/
    └── churn_pipeline.pkl    # Exported pipeline (joblib)
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

To load the exported pipeline for inference:
```python
import joblib
pipeline = joblib.load("outputs/churn_pipeline.pkl")
pipeline.predict(new_data)
```

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
