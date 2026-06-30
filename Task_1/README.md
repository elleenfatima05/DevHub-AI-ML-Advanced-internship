# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a transformer model (BERT) to classify news headlines into topic categories.

## Dataset
- **Source:** AG News Dataset (available on Hugging Face Datasets)
- **Classes:** World, Sports, Business, Sci/Tech

## Methodology / Approach
1. Tokenized and preprocessed the dataset
2. Fine-tuned `bert-base-uncased` using Hugging Face Transformers
3. Evaluated the model using accuracy and F1-score
4. Deployed the model using Streamlit/Gradio for live interaction

## Tools & Technologies
`Python`, `Hugging Face Transformers`, `PyTorch`, `scikit-learn`, `Streamlit` / `Gradio`

## Key Results / Observations
| Metric | Score |
|--------|-------|
| Accuracy | [fill in] |
| F1-score | [fill in] |

[Add 1–2 sentences on observations — e.g., which categories were easiest/hardest to classify.]

## Repository Structure
```
├── notebook.ipynb       # Tokenization, fine-tuning, evaluation
├── README.md
├── requirements.txt
├── app.py                # Streamlit/Gradio deployment interface
└── outputs/              # Saved model checkpoint, plots
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb

# To launch the live demo
streamlit run app.py
```

> Note: Fine-tuned BERT weights are not pushed directly to GitHub due to file size limits. Model is hosted on [Hugging Face Hub link] — see `outputs/model_card.md` for details.

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
