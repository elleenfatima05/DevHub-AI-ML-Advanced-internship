# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a transformer model (`bert-base-uncased`) to classify news headlines into 4 topic categories: World, Sports, Business, and Sci/Tech.

## Dataset
- **Source:** AG News Dataset (Hugging Face Datasets — `fancyzhx/ag_news`)
- **Training samples used:** 8,000
- **Test samples used:** 2,000
- **Classes:** World, Sports, Business, Sci/Tech

## Methodology / Approach
1. Loaded and preprocessed the AG News dataset using Hugging Face `datasets`
2. Tokenized headlines using `BertTokenizerFast` with a max length of 128 tokens
3. Fine-tuned `bert-base-uncased` for sequence classification using Hugging Face `Trainer` API
4. Trained for 3 epochs with a learning rate of 2e-5 and weight decay of 0.01
5. Evaluated using accuracy and weighted F1-score
6. Generated a confusion matrix and per-class classification report

## Tools & Technologies
`Python`, `Hugging Face Transformers`, `PyTorch`, `scikit-learn`, `Streamlit`

## Key Results / Observations

### Overall Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.9150 (91.5%) |
| Weighted F1-score | 0.9150 |
| Validation Loss | 0.3323 |

### Per-Class Breakdown
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| World | 0.92 | 0.93 | 0.93 | 497 |
| Sports | 0.98 | 0.97 | 0.98 | 483 |
| Business | 0.90 | 0.86 | 0.88 | 522 |
| Sci/Tech | 0.86 | 0.90 | 0.88 | 498 |
| **Weighted Avg** | **0.92** | **0.92** | **0.92** | **2000** |

### Observations
- The model achieved strong overall accuracy of **91.5%** after just 3 epochs of fine-tuning on 8,000 samples.
- **Sports** was the easiest category to classify (F1: 0.98), likely due to its distinct vocabulary (player names, scores, match terminology).
- **Business** and **Sci/Tech** showed the most confusion with each other (F1: 0.88 each), which is expected since both categories frequently involve technology companies, economic data, and innovation topics that overlap in phrasing.
- **World** news performed well (F1: 0.93), benefiting from geographically specific language.

## Repository Structure
```
Task_1/
├── task1_bert_news_classifier.ipynb    # Full notebook: preprocessing, training, evaluation
├── confusion_matrix.png                # Confusion matrix visualization
├── README.md
└── requirements.txt
```

## How to Run
Open `task1_bert_news_classifier.ipynb` in Google Colab with a T4 GPU runtime enabled:
`Runtime > Change runtime type > Hardware accelerator > T4 GPU`

Then run all cells top to bottom.

> Note: Fine-tuned model weights are not included in this repository due to file size (400MB+). The notebook contains all training outputs, metrics, and visualizations as cell outputs.

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
