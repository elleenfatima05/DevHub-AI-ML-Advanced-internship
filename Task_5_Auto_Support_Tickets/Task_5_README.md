# Task 5: Auto-Tagging Support Tickets Using LLM

## Objective
Automatically tag support tickets into categories using a Large Language Model (LLM) through prompt engineering, comparing zero-shot vs few-shot performance and outputting the top 3 most probable tags per ticket.

## Dataset
- **Source:** Synthetic support ticket dataset — 30 tickets across 6 categories (5 tickets per category)
- **Categories:** Billing, Technical, Account, Shipping, Refund, General

## Methodology / Approach
1. Created a realistic synthetic dataset of 30 support tickets covering 6 common support categories
2. Loaded `google/flan-t5-base` as a local LLM for inference (no API key required)
3. Implemented **zero-shot prompting** — model classifies tickets with no examples provided
4. Implemented **few-shot prompting** — model is given 6 labeled examples (one per category) before classifying
5. Compared accuracy of both approaches across all 30 tickets
6. Generated top-3 most probable tags per ticket using per-category confidence scoring
7. Evaluated using accuracy, classification report, and confusion matrix

## Tools & Technologies
`Python`, `HuggingFace Transformers`, `flan-t5-base`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## Key Results / Observations

### Performance Comparison
| Approach | Accuracy |
|----------|----------|
| Zero-Shot | 0.6333 (63.3%) |
| Few-Shot | 0.2333 (23.3%) |

**Notable finding:** Zero-shot outperformed few-shot by 40% — an unexpected but insightful result explained by the prediction distribution chart: the few-shot model over-generalized to the "General" category, classifying approximately 27 out of 30 tickets as "General" regardless of content. This is a known failure mode of small models with few-shot prompting where the model latches onto the last example's label rather than learning the pattern.

### Top-3 Tag Results (10 tickets evaluated)
The correct label appeared in the top-3 tags for **all 10 tested tickets (100% top-3 accuracy)**:

| Ticket | True Label | Top 3 Tags |
|--------|-----------|------------|
| Charged twice for subscription | Billing | Refund, **Billing**, Technical |
| Incorrect invoice amount | Billing | **Billing**, Technical, Account |
| Cannot find billing history | Billing | **Billing**, Technical, Account |
| Charged after cancellation | Billing | **Billing**, Technical, Account |
| Need receipt for purchase | Billing | **Billing**, Technical, Account |
| App crashes on upload | Technical | Billing, **Technical**, Account |
| 500 server error on login | Technical | Billing, **Technical**, Account |
| Dashboard extremely slow | Technical | Billing, **Technical**, Account |
| Data export not working | Technical | Billing, **Technical**, Account |
| Duplicate search results | Technical | Billing, **Technical**, Account |

### Prediction Distribution Observations
- **Zero-shot** produced a more balanced distribution across categories — Account (11), General (7), Technical (6), with fewer predictions for Billing, Refund, and Shipping
- **Few-shot** almost entirely collapsed to the "General" category (~27/30 tickets), showing that flan-t5-base is too small to reliably learn from in-context examples
- **True distribution** was perfectly balanced at 5 tickets per category

### Limitations
- `flan-t5-base` is a small model — a larger LLM (GPT-4, Llama 3, Mistral) would achieve significantly higher accuracy on both approaches and handle few-shot learning more reliably
- Dataset is synthetic — real-world tickets are more ambiguous, noisy, and often span multiple categories
- Top-3 confidence scoring is a heuristic approach via prompting — proper probability calibration would require access to model logits
- Few-shot performance degraded due to model size constraints, not the technique itself — few-shot learning is proven to work well with larger models

## Repository Structure
```
Task_5/
├── task_5_auto_support_tickets.ipynb    # Full notebook: dataset, prompting, evaluation
├── comparison_plot.png                   # Zero-shot vs few-shot accuracy + distribution
├── confusion_matrix.png                  # Few-shot confusion matrix
├── README.md
└── requirements.txt
```

## How to Run
Open the notebook in Google Colab (GPU recommended for faster inference):
`Runtime > Change runtime type > Hardware accelerator > T4 GPU`

Run all cells top to bottom.

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
