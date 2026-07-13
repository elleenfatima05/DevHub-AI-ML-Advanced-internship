# DevelopersHub Corporation — AI/ML Engineering Internship (Advanced Tasks)

## Overview
This repository contains my submissions for the **Advanced Internship Task Set** as part of the AI/ML Engineering Internship at **DevelopersHub Corporation**.

The internship required completing **at least 3 out of 5 advanced tasks** — all 5 tasks have been completed successfully.

**Due Date:** 21st July, 2026

## Technologies Used
`Python` `Hugging Face Transformers` `PyTorch` `scikit-learn` `LangChain` `FAISS` `Streamlit` `CNNs` `MobileNetV2` `joblib` `LLMs` `sentence-transformers` `flan-t5-base` `BERT`

## Tasks

| # | Task | Status | Folder |
|---|------|--------|--------|
| 1 | News Topic Classifier Using BERT | ✅ Complete | [`Task_1_BERT_News_Classifier/`](./Task_1_BERT_News_Classifier) |
| 2 | End-to-End ML Pipeline with Scikit-learn | ✅ Complete | [`Task_2_End-to-End_ML_Pipeline/`](./Task_2_End-to-End_ML_Pipeline) |
| 3 | Multimodal ML — Housing Price Prediction | ✅ Complete | [`Task_3_Multimodal_Housing_Price/`](./Task_3_Multimodal_Housing_Price) |
| 4 | Context-Aware Chatbot Using LangChain / RAG | ✅ Complete | [`Task_4_Context_Aware_ChatBot/`](./Task_4_Context_Aware_ChatBot) |
| 5 | Auto-Tagging Support Tickets Using LLM | ✅ Complete | [`Task_5_Auto_Support_Tickets/`](./Task_5_Auto_Support_Tickets) |

## Task Summaries

### Task 1 — News Topic Classifier Using BERT
Fine-tuned `bert-base-uncased` on the AG News dataset to classify news headlines into 4 categories (World, Sports, Business, Sci/Tech).
- **Accuracy:** 91.5% | **F1-Score:** 0.9150
- **Tools:** Hugging Face Transformers, PyTorch, scikit-learn

### Task 2 — End-to-End ML Pipeline with Scikit-learn
Built a reusable and production-ready ML pipeline for customer churn prediction using the Telco Churn dataset.
- **Best Model:** Random Forest (tuned with GridSearchCV)
- **Tools:** scikit-learn Pipeline API, joblib, pandas

### Task 3 — Multimodal ML — Housing Price Prediction
Predicted housing prices by fusing CNN-extracted image features with tabular data using a dual-branch PyTorch model.
- **MAE:** 0.4285 | **RMSE:** 0.6060 (17.9% MAE improvement over tabular-only baseline)
- **Tools:** PyTorch, MobileNetV2, scikit-learn

### Task 4 — Context-Aware Chatbot Using LangChain / RAG
Built a conversational RAG chatbot that retrieves answers from a custom AI/ML knowledge corpus using FAISS vector search and maintains context across turns via ConversationBufferMemory.
- **Retrieval accuracy:** 5/5 test queries returned perfectly relevant chunks
- **Tools:** LangChain, FAISS, sentence-transformers, flan-t5-base, Streamlit

### Task 5 — Auto-Tagging Support Tickets Using LLM
Automatically tagged support tickets using zero-shot and few-shot prompting with flan-t5-base, comparing both approaches and generating top-3 tags per ticket.
- **Zero-Shot Accuracy:** 63.3% | **Top-3 Accuracy:** 100% (10 tickets evaluated)
- **Tools:** Hugging Face Transformers, flan-t5-base, scikit-learn

## Repository Structure
```
.
├── README.md
├── .gitignore
├── Task_1_BERT_News_Classifier/
│   ├── task1_bert_news_classifier.ipynb
│   ├── confusion_matrix.png
│   └── README.md
├── Task_2_End-to-End_ML_Pipeline/
│   ├── task2_ml_pipeline_churn.ipynb
│   └── README.md
├── Task_3_Multimodal_Housing_Price/
│   ├── task_3_multimodal_housing_price_prediction.ipynb
│   ├── sample_images.png
│   └── README.md
├── Task_4_Context_Aware_ChatBot/
│   ├── task4_context_aware_chatbot.ipynb
│   ├── app.py
│   ├── app_Screenshot.png
│   └── README.md
└── Task_5_Auto_Support_Tickets/
    ├── task_5_auto_support_tickets.ipynb
    ├── comparison_plot.png
    ├── confusion_matrix.png
    └── README.md
```

## Author
**Elleen**
AI/ML Engineering Intern — DevelopersHub Corporation
