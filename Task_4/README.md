# Task 4: Context-Aware Chatbot Using LangChain / RAG

## Objective
Build a conversational chatbot that can remember context and retrieve external information during conversations.

## Dataset
- **Source:** Custom corpus (e.g., Wikipedia pages, internal documents, or any knowledge base)

## Methodology / Approach
1. Used LangChain with Retrieval-Augmented Generation (RAG)
2. Implemented context memory for conversational history
3. Embedded documents and retrieved answers from a vectorized document store
4. Deployed the chatbot with Streamlit

## Tools & Technologies
`Python`, `LangChain`, `FAISS` / `Chroma` (vector store), `sentence-transformers` (embeddings), `Streamlit`, LLM (open-source or API-based)

## Key Results / Observations
[Add notes on retrieval quality, response coherence, and any limitations observed — e.g., handling of out-of-corpus questions.]

## Repository Structure
```
├── app.py                # Streamlit app — chatbot interface
├── ingest.py              # Document loading, chunking, embedding, vector store creation
├── README.md
├── requirements.txt
└── data/                  # Source corpus / documents
```

## How to Run
```bash
pip install -r requirements.txt

# Step 1: Build the vector store from your corpus
python ingest.py

# Step 2: Launch the chatbot
streamlit run app.py
```

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
