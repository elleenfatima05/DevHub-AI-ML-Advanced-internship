# Task 4: Context-Aware Chatbot Using LangChain / RAG

## Objective
Build a conversational chatbot that remembers context and retrieves answers from a custom document corpus using Retrieval-Augmented Generation (RAG).

## Dataset / Knowledge Base
- **Source:** Custom AI/ML corpus covering 16 topics including Machine Learning, Deep Learning, NLP, Transformers, BERT, RAG, Vector Databases, LangChain, PyTorch, and LLMs
- **Chunks created:** Corpus split into overlapping 300-character chunks with 50-character overlap using `RecursiveCharacterTextSplitter`

## Methodology / Approach
1. Built a custom AI/ML knowledge corpus and split it into overlapping chunks using `RecursiveCharacterTextSplitter`
2. Generated dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, local CPU)
3. Stored and indexed all chunk embeddings in a **FAISS** vector store for fast similarity search
4. Loaded `google/flan-t5-base` as a local LLM for answer generation (no API key required)
5. Built a `ConversationalRetrievalChain` with `ConversationBufferMemory` to retain full chat history across turns
6. Deployed an interactive chatbot interface using **Streamlit** with expandable source chunk display

## Tools & Technologies
`Python`, `LangChain`, `FAISS`, `sentence-transformers`, `HuggingFace Transformers`, `flan-t5-base`, `Streamlit`

## System Overview
| Component | Choice | Reason |
|-----------|--------|--------|
| Embeddings | all-MiniLM-L6-v2 | Fast, local, no API key needed |
| Vector Store | FAISS | Free, in-memory, fast similarity search |
| LLM | google/flan-t5-base | Local CPU inference, no cost |
| Memory | ConversationBufferMemory | Retains full chat history across turns |
| Framework | LangChain | Modular, composable chain construction |

## Key Results / Observations

### Retrieval Quality
The retriever was tested on 5 diverse questions — all 5 returned perfectly relevant top chunks:

| Query | Retrieved Topic |
|-------|----------------|
| What is BERT used for? | BERT description and Google origin |
| How do vector databases work? | Vector DB embeddings and similarity search |
| What is prompt engineering? | Prompt engineering techniques |
| Tell me about PyTorch | PyTorch framework and Facebook origin |
| What are large language models? | LLMs definition and examples |

Semantic similarity search worked correctly even when query phrasing differed from the exact wording in the corpus — confirming that embedding-based retrieval generalizes beyond keyword matching.

### Conversational Memory
- The chatbot successfully maintained context across 5 conversation turns
- Follow-up questions (e.g., "How is it different from deep learning?" after asking about ML) were correctly resolved using prior conversation history stored in `ConversationBufferMemory`
- Full chat history was retained and visible in memory inspection, confirming no context loss between turns

### Streamlit Deployment
- Chatbot deployed successfully as a local Streamlit web app
- Each response includes an expandable "Source chunks used" section showing which corpus passages the answer was grounded in
- Sample queries tested: "What is Machine Learning?" → correctly answered; "What is RAG?" → returned "Retrieval-Augmented Generation"

### Limitations
- `flan-t5-base` is a small model — answers are factual but short; a larger LLM (e.g., Llama 3, Mistral) would produce more fluent and detailed responses
- Corpus is limited to the provided AI/ML text — questions outside this domain return generic or incomplete answers
- FAISS index is in-memory only — a persistent vector database (e.g., ChromaDB, Pinecone) would be needed for production deployment

## Repository Structure
```
Task_4/
├── task4_rag_chatbot.ipynb     # Full notebook: corpus prep, embeddings, RAG chain, evaluation
├── app.py                       # Streamlit chatbot deployment interface
├── app_screenshot.png           # Screenshot of live chatbot demo
├── README.md
└── requirements.txt
```

## How to Run

**Step 1 — Install dependencies:**
```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers torch streamlit
```

**Step 2 — Run the notebook** to build and save the FAISS index:
```bash
jupyter notebook task4_rag_chatbot.ipynb
```

**Step 3 — Launch the Streamlit app:**
```bash
streamlit run app.py
```

## Author
Elleen — AI/ML Engineering Internship, DevelopersHub Corporation
