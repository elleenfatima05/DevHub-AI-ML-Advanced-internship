import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Context-Aware RAG Chatbot")
st.write("Ask questions about AI and Machine Learning topics.")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        "outputs/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False,
        device=-1
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain

chain = load_chain()

# Initialize chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            result = chain.invoke({"question": question})
            answer = result["answer"]
            st.write(answer)

            with st.expander("Source chunks used"):
                for doc in result["source_documents"]:
                    st.write(f"- {doc.page_content[:150]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})