import streamlit as st
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


# ------------------------------------------------------------
#                   DATA LOADING & PROCESSING
# ------------------------------------------------------------


@st.cache_resource
def load_and_process_data():
    """
    Loads the Wikitext dataset and splits it into smaller text chunks for retrieval.
    ...
    """
    # 1. Load the Wikitext-2 raw dataset (train split)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_doc = [
        Document(page_content=item["text"]) for item in dataset if item["text"].strip()
    ]

    # 2. Split text into chunks for vector indexing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(train_doc)


@st.cache_resource
def get_embeddings():
    """
    Loads the Hugging Face sentence-transformer model for text embeddings.
    ...
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def build_vector_store(_train_chunks, _embeddings):
    """
    Builds a FAISS vector store from the processed document chunks.
    ...
    """
    vector_store = FAISS.from_documents(_train_chunks, _embeddings)
    return vector_store


@st.cache_resource
def get_llm():
    """
    Loads a Hugging Face text generation model pipeline as an LLM interface.
    ...
    """
    hf_pipeline = pipeline(
        "text-generation",
        model="bigscience/bloom-560m",
        max_length=350,
        do_sample=False,  # Reduce randomness
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


# ------------------------------------------------------------
#                   MAIN APP EXECUTION
# ------------------------------------------------------------


train_chunks = load_and_process_data()
embeddings = get_embeddings()
vector_store = build_vector_store(train_chunks, embeddings)
llm = get_llm()

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


# ------------------------------------------------------------
#                   STREAMLIT UI
# ------------------------------------------------------------


st.title("🧠 Melini — Your Wikipedia Insight")

st.markdown(
    """
    🧠 **Meet Melini, your Wikipedia Fact Finder!**

    Got a question? Melini is a specialized companion built on Wikipedia data. 
    It **intelligently searches** through thousands of indexed passages, **retrieves** the most relevant context, 
    and delivers a **clear, synthesized answer** based on factual data. 
    Ask anything and let Melini dig into the facts. 🔍
    """
)

st.sidebar.title("ℹ️ About Melini")
st.sidebar.markdown(
    """
    - Powered by Wikipedia 
    - Uses Hugging Face embeddings + Bloom model 
    - Retrieves the top 3 most relevant passages for context 
    - Answers in natural, easy-to-read language 
    """
)

st.success(
    f"✅ **Knowledge Base Loaded Successfully!** Wikipedia data has been split into a total of **{len(train_chunks):,} chunks** and indexed into the vector database. Melini is now ready to answer your questions."
)
st.markdown("---")

# User query input
query = st.text_input("Your question:")

if query:
    # Show progress while generating the response
    with st.spinner("Retrieving context and generating answer..."):
        result = qa_chain.invoke({"query": query})

    raw_text = result["result"].strip()

    if "Helpful Answer:" in raw_text:
        answer_text = raw_text.split("Helpful Answer:")[1].strip().replace("**", "")
        info_text = raw_text.split("Question:")[0].strip().replace("**", "")
    else:
        answer_text = raw_text

    st.subheader("Answer")

    # Highlight the extracted answer
    st.success(f"**{answer_text}**")

    st.markdown("---")

    st.subheader("Information")
    # Highlight the extracted answer
    st.markdown(info_text)

    # Display the retrieved source documents as before
    st.subheader("Sources")
    for i, doc in enumerate(result["source_documents"]):
        with st.expander(f"Source Document {i+1}"):
            st.write(doc.page_content)

st.caption(
    "⚠️ For Experimental Purposes Only. This AI assistant sources information from Wikipedia. As a language model, it can make mistakes. Please double-check critical information from reliable sources."
)
