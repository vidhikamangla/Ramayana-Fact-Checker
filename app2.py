# from langchain_community.document_loaders import TextLoader
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_experimental.text_splitter import SemanticChunker
# :)
import streamlit as st

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
    
import asyncio
import sys

if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

reranker_model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

def rerank_score(query, passage):
    inputs = reranker_tokenizer(query, passage, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = reranker_model(**inputs)
    score = outputs.logits[0, 1].item()
    return score


def rerank_documents(query, docs, top_k=5):
    scored = []
    for doc in docs:
        passage = doc.page_content
        score = rerank_score(query, passage)
        scored.append((score, doc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored[:top_k]]


doc = './ramayana_dataset.txt'
model = 'llama3.2:latest'

def load_excel_dataset(file_path):
    df = pd.read_excel(file_path)
    documents = []
    for _, row in df.iterrows():
        content = row['English Translation']
        metadata = {
            'book': row['Kanda/Book'],
            'chapter': str(row['Sarga/Chapter']),
            'verse': str(row['Shloka/Verse Number'])
        }
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"Loaded {len(documents)} documents from Excel.")
    return documents

def chunk_docs_by_verse(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", ".", "!", "?"]
    )
    chunked_docs = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(splits):
            metadata = dict(doc.metadata)
            metadata['chunk_index'] = i
            chunked_docs.append(Document(page_content=chunk, metadata=metadata))
    print(f"Chunked documents from {len(docs)} to {len(chunked_docs)} smaller pieces.")
    return chunked_docs

@st.cache_resource
def embed_excel_chunks():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists('./chroma_store'):
            vector_db = Chroma(
                collection_name='ramayana-col',
                embedding_function=embeddings,
                persist_directory='./chroma_store'
            )
            print('existing vector db already exists')
            return vector_db
        else:
            docs = load_excel_dataset('./valmiki_ramayana_dataset.xlsx')
            chunked = chunk_docs_by_verse(docs)
            vector_db = Chroma.from_documents(
                documents=chunked,
                collection_name='ramayana-col',
                embedding=embeddings,
                persist_directory='./chroma_store'
            )
            print('Embedded and stored new documents in the database')
            return vector_db
    except Exception as e:
        print("error: ", e)
        return None

def get_retriever(vector_db):
    if vector_db is None:
        print("no db exists")
        return None
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "fetch_k": 20, "lambda_mult": 0.6}
    )
    print('created retriever...')
    return retriever

def chatPrompt(retriever, llm):
    try:
        template = """
You are a strict fact-checker for ancient texts. Only respond in JSON format.

Given:
- CONTEXT: English translations from the Ramayana (1 verse per chunk)
- TRANSLATION: A user-submitted sentence

Your Task:
1. Determine whether the TRANSLATION accurately reflects any part of the CONTEXT. 
2. Minor differences in wording are acceptable *if the meaning is the same*.
3. If the translation is factually incorrect (e.g., wrong names, false claims), return false.
4. If true, extract the exact book, chapter, and verse from the matching CONTEXT's metadata.

Respond strictly in this JSON format:
{{
  "Answer": true or false,
  "Book": "<Kanda/Book name from metadata>",
  "Chapter": "<Sarga/Chapter from metadata>",
  "Verse": "<Shloka/Verse from metadata>"
}}

Return only the JSON. No explanations.
---

CONTEXT:
{context}

TRANSLATION:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        def run_chain(question):
            docs = retriever.get_relevant_documents(question)
            if not docs:
                return '{"Answer": false, "Book": "", "Chapter": "", "Verse": ""}'

            reranked_docs = rerank_documents(question, docs, top_k=5)

            for i, d in enumerate(reranked_docs):
                print(f"Reranked doc {i}: {d.page_content[:100]}... (Book: {d.metadata.get('book')}, Chapter: {d.metadata.get('chapter')}, Verse: {d.metadata.get('verse')})")

            context_text = "\n\n".join(
                f"{doc.page_content} (Book: {doc.metadata.get('book')}, Chapter: {doc.metadata.get('chapter')}, Verse: {doc.metadata.get('verse')})"
                for doc in reranked_docs
            )

            prompt_input = prompt.format(context=context_text, question=question)
            print("Prompt sent to LLM:", prompt_input[:500], "...")
            response = llm.invoke(prompt_input)
            print("Raw LLM response:", response)

            return response

        print('Chat prompt chain ready.')
        return run_chain
    except Exception as e:
        print("Prompt creation failed:", e)
        return None

def main():
    st.title('📜 Ramayana Translation Checker')
    llm = ChatOllama(model=model, temperature=0)
    st.write('Welcome! Enter your translation to check if it’s faithful to the original Ramayana.')
    input1 = st.text_input("🔍 Enter the translation to verify:")

    if input1:
        vector_db_load = embed_excel_chunks()
        if vector_db_load is None:
            st.write("failed to load vector data.")
            return
        retriever = get_retriever(vector_db_load)
        if retriever is None:
            st.write("failed to initialize retriever.")
            return
        chain = chatPrompt(retriever, llm)
        if chain is None:
            st.write("chain creation failed.")
            return
        response = chain(input1)
        st.code(response, language="json")
    else:
        st.info('ENTER TRANSLATION')

if __name__ == "__main__":
    main()