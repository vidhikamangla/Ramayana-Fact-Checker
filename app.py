# # from langchain.retrievers.multi_query import MultiQueryRetriever
# # from langchain_core.prompts import PromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.runnables import RunnablePassthrough
# # from langchain_experimental.text_splitter import SemanticChunker

# from langchain_community.document_loaders import TextLoader
# from langchain_chroma.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import streamlit as st
# import pandas as pd
# import torch

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# reranker_model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
# reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
# reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

# def rerank_score(query, passage):
#     inputs = reranker_tokenizer(query, passage, return_tensors='pt', truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = reranker_model(**inputs)
#     score = outputs.logits[0, 1].item()
#     return score


# def rerank_documents(query, docs, top_k=5):
#     scored = []
#     for doc in docs:
#         passage = doc.page_content
#         score = rerank_score(query, passage)
#         scored.append((score, doc))
#     scored.sort(reverse=True, key=lambda x: x[0])
#     return [doc for score, doc in scored[:top_k]]

# def load_excel_dataset(file_path):
#     df = pd.read_excel(file_path)
#     documents = []
#     for _, row in df.iterrows():
#         content = row['English Translation']
#         metadata = {
#             'book': row['Kanda/Book'],
#             'chapter': str(row['Sarga/Chapter']),
#             'verse': str(row['Shloka/Verse Number'])
#         }
#         documents.append(Document(page_content=content, metadata=metadata))
#     print(f"Loaded {len(documents)} documents from Excel.")
#     return documents

# def chunk_docs_by_verse(docs):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,
#         chunk_overlap=50,
#         separators=["\n", ".", "!", "?"]
#     )
#     chunked_docs = []
#     for doc in docs:
#         splits = splitter.split_text(doc.page_content)
#         for i, chunk in enumerate(splits):
#             metadata = dict(doc.metadata)
#             metadata['chunk_index'] = i
#             chunked_docs.append(Document(page_content=chunk, metadata=metadata))
#     print(f"Chunked documents from {len(docs)} to {len(chunked_docs)} smaller pieces.")
#     return chunked_docs

# @st.cache_resource
# def embed_excel_chunks():
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         if os.path.exists('./chroma_store'):
#             vector_db = Chroma(
#                 collection_name='ramayana-col',
#                 embedding_function=embeddings,
#                 persist_directory='./chroma_store'
#             )
#             print('existing vector db already exists')
#             return vector_db
#         else:
#             docs = load_excel_dataset('./valmiki_ramayana_dataset.xlsx')
#             chunked = chunk_docs_by_verse(docs)
#             vector_db = Chroma.from_documents(
#                 documents=chunked,
#                 collection_name='ramayana-col',
#                 embedding=embeddings,
#                 persist_directory='./chroma_store'
#             )
#             print('Embedded and stored new documents in the database')
#             return vector_db
#     except Exception as e:
#         print("error: ", e)
#         return None

# def get_retriever(vector_db):
#     if vector_db is None:
#         print("no db exists")
#         return None
#     retriever = vector_db.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 20, "fetch_k": 30, "lambda_mult": 0.6}
#     )
#     print('created retriever...')
#     return retriever

# def chatPrompt(retriever, llm):
#     try:
#         template = """
# You are a fact-checker for ancient texts. Only respond in JSON format.

# Given:
# - CONTEXT: 5 verses from English translations from the Ramayana 
# - TRANSLATION: A user-submitted sentence

# Your Task:
# 1. Determine whether the TRANSLATION is true by meaning from any verse from the 5 given of the CONTEXT, check all the 5 verses. 
# 2. Minor differences in wording are acceptable **if the meaning is the same**.
# 3. If true, extract the exact book, chapter, and verse from the matching CONTEXT's metadata.
# 4. If the translation is factually incorrect (e.g., wrong names, false claims), return true.

# Respond strictly in this JSON format:
# {{
#   "Answer": true or false,
#   "Book": "<Kanda/Book name from metadata>",
#   "Chapter": "<Sarga/Chapter from metadata>",
#   "Verse": "<Shloka/Verse from metadata>"
# }}

# Return only the JSON. then one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse.
# ---

# CONTEXT:
# {context}

# TRANSLATION:
# {question}
# """
#         prompt = ChatPromptTemplate.from_template(template)

#         def run_chain(question):
#             docs = retriever.get_relevant_documents(question)
#             if not docs:
#                 return '{"Answer": false, "Book": "", "Chapter": "", "Verse": ""}'

#             reranked_docs = rerank_documents(question, docs, top_k=5)

#             for i, d in enumerate(reranked_docs):
#                 print(f"Reranked doc {i+1}: {d.page_content[:100]}... (Book: {d.metadata.get('book')}, Chapter: {d.metadata.get('chapter')}, Verse: {d.metadata.get('verse')})")

#             context_text = "\n\n".join(
#                 f"{doc.page_content} (Book: {doc.metadata.get('book')}, Chapter: {doc.metadata.get('chapter')}, Verse: {doc.metadata.get('verse')})"
#                 for doc in reranked_docs
#             )
            
#             print("💗texts beings sent to prompt: ",context_text)

#             prompt_input = prompt.format(context=context_text, question=question)
#             # print("Prompt sent to LLM:", prompt_input[:500], "...")
#             print("💗Prompt sent to LLM:", prompt_input)
#             print("🎀")
#             response = llm.invoke(prompt_input)
#             print("Raw LLM response:", response)
#             return response

#         print('Chat prompt chain ready.')
#         return run_chain
#     except Exception as e:
#         print("Prompt creation failed:", e)
#         return None

# # doc = './ramayana_dataset.txt'
# model = 'llama3.2:latest'

# def main():
#     st.title('📜 Ramayana Translation Checker')
#     llm = ChatOllama(model=model, temperature=0)
#     st.write('Welcome! Enter your translation to check if it’s faithful to the original Ramayana.')
#     input = st.text_input("🔍 Enter the translation to verify:")
    
#     if input:
#         vector_db_load = embed_excel_chunks()
#         if vector_db_load is None:
#             st.write("failed to load vector data.")
#             return
#         retriever = get_retriever(vector_db_load)
#         if retriever is None:
#             st.write("failed to initialize retriever.")
#             return
#         chain = chatPrompt(retriever, llm)
#         if chain is None:
#             st.write("chain creation failed.")
#             return
#         response = chain(input)
#         st.code(response, language="json")
#     else:
#         st.info('ENTER TRANSLATION')

# if __name__ == "__main__":
#     main()

# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.document_loaders import TextLoader
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        search_kwargs={"k": 20, "fetch_k": 30, "lambda_mult": 0.6}
    )
    print('created retriever...')
    return retriever

def chatPrompt(retriever, llm):
    try:
        template = """
You are a fact-checker for ancient texts. Only respond in JSON format.

Given:
- CONTEXT: 5 verses from English translations from the Ramayana 
- TRANSLATION: A user-submitted sentence

Your Task:
1. Determine whether the TRANSLATION is true by meaning from any verse from the 5 given of the CONTEXT, check all the 5 verses. 
2. If true, extract the exact book, chapter, and verse from the matching CONTEXT's metadata.
3. If the translation is factually incorrect (e.g., wrong names, false claims), return false.

condition 1(irrelevence):
if irrelevent names and situations are used that are not used in ANY given verse, example a random name like riya or alice etc that means it is FALSE. but minor difference in the name and minor spelling mistake is TRUE, jump to other conditions. return THIS JSON format strictly:

{{
  "Answer": false
  "Reason":one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse.
}} (not the book name chapter name and verse name)

in other condition: 
Respond strictly in this JSON format:
{{
  "Answer": true or false,
  "Reason":one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse.
  "Book": "<Kanda/Book name from metadata>",
  "Chapter": "<Sarga/Chapter from metadata>",
  "Verse": "<Shloka/Verse from metadata>"
}}

Return only the JSON.  
if false, return THIS JSON format strictly:
{{
  "Answer": false,
  "Reason":one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse.
  "Correction": the reason why the translation is false and the correction
  "Book": "<Kanda/Book name from metadata>",
  "Chapter": "<Sarga/Chapter from metadata>",
  "Verse": "<Shloka/Verse from metadata>"
}}

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
                print(f"Reranked doc {i+1}: {d.page_content[:100]}... (Book: {d.metadata.get('book')}, Chapter: {d.metadata.get('chapter')}, Verse: {d.metadata.get('verse')})")

            context_text = "\n\n".join(
                f"{doc.page_content} (Book: {doc.metadata.get('book')}, Chapter: {doc.metadata.get('chapter')}, Verse: {doc.metadata.get('verse')})"
                for doc in reranked_docs
            )
            
            print("💗texts beings sent to prompt: ",context_text)

            prompt_input = prompt.format(context=context_text, question=question)
            # print("Prompt sent to LLM:", prompt_input[:500], "...")
            print("💗Prompt sent to LLM:", prompt_input)
            print("🎀")
            response = llm.invoke(prompt_input)
            print("Raw LLM response:", response)
            return response

        print('Chat prompt chain ready.')
        return run_chain
    except Exception as e:
        print("Prompt creation failed:", e)
        return None

# doc = './ramayana_dataset.txt'
model = 'llama3.2:latest'

import json
import re

def main():
    st.title('📜 Ramayana Translation Checker')
    llm = ChatOllama(model=model, temperature=0)
    st.write('Welcome! Enter your translation to check if it’s faithful to the original Ramayana.')
    input = st.text_input("🔍 Enter the translation to verify:")
    
    if input:
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
        response = chain(input)
        # st.code(response, language="json")
        print("🧸🧸🧸🧸")
        print(response)
        print("🧸🧸🧸🧸")
        print(type(response))
        print("🧸🧸🧸🧸")
        json_str = response.content
        print(json_str)
        print("🧸🧸🧸🧸")
        json_str_fixed = re.sub(
            r'("Reason": )([^\n"]+)', 
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"', 
            json_str
        )
        print(json_str_fixed)
        print("🧸🧸🧸🧸")
        try:
            result = json.loads(json_str_fixed)
        except Exception:
            try:
                # As a last resort, try ast.literal_eval (handles Python dicts)
                result = ast.literal_eval(json_str_fixed)
            except Exception as e:
                st.error(f"Could not parse content: {e}")
                result = {}
        
        st.markdown("### Result")
        if result:
            st.markdown(f"**Answer:** {result.get('Answer', '')}")
            st.markdown(f"**Reason:** {result.get('Reason', '')}")
        else:
            st.info("No valid result to display.")
        
    else:
        st.info('ENTER TRANSLATION')

if __name__ == "__main__":
    main()