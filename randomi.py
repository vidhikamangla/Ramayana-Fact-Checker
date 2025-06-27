
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#using this to prevent the tokenizer warning 

import requests
import json
import re
import unicodedata

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string

#we are using amberroad model which is a BERT-based reranker model for semantic passage scoring
#to get the verses that are most relevent to the translation that user inputs
reranker_model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

import requests

# #downloading the dataset from github link
# github_url = 'https://github.com/vidhikamangla/Ramayana-Fact-Checker/blob/main/valmiki_ramayana_dataset.xlsx' 
# local_filename = 'valmiki_ramayana_dataset.xlsx'
# response = requests.get(github_url)
# if response.status_code == 200:
#     with open(local_filename, 'wb') as f:
#         f.write(response.content)
#     print('File downloaded successfully')
# else:
#     print(f'Failed to download file, status code: {response.status_code}')

#text normalising part:
# name aliases for eacy text processing 
VARIANT_MAP = {
    "rama": ["raghu", "raghava", "ram", "raghunatha", "dasharathi", "ramachandra", "prince rama", "lord rama"],
    "bharata": ["bharatha", "bharat", "prince bharata"],
    "ravana": ["dashanan", "lankesh", "ravanaasura", "demon king", "demon king ravana"],
    "hanuman": ["anjaneya", "hanuma", "maruti", "pawanputra", "bajrangbali", "monkey warrior"],
    "sita": ["seetha", "janaki", "vaidehi", "maithili", "princess sita"],
    "lakshmana": ["lakshman", "saumitri", "prince lakshmana"],
    "sugriva": ["monkey king", "vanara king"],
    "dasaratha": ["dasharath", "king dasaratha", "raja dasaratha"],
    "kumbhakarna": ["kumbhakaran", "giant demon"]
}
# changing all the name variants to their standard (canonical) names
def build_flat_variant_map(variant_map):
    flat_map = {}
    for canonical, variants in variant_map.items():
        for v in variants:
            flat_map[v.lower()] = canonical
        #mapping the canonical name to itself j in case
        flat_map[canonical.lower()] = canonical
    return flat_map

FLAT_VARIANT_MAP = build_flat_variant_map(VARIANT_MAP)

def normalize(text):
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    # Remove quotes not part of contractions or possessives
    text = re.sub(r"(?<=\s)'|'(?=\s)|^'|'$", "", text)
    # Remove unwanted punctuation, keeping . , / - ' : ; ? ! " ( ) and apostrophes in contractions/in-text
    text = re.sub(r"[^\w\s'.,/\-:;?!\()]", "", text)
    # Normalize ellipses
    text = re.sub(r"\.{2,}", ".", text)
    # Remove punctuation at start/end if isolated
    text = re.sub(r"^[.,\s]", "", text)
  
    # Replace all variants with canonical forms
    # Sort variants by length (longest first) to avoid partial replacement
    for variant, canonical in sorted(FLAT_VARIANT_MAP.items(), key=lambda x: -len(x[0])):
        # Use word boundaries and allow for multi-word variants
        pattern = re.compile(rf"\b{re.escape(variant)}\b", flags=re.IGNORECASE)
        text = pattern.sub(canonical, text)
    return text

#this is the function that calculates a relevance score between the user
#input and a passage using the BERT reranker (we are using amberroad)
def rerank_score(query, passage):
    query=normalize(query)
    inputs = reranker_tokenizer(query, passage, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = reranker_model(**inputs)
    score = outputs.logits[0, 1].item()
    return score

#now this function will score and sort retrieved documents by relevance
#it will returning the top_k most relevant docs for which we have set the value as 6
#so we get top 6 reranked verses
def rerank_documents(query, docs, top_k=6):
    query=normalize(query)
    scored = []
    for doc in docs:
        passage = doc.page_content
        score = rerank_score(query, passage)
        scored.append((score, doc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored[:top_k]]

#this function is used to load the Ramayana verses and 
#metadata from an excel file into document objects.
#as metadeta we have added the reference of the verse
def load_excel_dataset(file_path):
    df = pd.read_excel(file_path)
    documents = []
    for _, row in df.iterrows():
        content = normalize(row['English Translation'])
        metadata = {
            'book': row['Kanda/Book'],
            'chapter': str(row['Sarga/Chapter']),
            'verse': str(row['Shloka/Verse Number'])
        }
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"Loaded {len(documents)} documents from Excel.")
    return documents

#this splits each verse into smaller chunks for better semantic search
#chunks are also overlapping
def chunk_docs_by_verse(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", ".", "!", "?"]
    )
    chunked_docs = []
    for doc in docs:
        splits = splitter.split_text(normalize(doc.page_content))
        for i, chunk in enumerate(splits):
            metadata = dict(doc.metadata)
            metadata['chunk_index'] = i
            chunked_docs.append(Document(page_content=chunk, metadata=metadata))
    print(f"Chunked documents from {len(docs)} to {len(chunked_docs)} smaller pieces.")
    return chunked_docs

#this function embeds and stores all verse chunks in a vector database for FASTER semantic search
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
            print('creating vector db')
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

#thi fxn sets up a retriever that uses MMR that is maximal marginal relevance for 
#diverse and relevant search results
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

def hybrid_retrieve(query, vector_db, bm25, chunked_docs, top_k=20, rrf_k=60):
    # Dense retrieval
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k*2, "lambda_mult": 0.6}
    )
    dense_results = retriever.invoke(query)

    def find_doc_index(doc, doc_list):
        for i, d in enumerate(doc_list):
            if d.page_content == doc.page_content and d.metadata == doc.metadata:
                return i
        return -1

    dense_indices = []
    for doc in dense_results:
        idx = find_doc_index(doc, chunked_docs)
        if idx != -1:
            dense_indices.append(idx)

    # Sparse retrieval (BM25)
    query_tokens = [token for token in word_tokenize(query.lower()) if token not in string.punctuation]
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # Reciprocal Rank Fusion (RRF)
    def rrf_score(rank, k=rrf_k):
        return 1.0 / (rank + k)

    rrf_scores = {}
    for rank, idx in enumerate(dense_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score(rank)
    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score(rank)

    merged_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
    merged_docs = [chunked_docs[i] for i in merged_indices]
    return merged_docs

#now this function prepares the LLM prompt template and it also defines the fact-checking chain
def chatPrompt(vector_db, bm25, chunked_docs, llm):
    try:
        template = """
You are a fact-checker for ancient texts. Only respond in JSON format.

Given:
- CONTEXT: 6 verses from English translations from the Ramayana 
- TRANSLATION: A user-submitted sentence

Your Task:
1. Determine whether the TRANSLATION is true by meaning from any verse from the 6 given verses of the CONTEXT, check all the 6 verses.
2. Do not assume something is true just because it is not mentioned. Only answer TRUE if the context explicitly supports the translation and does not contradict it. If any verse in the CONTEXT explicitly contradicts the TRANSLATION, return FALSE, give the contradicting verse in "Reason".
3. If any verse in the CONTEXT explicitly contradicts the TRANSLATION, the answer must be FALSE. Give the contradicting line reason in the Reason.
4. Internally, compare each verse line-by-line with the TRANSLATION, to make correct prediction before making a decision, but only output the final JSON. even if there's a hint of the correct prediction dont ignore the events.
5. Minor spelling mistakes (like seeta and sita are the same)
6. Minor differences in wording are acceptable, if the overall gist of 6 verses is the same. Minor differences in names or wording are acceptable if the overall meaning is preserved. Major factual errors, such as incorrect relationships or events, are not acceptable.
7. If true, extract the exact book, chapter, and verse from the matching CONTEXT's metadata.
8. If the translation is factually incorrect (e.g., wrong names, false claims), return false.
9. When answering, always explain exactly what from the CONTEXT support your answer in the "Reason" field.

if irrelevent names and situations are used that are not even related a little to ANY given verse, return THIS JSON format strictly:

{{
  "Answer": "NOT RELEVANT",
  "Reason": "irrelevent"
}} (not the book name chapter name and verse name)

in other condition: 
Respond strictly in this JSON format: if it is true:
{{
  "Answer": true,
  "Reason":"<one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse. >",
  "Reference": "<Kanda/Book name from metadata, Sarga/Chapter from metadata, Shloka/Verse from metadata>" (double quotes in the start and end of the whole reference. just give the reference info from metadata.)
}}

Return only the JSON. then one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse. 

if false, return THIS JSON format strictly:
{{
  "Answer": false,
  "Reason":"<one line explanation why the translation is true or false by also stating what part of the verse states that our translation is true and mention that line from the verse.>",
  "Correction": the reason why the translation is false and the correction
  "Reference": "<Kanda/Book name from metadata, Sarga/Chapter from metadata>, Shloka/Verse from metadata>" (double quotes in the start and end of the whole reference, end with a curly bracket for json format)
}}
Hence ur final answer must Return only one JSON.
---

CONTEXT:
{context}

TRANSLATION:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        def run_chain(question):
            
            #this will retrieve relevant verse chunks for the input user's translation
            # docs = retriever.invoke(question)
            docs = hybrid_retrieve(question, vector_db, bm25, chunked_docs, top_k=20)
            if not docs:
                return '{"Answer": NOT RELEVANT, "Book": "", "Chapter": "", "Verse": ""}'

            #reranking the docs using the BERT reranker for best matches as we have defined above
            reranked_docs = rerank_documents(question, docs, top_k=6)
            
            #we have printed out the reranked docs for debugging purposes for ourselves
            for i, d in enumerate(reranked_docs):
                print(f"Reranked doc {i+1}: {d.page_content[:100]}... (Book: {d.metadata.get('book')}, Chapter: {d.metadata.get('chapter')}, Verse: {d.metadata.get('verse')})")

            #now we are building the context string for the LLM prompt
            context_text = "\n\n".join(
                f"{doc.page_content} (Book: {doc.metadata.get('book')}, Chapter: {doc.metadata.get('chapter')}, Verse: {doc.metadata.get('verse')})"
                for doc in reranked_docs
            )
            
            #again, we have printed the context text to debug
            print("texts beings sent to prompt: ",context_text)
            prompt_input = prompt.format(context=context_text, question=question)
            response = llm.invoke(prompt_input)
            
            #debug printing
            print("Raw LLM response:", response)
            return response

        print('Chat prompt chain ready.')
        return run_chain
    except Exception as e:
        print("Prompt creation failed:", e)
        return None

# doc = './ramayana_dataset.txt'

#we are using this model from ollama which is available for free
model = 'llama3.2:latest'

def main():
    print('📜 Ramayana Translation Checker')
    llm = ChatOllama(model=model, temperature=0)
    
    #our final csv containing all the verses
    df = pd.read_csv('Valmiki Ramanaya Verses - Final Evaluation.csv') 

    #extracting the 'Statement' column to get a list of all the verses
    statements = df['Statement'].tolist()
    # statements=["ba ba black sheep"]
    # statements=["Bharata accepted the throne without any longing for Ramaâ€™s return."]
    # statements= ["Sita is found by king janaka while he is ploughing the field."]
    print(f"Fact checking {len(statements)} verses given by user...")
    
    #looading the Ramayana vector database
    vector_db_load = embed_excel_chunks()
    
    if vector_db_load is None:
        print("failed to load vector data.")
        return
    
    docs = load_excel_dataset('./valmiki_ramayana_dataset.xlsx')
    chunked_docs = chunk_docs_by_verse(docs)

    # Build BM25 index
    bm25_corpus = [
        [token for token in word_tokenize(doc.page_content.lower()) if token not in string.punctuation]
        for doc in chunked_docs
    ]
    bm25 = BM25Okapi(bm25_corpus)
    
    retriever = get_retriever(vector_db_load)
    if retriever is None:
        print("failed to initialize retriever.")
        return
    
    chain = chatPrompt(vector_db_load, bm25, chunked_docs, llm)
    if chain is None:
        print("chain creation failed.")
        return
    
    columns = ['ID', 'Statement', 'Prediction', 'Reference','Correction']
    df = pd.DataFrame(columns=columns)
    
    #processing each of the verses separately
    for i,input in enumerate(statements):
        input=normalize(input)
        print(f"🔍 Processing translation {i+1}: \n{input}\n......")
        
        #result for each verse:
        response = chain(input)
        print(response)
        json_str = response.content
        print(json_str)
        
        #we have fixed the common JSON formatting issues from LLM output  
        #quoting the reason value if it is not quoted
        json_str_fixed = re.sub(
            r'("Reason": )([^\n"]+)', 
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"', 
            json_str
        )
        
        #quoting the correction value if it is not quoted
        json_str_fixed = re.sub(
            r'("Correction": )([^\n"][^\n]*)(?=\n)',  # not starting with quote, up to line end
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"',
            json_str_fixed
        )
    #     # print(json_str_fixed)   
        
        #adding a comma after reason value if other attributes also exist
        if '"Book"' in json_str_fixed and '"Verse"' in json_str_fixed:
            json_str_fixed = re.sub(
                r'("Reason":\s*"[^"]*")\s*\n(\s*"[A-Za-z]+":)', 
                r'\1,\n\2', 
                json_str_fixed
            )
            
        #adding a comma after correction value too if other attributes also exist
        if '"Correction"' in json_str_fixed:
            # print("Checkpoint pass.")
            
            json_str_fixed = re.sub(
                r'("Correction":\s*"[^"]*")\s*\n(\s*"[A-Za-z]+":)', 
                r'\1,\n\2', 
                json_str_fixed
            )
            
        if not json_str_fixed.strip().endswith('}'):
            json_str_fixed = json_str_fixed.strip() + '}'

        print(json_str_fixed) 
        
        #now converting the formatted result to json format
        try:
            result = json.loads(json_str_fixed)
            if 'Answer' in result:
                result['Prediction'] = result.pop('Answer')
                result['ID'] = i+1
                result['Statement'] = input
        except Exception as e:
            print(f"Could not parse content: {e}")
            result = {}
            result['ID'] = i+1
            result['Statement'] = input
            if 'true' in json_str_fixed:
                result['Prediction'] = True
            if 'false' in json_str_fixed:
                result['Prediction'] = False
            
        #adding all the keys that match dataframe columns exactly
        row = {col: result[col] for col in columns if col in result}
        new_row_df = pd.DataFrame([row])

        #appending this row to the df
        df = pd.concat([df, new_row_df], ignore_index=True)
    print(df)
    df.to_csv('output_trial4.csv', index=False)
   
if __name__ == "__main__":
    main()