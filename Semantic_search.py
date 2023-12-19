import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from gensim.models import Word2Vec
from elasticsearch import Elasticsearch
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if pd.isna(text):
        return 'N/A'
    text = text.replace('\r\n', ' ').replace('\n', ' ').strip()
    text = ' '.join(text.split())
    return text if text else 'N/A'

def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        # Apply the cleaning function to the necessary columns
        data['JD'] = data['JD'].apply(clean_text)
        data['Qualification:'] = data['Qualification:'].apply(clean_text)
        data['Location'] = data['Location'].apply(clean_text)
        # Add other columns to clean if needed
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def preprocess_document(document):
    try:
        tokens = word_tokenize(document.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        return stemmed_tokens
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return []

def train_word2vec_model(preprocessed_documents):
    try:
        model = Word2Vec(sentences=preprocessed_documents, vector_size=300, window=5, min_count=1, workers=4)
        return model
    except Exception as e:
        print(f"Error training Word2Vec model: {e}")
        return None

def setup_elasticsearch():
    try:
        es = Elasticsearch([{'host':'localhost','port':9200,'scheme': "https"}], basic_auth=('elastic', 'I2A-bp83GOgVKbJAdigi'), 
                       verify_certs=False, ssl_show_warn=False, request_timeout=30)
        return es
    except Exception as e:
        print(f"Error setting up Elasticsearch: {e}")
        return None

def create_index(es, index_name):
    index_settings = {
        "settings": {},
        "mappings": {
            "properties": {
                "job_name": {"type": "text"},
                "job_description": {"type": "text"},
                "company_name": {"type": "text"},
                "location":  {"type": "text"},
                "processed_jd": {"type": "text"}
            }
        }
    }
    try:
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=index_settings)
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
    except Exception as e:
        print(f"Error creating index: {e}")

def index_data(es, data, index_name):
    for i, row in data.iterrows():
        try:
            es.index(index=index_name, id=i, body={
                "job_name": row['Job Name'],
                "job_description": row['JD'],
                "company_name": row['Company Name'],
                "location": row['Location'],
                "processed_jd": ' '.join(preprocess_document(row['JD']))
            })
        except Exception as e:
            print(f"Error indexing document {i}: {e}")

def semantic_search(es, word2vec_model, query, top_n=10):
    processed_query = preprocess_document(query)
    #print(f"Processed Query: {processed_query}")
    
    query_vector = np.mean([word2vec_model.wv[word] for word in processed_query if word in word2vec_model.wv], axis=0)
    response = es.search(index='job_listings', body={
        "query": {
            "match": {
                "job_description": query
            }
        }
    }, size=top_n)

    #print(f"Number of hits: {len(response['hits']['hits'])}")

    results = []
    for hit in response['hits']['hits']:
        doc_vector = np.mean([word2vec_model.wv[word] for word in hit['_source']['processed_jd'].split() if word in word2vec_model.wv], axis=0)
        similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
        results.append((hit, similarity))
        #print(f"Hit: {hit['_source']['job_name']}, Similarity: {similarity}")
    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    file_path = './Scrapped_data.csv' 
    data = load_csv(file_path)
    if data is not None:
        preprocessed_data = [preprocess_document(doc) for doc in data['JD'].astype(str)]
        word2vec_model = train_word2vec_model(preprocessed_data)
        es = setup_elasticsearch()
        if es is not None:
            if es.indices.exists(index="job_listings"):
                    es.indices.delete(index="job_listings")
            create_index(es, "job_listings")
            index_data(es, data, "job_listings")
            while True:
                user_query = input("Enter your query (or type 'exit' to quit): ")
                if user_query.lower() == 'exit':
                    break
                search_results = semantic_search(es, word2vec_model, user_query)
                for hit, similarity in search_results:
                    print(f"Job Title: {hit['_source']['job_name']}, Company Name: {hit['_source']['company_name']}, Location: {hit['_source']['location']}, Similarity: {similarity}")
        else:
            print("Failed to setup Elasticsearch.")
