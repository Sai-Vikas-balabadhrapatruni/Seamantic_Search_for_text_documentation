import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2
from gensim.models import Word2Vec
from elasticsearch import Elasticsearch

nltk.download('punkt')
nltk.download('stopwords')

def load_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_document(document):
    tokens = word_tokenize(document.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

def train_word2vec_model(preprocessed_documents):
    model = Word2Vec(sentences=preprocessed_documents, vector_size=200, window=10, min_count=10, workers=4)
    #model.save("word2vec_model.model")
    return model

def create_index(es, first_word, word2vec_model):
    index_name = 'technical_document'
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Skipping index creation.")
        return
    index_mapping = {
    'mappings': {
        'properties': {
            'title': {'type': 'text'},
            'content': {'type': 'text'},
            'embedding': {'type': 'dense_vector', 'dims': len(word2vec_model.wv[first_word])} 
            }
        }
    }
    #this line should be executed only at start. Once its created this is not needed
    es.indices.create(index='technical_document', body=index_mapping)

def index_documents(es, preprocessed_document, word2vec_model, first_word):
    for i, doc in enumerate(preprocessed_document):
        # Use the Word2Vec model to get the embedding for an example word
        embedding = list(word2vec_model.wv[first_word])
        # Index a document with title, content, and embedding
        indexed_doc = {
            'title': f'Document {i + 1}',
            'content': ' '.join(doc),
            'embedding': embedding
        }
        # Index the document in Elasticsearch
        es.index(index='technical_document', body=indexed_doc)

def preprocess_query(query):
    # Tokenize, remove stop words, and perform stemming
    tokens = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    preprocessed_query = [stemmer.stem(word) for word in filtered_tokens]
    return preprocessed_query

def get_query_embedding(word2vec_model, preprocessed_query):
    embedding = list(word2vec_model.wv[preprocessed_query[0]])  # Assuming the first word in the query for simplicity
    return embedding

def search_and_rank(es, query_embedding):
    search_query = {
        'query': {
            'script_score': {
                'query': {'match_all': {}},
                'script': {
                    'source': 'cosineSimilarity(params.query_vector, doc["embedding"]) + 1.0',
                    'params': {'query_vector': query_embedding}
                }
            }
        }
    }
    print(search_query)
    results = es.search(index='technical_document', body=search_query)
    ranked_documents = results['hits']['hits']
    return ranked_documents


def main():
    pdf_path = '.\Technical_report.pdf'
    pdf_text = load_pdf(pdf_path)
    preprocessed_document = preprocess_document(pdf_text)
    word2vec_model = train_word2vec_model(preprocessed_document)
    first_word = list(word2vec_model.wv.index_to_key)[0]
    es = Elasticsearch([{'host':'localhost','port':9200,'scheme': "https"}], basic_auth=('elastic', 'I2A-bp83GOgVKbJAdigi'), verify_certs=False, ssl_show_warn=False)
    create_index(es, first_word, word2vec_model)
    index_documents(es, preprocessed_document,word2vec_model, first_word)
    user_query = "GPT-4 benchmarks"
    preprocessed_query = preprocess_query(first_word)
    query_embedding = get_query_embedding(word2vec_model, preprocessed_query)
    ranked_documents = search_and_rank(es, query_embedding)
    for i, doc in enumerate(ranked_documents, start=1):
        print(f"Rank {i}: Document Title - {doc['_source']['title']}")

if __name__ == "__main__":
    main()