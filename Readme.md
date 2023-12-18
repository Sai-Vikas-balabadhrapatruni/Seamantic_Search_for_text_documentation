This Python script utilizes natural language processing and word embeddings to create a document search system with Elasticsearch. It includes functionality to index and search technical documents based on their content and embeddings.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- NLTK (Natural Language Toolkit)
- PyPDF2
- Gensim
- Elasticsearch
- Elasticsearch Python client

You can install the required Python packages using:

```bash
pip install nltk PyPDF2 gensim elasticsearch
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Sai-Vikas-balabadhrapatruni/Seamantic_Search_for_text_documentation.git
   ```

2. **Install Dependencies:**

   ```bash
   cd your-repository
   pip install numpy pandas nltk tensorflow elasticsearch gensim
   ```
   Follow the ElasticSearch Guide (https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) to install it on the machine 

3. **Configure Elasticsearch:**

   Ensure that you have an Elasticsearch instance running. Modify the Elasticsearch connection details in the `main()` function:

   ```python
   es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'https'}], basic_auth=('elastic', 'your-password'), verify_certs=False, ssl_show_warn=False)
   ```

4. **Run the Script:**

   Execute the script by running:

   ```bash
   python your_script_name.py
   ```

   This will load a sample PDF document (`Technical_report.pdf`), preprocess it, train a Word2Vec model, and index it in Elasticsearch. It then performs a sample search using the query "GPT-4 benchmarks."

## Customization

- **PDF Input:**
  - Change the `pdf_path` variable in the `main()` function to point to your desired PDF document.

- **Word Embedding Model:**
  - Adjust the parameters in the `train_word2vec_model` function for the Word2Vec model according to your preferences.

- **Search Query:**
  - Modify the `user_query` variable in the `main()` function to test different search queries.