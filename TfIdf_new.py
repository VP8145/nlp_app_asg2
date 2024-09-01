import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK data
nltk.download('punkt')


given_corpus = ['The Renaissance was a period of great cultural change in Europe.',
                'Ancient Egypt is known for its pyramids and pharaohs.',
                'The Industrial Revolution marked a major turning point in history.',
                'World War II had a profound impact on global politics.',
                'The Silk Road was an ancient trade route connecting the East and West.']

query=['cultural changes in history']

# Define the tokenize and stem function
def tokenize_and_stem(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def generate_TFIDF(df):
    
    tfidf_vectorizer=TfidfVectorizer(use_idf=True,stop_words='english',tokenizer=tokenize_and_stem)
    #tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(df_cleaned)

    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(df)
    #print(tfidf_vectorizer_vectors)

    # Get the feature names (token words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Convert the sparse matrix to a dense matrix (or array)
    dense_matrix = tfidf_vectorizer_vectors.toarray()

    # Create a DataFrame to hold the tokens and their vectors
    tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)

    return tfidf_df


#dfc = pd.DataFrame(given_corpus)
#dfq = pd.DataFrame(query)


tfidf_doc_df = generate_TFIDF(given_corpus)
tfidf_doc_qy  = generate_TFIDF(query)

print('---------------- TD IDF vectors for given docuuments ----------------')
print(tfidf_doc_df)

print('---------------- TD IDF vectors for given docuumnets ----------------')
print(tfidf_doc_qy)

# Align the columns of both DataFrames
tfidf_doc_df, tfidf_doc_qy = tfidf_doc_df.align(tfidf_doc_qy, join='outer', axis=1, fill_value=0)

# Concatenate the aligned DataFrames
tfidf_combined_df = pd.concat([tfidf_doc_df, tfidf_doc_qy], ignore_index=True)

print(tfidf_combined_df)

print('--------------------- Cosine Similarity --------------------------')
# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_combined_df)

# Get similarity scores for the query document (last row)
query_similarity_scores = cosine_sim[-1][:-1]

# Retrieve indices of the top 2 most similar documents
top_2_indices = query_similarity_scores.argsort()[-2:][::-1]

# Finally printing top 2 matching documents for the given query
print("\nTop 2 matching documents:")
for index in top_2_indices:
    print(f"Document {index + 1}: {given_corpus[index]} (Score: {query_similarity_scores[index]:.4f})")