from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
from scipy import sparse

# Leer el dataset limpio
data_cleaned = pd.read_csv('../../data/processed/movies_cleaned.csv')

# Unir las columnas relevantes en un solo campo para vectorización
data_cleaned['combined_features'] = data_cleaned['synopsis'] + " " + data_cleaned['genre'] + " " + data_cleaned['director'] + " " + data_cleaned['writer'] + " " + data_cleaned['crew']

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Vectorizar el texto combinado
tfidf_matrix = tfidf_vectorizer.fit_transform(data_cleaned['combined_features'])

# Guardar el vectorizador TF-IDF
joblib.dump(tfidf_vectorizer, '../../neural_network/models/tfidf_vectorizer.pkl')

# Guardar la matriz TF-IDF esparsa
sparse.save_npz('../../neural_network/models/tfidf_matrix.npz', tfidf_matrix)

print(tfidf_matrix)