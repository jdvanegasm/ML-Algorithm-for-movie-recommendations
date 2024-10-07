import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy import sparse

# Agregar la parte para guardar la matriz de similitud del coseno
def train_model(tfidf_matrix):
    """
    Entrena el modelo calculando la similitud del coseno entre las películas.
    
    :param tfidf_matrix: Matriz TF-IDF con las características vectorizadas de las películas.
    :return: Matriz de similitud del coseno.
    """
    # Calcular la similitud del coseno entre todas las películas
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Guardar la matriz de similitud del coseno
    joblib.dump(cosine_sim, '../../neural_network/models/cosine_sim.pkl')
    
    return cosine_sim

def load_data_and_vectorize():
    """
    Carga los datos limpios y la matriz TF-IDF ya vectorizada.
    
    :return: DataFrame de datos limpios, matriz TF-IDF.
    """
    # Cargar el dataset limpio
    data_cleaned = pd.read_csv('../../data/processed/movies_cleaned.csv')
    
    # Cargar la matriz TF-IDF guardada
    tfidf_matrix = sparse.load_npz('../../neural_network/models/tfidf_matrix.npz')
    
    return data_cleaned, tfidf_matrix

def get_recommendations(title, cosine_sim, data_cleaned):
    """
    Obtiene las recomendaciones de películas similares basadas en la similitud del coseno.
    
    :param title: Título de la película de entrada.
    :param cosine_sim: Matriz de similitud del coseno.
    :param data_cleaned: DataFrame con los datos limpios de las películas.
    :return: Lista de títulos de películas recomendadas.
    """
    try:
        # Obtener el índice de la película que coincide con el título
        idx = data_cleaned[data_cleaned['title'].str.lower() == title.lower()].index[0]

        # Obtener las puntuaciones de similitud para todas las películas con respecto a la película dada
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ordenar las películas en función de la similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtener los índices de las películas más similares
        sim_scores = sim_scores[1:11]  # Excluir la propia película

        # Obtener los títulos de las películas recomendadas
        movie_indices = [i[0] for i in sim_scores]
        recommendations = data_cleaned['title'].iloc[movie_indices]

        # Convertir a un conjunto para eliminar duplicados y luego a lista
        recommendations = list(dict.fromkeys(recommendations))
        
        return recommendations
    
    except IndexError:
        return f"'{title}' no se encontró en la base de datos."

# Ejecución del entrenamiento y prueba del modelo
if __name__ == "__main__":
    data_cleaned, tfidf_matrix = load_data_and_vectorize()
    cosine_sim = train_model(tfidf_matrix)
    # Probar el sistema de recomendación
    print(get_recommendations("Mad Max: Fury Road", cosine_sim, data_cleaned))
