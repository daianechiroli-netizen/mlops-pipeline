import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopy.distance


def predict(model: KMeans, scaler: StandardScaler, lat: float, lng: float) -> dict:
    """
    Dado um ponto (lat, lng), retorna o cluster mais próximo e a distância ao centróide.

    Passos esperados:
        1. Normalizar o ponto com o scaler recebido
        2. Predizer o cluster com o modelo
        3. Recuperar o centróide do cluster (invertendo a normalização)
        4. Calcular a distância geodésica em km entre o ponto e o centróide

    Parâmetros:
        model: modelo KMeans treinado
        scaler: StandardScaler usado no treino (para normalizar o ponto de entrada)
        lat: latitude do ponto a ser classificado
        lng: longitude do ponto a ser classificado

    Retorna:
        Dicionário com as chaves:
        - 'cluster': int — índice do cluster mais próximo
        - 'distance_km': float — distância em km ao centróide (arredondado em 4 casas)
    """
    # 1. Normalizar o ponto com o mesmo scaler do treino
    point = np.array([[lat, lng]], dtype=float)
    point_scaled = scaler.transform(point)

    # 2. Predizer o cluster
    cluster = int(model.predict(point_scaled)[0])

    # 3. Recuperar o centróide em lat/lng (desfazendo a normalização)
    centroid = scaler.inverse_transform(model.cluster_centers_[cluster].reshape(1, -1))[0]
    centroid_lat, centroid_lng = float(centroid[0]), float(centroid[1])

    # 4. Distância geodésica em km entre o ponto e o centróide
    distance_km = geopy.distance.geodesic((lat, lng), (centroid_lat, centroid_lng)).km

    return {
        "cluster": cluster,
        "distance_km": round(float(distance_km), 4),
    }
