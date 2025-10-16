import cv2
import numpy as np
import os
import json
import pandas as pd
from skimage.measure import shannon_entropy

# A. Inicialización
# Definir parámetros de entrada
input_image_path = "./data/imagen_original.png"
output_dir = "./resultados_clahe/"
clip_limits = np.arange(1.0, 5.1, 0.5)  # α: 1.0 a 5.0, paso 0.5
tile_sizes = [8, 16, 32]  # ω: Tamaños de baldosa
metrics_data = []  # Lista para almacenar datos del DataFrame maestro

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Cargar imagen original (asegurarse de que esté en escala de grises)
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {input_image_path}")

# B. Bucle de Experimentación
iteration_id = 0
for alpha in clip_limits:
    for omega in tile_sizes:
        iteration_id += 1
        # Crear carpeta para la iteración
        iteration_dir = os.path.join(output_dir, f"iteracion_{iteration_id}")
        os.makedirs(iteration_dir, exist_ok=True)

        # C. Procesamiento: Aplicar CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(omega, omega))
            modified_image = clahe.apply(image)
        except Exception as e:
            print(f"Error aplicando CLAHE en iteración {iteration_id}: {e}")
            continue

        # D. Cálculo de Métricas
        # Entropía
        entropy = shannon_entropy(modified_image)
        
        # Contraste de Michelson: (max - min) / (max + min)
        img_min, img_max = np.min(modified_image), np.max(modified_image)
        michelson_contrast = (img_max - img_min) / (img_max + img_min + 1e-10)  # Evitar división por cero

        # E. Almacenamiento Local (Trazabilidad)
        # Guardar imagen modificada
        output_image_path = os.path.join(iteration_dir, "imagen_modificada.png")
        cv2.imwrite(output_image_path, modified_image)

        # Guardar parámetros y métricas en JSON
        result_dict = {
            "ClipLimit": float(alpha),
            "TileSize": int(omega),
            "Entropy": float(entropy),
            "MichelsonContrast": float(michelson_contrast)
        }
        json_path = os.path.join(iteration_dir, "parametros_resultados.json")
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=4)

        # F. Almacenamiento Maestro (Análisis)
        metrics_data.append({
            "ID_Experimento": iteration_id,
            "ClipLimit": float(alpha),
            "TileSize": int(omega),
            "Entropy": float(entropy),
            "MichelsonContrast": float(michelson_contrast),
            "Ruta_Imagen": output_image_path
        })

# G. Finalización: Guardar DataFrame maestro en CSV
df = pd.DataFrame(metrics_data)
master_csv_path = os.path.join(output_dir, "resultados_maestros.csv")
df.to_csv(master_csv_path, index=False)
print(f"Resultados guardados en: {master_csv_path}")