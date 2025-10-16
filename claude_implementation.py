import cv2
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from skimage import filters

def calcular_entropia(imagen):
    """Calcula la entropía de Shannon de la imagen."""
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    histograma = histograma.flatten() / histograma.sum()
    histograma = histograma[histograma > 0]
    return entropy(histograma, base=2)

def calcular_contraste_local_promedio(imagen, kernel_size=3):
    """Calcula el contraste local promedio usando la desviación estándar local."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    media_local = cv2.filter2D(imagen.astype(np.float32), -1, kernel)
    diff_cuadrado = (imagen.astype(np.float32) - media_local) ** 2
    varianza_local = cv2.filter2D(diff_cuadrado, -1, kernel)
    std_local = np.sqrt(varianza_local)
    return np.mean(std_local)

def calcular_nitidez_borde(imagen):
    """Calcula la nitidez de borde usando el gradiente de Sobel."""
    sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(magnitud)

def calcular_contraste_michelson(imagen):
    """Calcula el contraste de Michelson global."""
    I_max = np.max(imagen)
    I_min = np.min(imagen)
    if I_max + I_min == 0:
        return 0
    return (I_max - I_min) / (I_max + I_min)

def procesar_imagenes_clahe(input_dir, output_dir, clip_limits, tile_sizes):
    """
    Función principal para experimentación CLAHE con trazabilidad completa.
    
    Args:
        input_dir: Directorio con imágenes originales
        output_dir: Directorio raíz para resultados
        clip_limits: Array de valores para clip limit
        tile_sizes: Lista de tamaños de baldosa
    """
    
    # A. INICIALIZACIÓN
    print("=" * 70)
    print("INICIO DE EXPERIMENTACIÓN CLAHE CON TRAZABILIDAD ESTRUCTURADA")
    print("=" * 70)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Directorio de salida creado: {output_dir}")
    
    # Buscar imágenes en el directorio de entrada
    input_path = Path(input_dir)
    extensiones_validas = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    imagenes = [f for f in input_path.iterdir() 
                if f.suffix.lower() in extensiones_validas]
    
    if not imagenes:
        print(f"✗ No se encontraron imágenes en {input_dir}")
        return
    
    print(f"✓ Encontradas {len(imagenes)} imagen(es) para procesar")
    
    # Preparar DataFrame maestro
    resultados_maestros = []
    
    # Contador de iteraciones global
    id_experimento = 0
    
    # Procesar cada imagen
    for img_path in imagenes:
        print(f"\n{'─' * 70}")
        print(f"Procesando: {img_path.name}")
        print(f"{'─' * 70}")
        
        # Cargar imagen en escala de grises
        imagen_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if imagen_original is None:
            print(f"✗ Error al cargar {img_path.name}")
            continue
        
        print(f"✓ Imagen cargada: {imagen_original.shape}")
        
        # B. BUCLE DE EXPERIMENTACIÓN
        total_iteraciones = len(clip_limits) * len(tile_sizes)
        iteracion_actual = 0
        
        for alpha in clip_limits:
            for omega in tile_sizes:
                iteracion_actual += 1
                id_experimento += 1
                
                # Crear carpeta de iteración
                carpeta_iteracion = os.path.join(output_dir, f"iteracion_{id_experimento:04d}")
                os.makedirs(carpeta_iteracion, exist_ok=True)
                
                print(f"\n[{iteracion_actual}/{total_iteraciones}] Experimento {id_experimento}: α={alpha:.2f}, ω={omega}")
                
                # C. PROCESAMIENTO - Aplicar CLAHE
                clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(omega, omega))
                imagen_modificada = clahe.apply(imagen_original)
                
                # D. CÁLCULO DE MÉTRICAS
                try:
                    metricas = {
                        'entropia': calcular_entropia(imagen_modificada),
                        'contraste_local_promedio': calcular_contraste_local_promedio(imagen_modificada),
                        'nitidez_borde': calcular_nitidez_borde(imagen_modificada),
                        'contraste_michelson': calcular_contraste_michelson(imagen_modificada)
                    }
                    
                    print(f"  Entropía: {metricas['entropia']:.4f}")
                    print(f"  Contraste Local: {metricas['contraste_local_promedio']:.4f}")
                    print(f"  Nitidez Borde: {metricas['nitidez_borde']:.4f}")
                    print(f"  Contraste Michelson: {metricas['contraste_michelson']:.4f}")
                    
                except Exception as e:
                    print(f"  ✗ Error calculando métricas: {e}")
                    metricas = {
                        'entropia': 0,
                        'contraste_local_promedio': 0,
                        'nitidez_borde': 0,
                        'contraste_michelson': 0
                    }
                
                # E. ALMACENAMIENTO LOCAL (TRAZABILIDAD)
                # 1. Guardar imagen modificada
                ruta_imagen = os.path.join(carpeta_iteracion, "imagen_modificada.png")
                cv2.imwrite(ruta_imagen, imagen_modificada)
                
                # 2. Guardar parámetros y métricas en JSON
                datos_json = {
                    'ID_Experimento': id_experimento,
                    'Imagen_Original': img_path.name,
                    'ClipLimit': float(alpha),
                    'TileSize': int(omega),
                    'Metricas': {
                        'Entropia': float(metricas['entropia']),
                        'Contraste_Local_Promedio': float(metricas['contraste_local_promedio']),
                        'Nitidez_Borde': float(metricas['nitidez_borde']),
                        'Contraste_Michelson': float(metricas['contraste_michelson'])
                    }
                }
                
                ruta_json = os.path.join(carpeta_iteracion, "parametros_resultados.json")
                with open(ruta_json, 'w', encoding='utf-8') as f:
                    json.dump(datos_json, f, indent=4, ensure_ascii=False)
                
                # F. ALMACENAMIENTO MAESTRO (ANÁLISIS)
                fila_resultado = {
                    'ID_Experimento': id_experimento,
                    'Imagen_Original': img_path.name,
                    'ClipLimit': alpha,
                    'TileSize': omega,
                    'Entropia': metricas['entropia'],
                    'Contraste_Local_Promedio': metricas['contraste_local_promedio'],
                    'Nitidez_Borde': metricas['nitidez_borde'],
                    'Contraste_Michelson': metricas['contraste_michelson']
                }
                resultados_maestros.append(fila_resultado)
    
    # G. FINALIZACIÓN - Guardar DataFrame maestro
    print(f"\n{'=' * 70}")
    print("FINALIZANDO EXPERIMENTACIÓN")
    print(f"{'=' * 70}")
    
    df_maestro = pd.DataFrame(resultados_maestros)
    ruta_csv = os.path.join(output_dir, "resultados_maestros.csv")
    df_maestro.to_csv(ruta_csv, index=False, encoding='utf-8')
    
    print(f"\n✓ Tabla maestra guardada: {ruta_csv}")
    print(f"✓ Total de experimentos realizados: {id_experimento}")
    print(f"✓ Carpetas de trazabilidad creadas: {id_experimento}")
    
    # Mostrar estadísticas resumidas
    print(f"\n{'─' * 70}")
    print("ESTADÍSTICAS RESUMIDAS")
    print(f"{'─' * 70}")
    print(df_maestro.describe())
    
    print(f"\n{'=' * 70}")
    print("✓ PROCESO COMPLETADO EXITOSAMENTE")
    print(f"{'=' * 70}\n")
    
    return df_maestro


if __name__ == "__main__":
    # CONFIGURACIÓN DE PARÁMETROS
    INPUT_DIR = "./data"
    OUTPUT_DIR = "./resultados_clahe/"
    
    # Rangos de búsqueda
    CLIP_LIMITS = np.arange(1.0, 5.1, 0.5)  # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    TILE_SIZES = [8, 16, 32]
    
    print("\nCONFIGURACIÓN DE EXPERIMENTACIÓN:")
    print(f"  - Directorio de entrada: {INPUT_DIR}")
    print(f"  - Directorio de salida: {OUTPUT_DIR}")
    print(f"  - Clip Limits: {list(CLIP_LIMITS)}")
    print(f"  - Tile Sizes: {TILE_SIZES}")
    print(f"  - Total de combinaciones por imagen: {len(CLIP_LIMITS) * len(TILE_SIZES)}")
    
    # Ejecutar experimentación
    df_resultados = procesar_imagenes_clahe(INPUT_DIR, OUTPUT_DIR, CLIP_LIMITS, TILE_SIZES)
    
    if df_resultados is not None:
        print("\nPara analizar los resultados, puedes:")
        print("1. Abrir 'resultados_maestros.csv' en Excel o Python")
        print("2. Ordenar por la métrica más relevante para tu caso")
        print("3. Identificar las mejores iteraciones")
        print("4. Revisar las imágenes en las carpetas correspondientes")