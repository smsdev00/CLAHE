# Optimización CLAHE en Python con Trazabilidad Estructurada

## 📋 Descripción General

Sistema automatizado de experimentación con el algoritmo **CLAHE** (Contrast Limited Adaptive Histogram Equalization) que registra cada resultado en una estructura de directorios organizada, facilitando la evaluación posterior de la delimitación de estructuras en imágenes médicas o científicas.

El sistema implementa un enfoque de dos fases que separa la **generación de datos** del **análisis y visualización**, garantizando trazabilidad completa y evaluación objetiva.

---

## 🎯 Objetivo

Automatizar la búsqueda de parámetros óptimos para CLAHE mediante:
- Experimentación exhaustiva de combinaciones de parámetros
- Registro estructurado de todos los resultados
- Cálculo de múltiples métricas de calidad de imagen
- Análisis visual y estadístico para identificar configuraciones óptimas

---

## 🛠️ Requisitos e Instalación

### Dependencias

```bash
pip install -r requirements.txt
```

### Librerías utilizadas:
- **OpenCV** (`cv2`): Procesamiento de imágenes y aplicación de CLAHE
- **NumPy**: Manejo de arrays y operaciones numéricas
- **Pandas**: Gestión de datos tabulares y análisis
- **Matplotlib/Seaborn**: Visualización de resultados
- **scikit-image**: Algoritmos de procesamiento de imágenes
- **SciPy**: Cálculo de entropía y estadísticas

---

## 📁 Estructura del Proyecto

```
proyecto_clahe/
├── data/                          # Directorio de imágenes originales
│   ├── imagen_original.png
│   └── ...
├── 1_generar_datos_clahe.py      # Script Fase 1: Generación de datos
├── 2_analisis_resultados.py      # Script Fase 2: Análisis y visualización
└── resultados_clahe/             # Directorio de salida (generado automáticamente)
    ├── iteracion_0001/
    │   ├── imagen_modificada.png
    │   └── parametros_resultados.json
    ├── iteracion_0002/
    │   └── ...
    ├── resultados_maestros.csv    # Tabla maestra con todas las métricas
    ├── parametros_optimos.json    # Parámetros óptimos identificados
    ├── graficos_analisis/         # Visualizaciones generadas
    │   ├── heatmap_entropia.png
    │   ├── scatter_3d_all_metrics.png
    │   └── ...
    └── reporte_final/             # Reporte de evaluación
        ├── comparacion_top_resultados.png
        └── reporte_detallado.txt
```

---

## 🔧 Parámetros de Configuración

### Parámetros de Entrada

| Parámetro | Descripción | Valor por Defecto | Rango Sugerido |
|:----------|:------------|:------------------|:---------------|
| **Ruta de Entrada** | Directorio con imágenes originales | `"./data"` | - |
| **Directorio de Salida** | Carpeta raíz para resultados | `"./resultados_clahe/"` | - |
| **Límite de Clip** (α) | Control de amplificación de contraste | `1.0 - 5.0` (paso 0.5) | `0.5 - 10.0` |
| **Tamaño de Baldosa** (ω) | Dimensión de la cuadrícula local (N×N) | `[8, 16, 32]` | `4 - 64` |

### Métricas Calculadas

1. **Entropía de Shannon**: Mide la cantidad de información en la imagen (mayor = más detalles)
2. **Contraste Local Promedio**: Evalúa la variabilidad local de intensidades
3. **Nitidez de Borde**: Cuantifica la definición de bordes (gradiente de Sobel)
4. **Contraste de Michelson**: Contraste global de la imagen

---

## 🚀 Uso del Sistema

### Fase 1: Generación y Almacenamiento de Datos

Este script ejecuta el proceso intensivo de experimentación, calculando todas las combinaciones posibles de parámetros.

#### Ejecución:

```bash
python 1_generar_datos_clahe.py
```

#### Funcionalidades:

- ✅ **Carga automática** de todas las imágenes en `./data/`
- ✅ **Bucle anidado** sobre todas las combinaciones de (α, ω)
- ✅ **Aplicación de CLAHE** con cada configuración
- ✅ **Cálculo de 4 métricas** de calidad por iteración
- ✅ **Almacenamiento estructurado**:
  - Imagen modificada en carpeta individual
  - Parámetros y métricas en JSON
  - Registro en tabla maestra CSV
- ✅ **Progreso en tiempo real** con estadísticas

#### Salida:
```
resultados_clahe/
├── iteracion_0001/ ... iteracion_0027/  (27 iteraciones: 9 α × 3 ω)
└── resultados_maestros.csv              (Tabla maestra completa)
```

#### Configuración del Script:

```python
# En 1_generar_datos_clahe.py (líneas 217-223)
INPUT_DIR = "./data"
OUTPUT_DIR = "./resultados_clahe/"

CLIP_LIMITS = np.arange(1.0, 5.1, 0.5)  # [1.0, 1.5, ..., 5.0]
TILE_SIZES = [8, 16, 32]

# Total de experimentos = len(CLIP_LIMITS) × len(TILE_SIZES) × N_imágenes
```

---

### Fase 2: Análisis y Visualización

Este script opera exclusivamente sobre el CSV generado, separando completamente la lógica de análisis de la generación de datos.

#### Ejecución:

```bash
python 2_analisis_resultados.py
```

#### Funcionalidades:

- 📊 **Análisis estadístico** del CSV maestro
- 📈 **Visualizaciones avanzadas**:
  - Mapas de calor (α vs ω) para cada métrica
  - Gráficos 3D de relaciones paramétricas
  - Evolución de métricas por Tile Size
  - Histogramas de distribución
- 🏆 **Identificación de top 10 configuraciones**
- 🖼️ **Comparación visual** lado a lado (Original vs CLAHE)
- 📝 **Reporte detallado** con evaluación cualitativa
- 🎯 **Determinación de parámetros óptimos** (α*, ω*)

#### Salida:
```
resultados_clahe/
├── graficos_analisis/           (4 tipos de visualizaciones)
├── reporte_final/               (Comparación visual + reporte textual)
└── parametros_optimos.json      (Mejor configuración encontrada)
```

#### Configuración del Script:

```python
# En 2_analisis_resultados.py (líneas 339-346)
RUTA_CSV = "./resultados_clahe/resultados_maestros.csv"
RESULTADOS_DIR = "./resultados_clahe/"
OUTPUT_DIR = "./resultados_clahe/"

METRICA_PRINCIPAL = 'Contraste_Local_Promedio'  # Métrica para ordenamiento
TOP_N_PRESELECCION = 10  # Mejores combinaciones a analizar
TOP_N_EVALUACION = 5     # Top N para reporte visual detallado
```

---

## 📊 Interpretación de Resultados

### Tabla Maestra (`resultados_maestros.csv`)

Contiene todas las iteraciones con sus parámetros y métricas:

```csv
ID_Experimento,Imagen_Original,ClipLimit,TileSize,Entropia,Contraste_Local_Promedio,Nitidez_Borde,Contraste_Michelson
1,imagen.png,1.0,8,7.2341,23.4567,15.8923,0.8234
2,imagen.png,1.0,16,7.3456,25.1234,16.2341,0.8456
...
```

### Parámetros Óptimos (`parametros_optimos.json`)

```json
{
    "alpha_optimo": 2.5,
    "omega_optimo": 16,
    "id_experimento": 14,
    "metricas": {
        "entropia": 7.4523,
        "contraste_local": 28.3421,
        "nitidez_borde": 18.2341,
        "contraste_michelson": 0.8723
    }
}
```

### Reporte Detallado (`reporte_detallado.txt`)

Incluye:
- Ranking completo de las mejores configuraciones
- Parámetros exactos de cada experimento
- Valores de todas las métricas
- Evaluación cualitativa automática basada en percentiles

---

## 🎨 Visualizaciones Generadas

### 1. Mapas de Calor (Heatmaps)
Muestran el valor de cada métrica para todas las combinaciones (α, ω):
- `heatmap_entropia.png`
- `heatmap_contraste_local_promedio.png`
- `heatmap_nitidez_borde.png`
- `heatmap_contraste_michelson.png`

**Utilidad**: Identificar regiones óptimas en el espacio de parámetros.

### 2. Gráficos 3D de Dispersión
Relación tridimensional entre Clip Limit, Tile Size y cada métrica.

**Utilidad**: Visualizar tendencias y patrones no lineales.

### 3. Gráficos de Evolución
Líneas que muestran cómo varía cada métrica con α para diferentes valores de ω.

**Utilidad**: Entender el efecto individual de cada parámetro.

### 4. Histogramas de Distribución
Distribución de valores de cada métrica con línea de media.

**Utilidad**: Identificar valores atípicos y normalidad de las métricas.

### 5. Comparación Visual
Imagen original vs. las 5 mejores aplicaciones CLAHE lado a lado.

**Utilidad**: Evaluación subjetiva de mejora en delimitación de estructuras.

---

## 🔍 Criterios de Selección

### Métricas Prioritarias según Aplicación

| Aplicación | Métrica Principal | Justificación |
|:-----------|:------------------|:--------------|
| **Imágenes médicas** | Contraste Local Promedio | Maximiza la visibilidad de estructuras anatómicas |
| **Microscopía** | Nitidez de Borde | Crítico para delimitación celular |
| **Fotografía de bajo contraste** | Contraste de Michelson | Mejora percepción visual global |
| **Documentos históricos** | Entropía | Recupera máxima información de la imagen |

### Evaluación Cualitativa Automática

El sistema evalúa automáticamente cada resultado comparando con percentiles:
- **Excelente** (> percentil 75): ✓ Marcado en verde
- **Moderado** (percentil 25-75): • Marcado en amarillo
- **Bajo** (< percentil 25): ✗ Requiere atención

---

## ⚙️ Personalización Avanzada

### Modificar Rangos de Parámetros

```python
# En 1_generar_datos_clahe.py
CLIP_LIMITS = np.arange(0.5, 10.1, 0.5)  # Rango más amplio
TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64]  # Más opciones
```

### Agregar Nuevas Métricas

```python
# En 1_generar_datos_clahe.py - Agregar función de cálculo
def calcular_nueva_metrica(imagen):
    # Tu implementación aquí
    return valor

# En la sección de cálculo de métricas (línea ~135)
metricas['nueva_metrica'] = calcular_nueva_metrica(imagen_modificada)

# Agregar a datos_json y fila_resultado
```

### Cambiar Métrica de Ordenamiento

```python
# En 2_analisis_resultados.py (línea 342)
METRICA_PRINCIPAL = 'Nitidez_Borde'  # O cualquier otra métrica
```

### Procesar Solo Imágenes Específicas

```python
# En 1_generar_datos_clahe.py (línea ~80)
extensiones_validas = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm']
```

---

## 📖 Flujo de Trabajo Recomendado

1. **Preparación**
   ```bash
   # Colocar imágenes en ./data/
   mkdir -p data
   cp /ruta/imagenes/* data/
   ```

2. **Experimentación (Fase 1)**
   ```bash
   python 1_generar_datos_clahe.py
   # Esperar completación (puede tomar minutos/horas según N experimentos)
   ```

3. **Análisis (Fase 2)**
   ```bash
   python 2_analisis_resultados.py
   # Revisar visualizaciones en ./resultados_clahe/graficos_analisis/
   ```

4. **Evaluación Manual**
   - Abrir `reporte_final/comparacion_top_resultados.png`
   - Revisar `reporte_final/reporte_detallado.txt`
   - Validar parámetros óptimos en `parametros_optimos.json`

5. **Aplicación**
   ```python
   # Usar parámetros óptimos en tu pipeline
   import json
   with open('./resultados_clahe/parametros_optimos.json') as f:
       optimos = json.load(f)
   
   clahe = cv2.createCLAHE(
       clipLimit=optimos['alpha_optimo'],
       tileGridSize=(optimos['omega_optimo'], optimos['omega_optimo'])
   )
   imagen_mejorada = clahe.apply(tu_imagen_gris)
   ```

---

## 🧪 Ejemplo de Uso Completo

```bash
# 1. Clonar o descargar scripts
# 2. Instalar dependencias
pip install opencv-python numpy pandas matplotlib seaborn scikit-image scipy

# 3. Preparar datos
mkdir data
cp imagen_baja_contraste.png data/

# 4. Ejecutar experimentación
python 1_generar_datos_clahe.py
# Salida: 27 iteraciones completadas, CSV maestro generado

# 5. Analizar resultados
python 2_analisis_resultados.py
# Salida: 8 visualizaciones, reporte detallado, parámetros óptimos

# 6. Revisar resultados
# - Abrir graficos_analisis/heatmap_contraste_local_promedio.png
# - Leer reporte_final/reporte_detallado.txt
# - Aplicar parámetros de parametros_optimos.json
```

---

## 📈 Rendimiento y Escalabilidad

| Escenario | N Imágenes | Combinaciones | Tiempo Estimado | Espacio en Disco |
|:----------|:-----------|:--------------|:----------------|:-----------------|
| Pequeño | 1 | 27 (9×3) | 1-2 min | ~50 MB |
| Mediano | 5 | 135 | 5-10 min | ~250 MB |
| Grande | 10 | 270 | 10-20 min | ~500 MB |
| Exhaustivo | 1 | 200 (20×10) | 5-10 min | ~400 MB |

*Tiempos aproximados en hardware estándar (CPU moderna, imágenes 512×512)*

---

## 🐛 Solución de Problemas

### Error: "No se encontraron imágenes"
```bash
# Verificar que data/ contiene imágenes
ls -la data/
# Formatos soportados: .png, .jpg, .jpeg, .bmp, .tiff
```

### Error: "No se encuentra resultados_maestros.csv"
```bash
# Ejecutar primero el script de Fase 1
python 1_generar_datos_clahe.py
```

### Advertencia: "No se encontró imagen para ID X"
- Verificar que la carpeta `iteracion_XXXX` existe en `resultados_clahe/`
- Re-ejecutar Fase 1 si faltan iteraciones

### Métricas con valores 0
- Verificar que la imagen se cargó correctamente en escala de grises
- Revisar que la imagen no está corrupta

---

## 📚 Referencias Técnicas

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Publicación original**: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- **Ventajas**: Mejora contraste local sin amplificar ruido excesivamente
- **Aplicaciones**: Imágenes médicas, microscopía, fotografía de bajo contraste

### Métricas de Calidad de Imagen
- **Entropía**: Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- **Gradiente de Sobel**: Sobel, I. (1968). "An Isotropic 3×3 Image Gradient Operator"
- **Contraste de Michelson**: Michelson, A.A. (1927). "Studies in Optics"

---

## 🤝 Contribuciones y Extensiones

### Ideas para Extensiones Futuras
- [ ] Soporte para imágenes DICOM (`.dcm`)
- [ ] Procesamiento paralelo con `multiprocessing`
- [ ] Interfaz gráfica (GUI) con PyQt o Tkinter
- [ ] Exportación automática de reporte en PDF
- [ ] Integración con bases de datos (SQLite/PostgreSQL)
- [ ] Soporte para video (frame por frame)
- [ ] Optimización Bayesiana de hiperparámetros

---

