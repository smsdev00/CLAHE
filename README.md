## Optimización CLAHE en Python con Trazabilidad Estructurada

El objetivo es automatizar la experimentación con el algoritmo **CLAHE** y registrar cada resultado en una estructura de directorios que facilite la evaluación posterior de la delimitación de estructuras.

### 1. Preparación y Dependencias (Python)

Se requiere la instalación de librerías esenciales para procesamiento de imágenes y manejo de *arrays*:

* **Librerías:** `cv2` (OpenCV) o `skimage`, `numpy`, `os`, `json` (o `io` para TXT).

### 2. Parámetros de Entrada

| Parámetro | Descripción | Ejemplo de Rango de Búsqueda |
| :--- | :--- | :--- |
| **Ruta Imagen Original** | Dirección del archivo de la imagen con bajo contraste. | `"./data/imagen_original.png"` |
| **Directorio de Salida** | Carpeta raíz donde se almacenarán los resultados. | `"./resultados_clahe/"` |
| **Límite de Clip** ($\alpha$) | Array o rango de valores flotantes para el *clip limit*. | `np.arange(1.0, 5.1, 0.5)` |
| **Tamaño de Baldosa** ($\omega$) | Lista de valores enteros para el tamaño del lado de la baldosa ($N \times N$). | `[8, 16, 32]` |

### 3. Lógica del Algoritmo y Estructura de Salida

Se requiere un doble bucle (anidado) en Python para iterar sobre todas las combinaciones de $(\alpha, \omega)$, ejecutando los siguientes pasos en cada ciclo:

#### 3.1. Iteración y Aplicación de CLAHE

1.  **Generar ID de Experimento:** Asignar un índice único (`$i$`) a la iteración actual.
2.  **Crear Carpeta de Iteración:**
    * Ruta: `[Directorio de Salida]/iteracion_[i]`
    * Utilizar `os.makedirs()` para asegurar la existencia del directorio.
3.  **Aplicar CLAHE:**
    * Instanciar el objeto CLAHE: `clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(omega, omega))`
    * Aplicar la transformación: `imagen_modificada = clahe.apply(imagen_original_gray)`

#### 3.2. Evaluación y Almacenamiento

1.  **Cálculo de Métricas:** Calcule las métricas de calidad seleccionadas (p. ej., Entropía, Contraste de Michelson, etc.) sobre la `imagen_modificada`.
2.  **Guardar Imagen:**
    * Ruta: `[Directorio de Salida]/iteracion_[i]/imagen_modificada.png`
    * Utilizar `cv2.imwrite()` para guardar el resultado.
3.  **Guardar Parámetros y Métricas (JSON/TXT):**
    * Crear un diccionario/estructura de datos con:
        * `ClipLimit`: $\alpha$
        * `TileSize`: $\omega$
        * Métrica 1: Valor
        * Métrica 2: Valor
    * **Almacenamiento JSON:** Utilizar `json.dump()` para guardar esta estructura en: `[Directorio de Salida]/iteracion_[i]/parametros_resultados.json`

### 4. Evaluación Final (Reporte)

En lugar de que el algoritmo determine por si solo las mejores imágenes durante la ejecución, el proceso debe **almacenar los resultados de las métricas** de todas las iteraciones en una **Tabla Maestra (CSV o DataFrame de Pandas)**.

* **Tabla Maestra:** Esta tabla (`resultados_maestros.csv`) debe contener la columna `ID_Experimento`, los valores de los parámetros y todas las métricas.
* **Análisis:** Una vez completada la ejecución, esta tabla se utilizará para **ordenar** los resultados y **filtrar** las 5 o 10 mejores iteraciones según la métrica más relevante para la delimitación de estructuras.

**Este enfoque garantiza la trazabilidad completa, la evaluación objetiva y la organización de la información en el formato solicitado.**

## Fase 1: Generación y Almacenamiento de Datos Brutos (Script de Ejecución)

Este *script* se encargará de la parte intensiva: iterar, aplicar CLAHE, calcular métricas y almacenar los resultados en la estructura de carpetas requerida, además de generar el archivo maestro de resultados.

### `1_generar_datos_clahe.py`

| Tarea | Descripción y Requisitos de Código |
| :--- | :--- |
| **A. Inicialización** | Definir los rangos de búsqueda de $\alpha$ (Clip Limit) y $\omega$ (Tile Size). Configurar el directorio de salida (`os.makedirs`). |
| **B. Bucle de Experimentación** | Bucle anidado (itera sobre $\alpha$ y $\omega$). En cada iteración: crear una carpeta `iteracion_[i]`. |
| **C. Procesamiento** | Aplicar CLAHE (`cv2.createCLAHE().apply()`). |
| **D. Cálculo de Métricas** | Calcular las métricas clave (Entropía, Contraste Local Promedio, Nitidez de Borde) sobre la imagen modificada. |
| **E. Almacenamiento Local (Trazabilidad)** | 1. Guardar la imagen modificada (`cv2.imwrite()`). 2. Guardar los parámetros y los valores de las métricas en un archivo **JSON** dentro de la carpeta de la iteración. |
| **F. Almacenamiento Maestro (Análisis)** | Almacenar los resultados de cada iteración (ID, $\alpha$, $\omega$, Métrica 1, Métrica 2...) como una fila en un *DataFrame* de **Pandas**. |
| **G. Finalización** | Guardar el *DataFrame* maestro en un archivo CSV: **`resultados_maestros.csv`** en el directorio de salida. |

El producto final de este *script* es el directorio completo de resultados y el archivo `resultados_maestros.csv`, el cual contendrá todos los **datos brutos** para el análisis.

***

## Fase 2: Análisis y Visualización (Script de Procesamiento)

Este *script* opera exclusivamente sobre el archivo `resultados_maestros.csv` generado, separando la lógica de cálculo y generación de datos de la lógica de análisis y presentación.

### `2_analisis_resultados.py`

| Tarea | Descripción y Requisitos de Código |
| :--- | :--- |
| **A. Carga de Datos** | Cargar el archivo `resultados_maestros.csv` en un *DataFrame* de Pandas. |
| **B. Preselección Objetiva** | Ordenar el *DataFrame* según la métrica de calidad prioritaria (ej. Contraste Local Promedio). Filtrar y mostrar las **10 mejores combinaciones** de parámetros. |
| **C. Visualización de Resultados** | **Gráficos de Dispersión o Mapas de Calor:** Generar visualizaciones que muestren la relación entre los parámetros ($\alpha$ vs. $\omega$) y el valor de la métrica clave. Esto ayuda a identificar tendencias y regiones óptimas. |
| **D. Evaluación Subjetiva (Reporte Final)** | Seleccionar los IDs de las 5 mejores iteraciones y generar un reporte que incluya: 1. La combinación de parámetros. 2. La imagen resultante (cargada desde su respectiva carpeta). 3. Un **juicio cualitativo** sobre la mejora en la delimitación de estructuras. |
| **E. Conclusión** | Determinar y reportar la combinación óptima $(\alpha^*, \omega^*)$ que maximiza la calidad y presenta la mejor delimitación visual de las estructuras. |

Este enfoque garantiza que el **proceso de optimización (Fase 1)** es puramente mecánico y que el **proceso de decisión (Fase 2)** es analítico, facilitando la auditoría y la toma de decisiones.