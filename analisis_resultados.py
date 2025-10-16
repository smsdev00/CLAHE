import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from pathlib import Path
import json

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def cargar_datos_maestros(ruta_csv):
    """A. CARGA DE DATOS - Cargar archivo CSV maestro."""
    print("=" * 70)
    print("ANÁLISIS DE RESULTADOS CLAHE")
    print("=" * 70)
    
    if not os.path.exists(ruta_csv):
        print(f"✗ Error: No se encuentra el archivo {ruta_csv}")
        return None
    
    df = pd.read_csv(ruta_csv)
    print(f"\n✓ Datos cargados exitosamente: {ruta_csv}")
    print(f"✓ Total de experimentos: {len(df)}")
    print(f"✓ Columnas disponibles: {list(df.columns)}")
    
    return df

def preseleccion_objetiva(df, metrica_principal='Contraste_Local_Promedio', top_n=10):
    """B. PRESELECCIÓN OBJETIVA - Ordenar y filtrar mejores resultados."""
    print(f"\n{'─' * 70}")
    print(f"PRESELECCIÓN OBJETIVA - Top {top_n} según {metrica_principal}")
    print(f"{'─' * 70}")
    
    # Ordenar por métrica principal (descendente)
    df_ordenado = df.sort_values(by=metrica_principal, ascending=False)
    top_resultados = df_ordenado.head(top_n)
    
    print(f"\nTop {top_n} combinaciones de parámetros:")
    print("─" * 70)
    
    # Mostrar tabla formateada
    columnas_mostrar = ['ID_Experimento', 'ClipLimit', 'TileSize', 
                        'Entropia', 'Contraste_Local_Promedio', 
                        'Nitidez_Borde', 'Contraste_Michelson']
    
    # Filtrar columnas que existan en el DataFrame
    columnas_existentes = [col for col in columnas_mostrar if col in top_resultados.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(top_resultados[columnas_existentes].to_string(index=False))
    
    return top_resultados

def visualizar_relaciones_parametros(df, output_dir):
    """C. VISUALIZACIÓN DE RESULTADOS - Gráficos de dispersión y mapas de calor."""
    print(f"\n{'─' * 70}")
    print("GENERANDO VISUALIZACIONES")
    print(f"{'─' * 70}")
    
    # Crear directorio para gráficos
    graficos_dir = os.path.join(output_dir, "graficos_analisis")
    os.makedirs(graficos_dir, exist_ok=True)
    
    metricas = ['Entropia', 'Contraste_Local_Promedio', 'Nitidez_Borde', 'Contraste_Michelson']
    metricas_existentes = [m for m in metricas if m in df.columns]
    
    # 1. MAPAS DE CALOR para cada métrica
    print("\n1. Generando mapas de calor (α vs ω)...")
    
    for metrica in metricas_existentes:
        plt.figure(figsize=(10, 8))
        
        # Crear tabla pivote
        pivot_table = df.pivot_table(
            values=metrica,
            index='TileSize',
            columns='ClipLimit',
            aggfunc='mean'
        )
        
        # Crear mapa de calor
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={'label': metrica})
        plt.title(f'Mapa de Calor: {metrica}\n(α: ClipLimit, ω: TileSize)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Clip Limit (α)', fontsize=12)
        plt.ylabel('Tile Size (ω)', fontsize=12)
        plt.tight_layout()
        
        # Guardar
        filename = f"heatmap_{metrica.lower()}.png"
        plt.savefig(os.path.join(graficos_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {filename}")
    
    # 2. GRÁFICOS DE DISPERSIÓN 3D (Clip Limit vs Tile Size vs Métrica)
    print("\n2. Generando gráficos de dispersión 3D...")
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, metrica in enumerate(metricas_existentes, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        scatter = ax.scatter(df['ClipLimit'], df['TileSize'], df[metrica],
                            c=df[metrica], cmap='viridis', s=100, alpha=0.6)
        
        ax.set_xlabel('Clip Limit (α)', fontsize=10)
        ax.set_ylabel('Tile Size (ω)', fontsize=10)
        ax.set_zlabel(metrica, fontsize=10)
        ax.set_title(f'{metrica}', fontsize=12, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    plt.suptitle('Relación entre Parámetros CLAHE y Métricas de Calidad', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, "scatter_3d_all_metrics.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ scatter_3d_all_metrics.png")
    
    # 3. GRÁFICO DE LÍNEAS: Evolución de métricas por Clip Limit
    print("\n3. Generando gráficos de líneas por Tile Size...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metrica in enumerate(metricas_existentes):
        ax = axes[idx]
        
        for tile_size in sorted(df['TileSize'].unique()):
            df_filtrado = df[df['TileSize'] == tile_size]
            df_agrupado = df_filtrado.groupby('ClipLimit')[metrica].mean()
            ax.plot(df_agrupado.index, df_agrupado.values, 
                   marker='o', linewidth=2, label=f'ω={tile_size}')
        
        ax.set_xlabel('Clip Limit (α)', fontsize=11)
        ax.set_ylabel(metrica, fontsize=11)
        ax.set_title(f'Evolución de {metrica} por Tile Size', 
                    fontsize=12, fontweight='bold')
        ax.legend(title='Tile Size', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Evolución de Métricas según Clip Limit', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, "lineas_evolucion_metricas.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ lineas_evolucion_metricas.png")
    
    # 4. DISTRIBUCIONES: Histogramas de métricas
    print("\n4. Generando histogramas de distribución...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metrica in enumerate(metricas_existentes):
        ax = axes[idx]
        ax.hist(df[metrica], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(df[metrica].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Media: {df[metrica].mean():.3f}')
        ax.set_xlabel(metrica, fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'Distribución de {metrica}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Distribución de Métricas de Calidad', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graficos_dir, "histogramas_distribucion.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ histogramas_distribucion.png")
    
    print(f"\n✓ Todas las visualizaciones guardadas en: {graficos_dir}")
    
    return graficos_dir

def evaluar_subjetivamente(df, top_ids, resultados_dir, output_dir):
    """D. EVALUACIÓN SUBJETIVA - Generar reporte visual de las mejores iteraciones."""
    print(f"\n{'─' * 70}")
    print(f"EVALUACIÓN SUBJETIVA - Top {len(top_ids)} Iteraciones")
    print(f"{'─' * 70}")
    
    # Crear directorio para reporte
    reporte_dir = os.path.join(output_dir, "reporte_final")
    os.makedirs(reporte_dir, exist_ok=True)
    
    # Filtrar datos de los top IDs
    df_top = df[df['ID_Experimento'].isin(top_ids)].sort_values(
        by='Contraste_Local_Promedio', ascending=False
    )
    
    # Crear figura para comparación visual
    n_top = len(top_ids)
    fig, axes = plt.subplots(n_top, 2, figsize=(14, 5 * n_top))
    
    if n_top == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(df_top.iterrows()):
        exp_id = int(row['ID_Experimento'])
        
        # Cargar imagen original (primera imagen encontrada en data)
        data_dir = Path("./data")
        imagenes_originales = list(data_dir.glob("*.*"))
        if imagenes_originales:
            img_original = cv2.imread(str(imagenes_originales[0]), cv2.IMREAD_GRAYSCALE)
        else:
            img_original = None
        
        # Cargar imagen modificada
        ruta_img_modificada = os.path.join(
            resultados_dir, 
            f"iteracion_{exp_id:04d}", 
            "imagen_modificada.png"
        )
        
        if os.path.exists(ruta_img_modificada):
            img_modificada = cv2.imread(ruta_img_modificada, cv2.IMREAD_GRAYSCALE)
        else:
            img_modificada = None
            print(f"  ✗ Advertencia: No se encontró imagen para ID {exp_id}")
        
        # Mostrar imagen original
        if img_original is not None:
            axes[idx, 0].imshow(img_original, cmap='gray')
            axes[idx, 0].set_title('Imagen Original', fontsize=11, fontweight='bold')
            axes[idx, 0].axis('off')
        else:
            axes[idx, 0].text(0.5, 0.5, 'No disponible', 
                             ha='center', va='center', fontsize=12)
            axes[idx, 0].axis('off')
        
        # Mostrar imagen modificada
        if img_modificada is not None:
            axes[idx, 1].imshow(img_modificada, cmap='gray')
            titulo = (f"ID {exp_id}: α={row['ClipLimit']:.1f}, ω={int(row['TileSize'])}\n"
                     f"Contraste: {row['Contraste_Local_Promedio']:.3f}, "
                     f"Entropía: {row['Entropia']:.3f}")
            axes[idx, 1].set_title(titulo, fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
        else:
            axes[idx, 1].text(0.5, 0.5, 'Imagen no encontrada', 
                             ha='center', va='center', fontsize=12)
            axes[idx, 1].axis('off')
        
        print(f"\n{idx + 1}. Experimento ID {exp_id}:")
        print(f"   Parámetros: α={row['ClipLimit']:.2f}, ω={int(row['TileSize'])}")
        print(f"   Entropía: {row['Entropia']:.4f}")
        print(f"   Contraste Local: {row['Contraste_Local_Promedio']:.4f}")
        print(f"   Nitidez Borde: {row['Nitidez_Borde']:.4f}")
        print(f"   Contraste Michelson: {row['Contraste_Michelson']:.4f}")
    
    plt.suptitle('Comparación Visual: Original vs CLAHE Optimizado', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(reporte_dir, "comparacion_top_resultados.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparación visual guardada en: {reporte_dir}")
    
    # Generar reporte textual detallado
    ruta_reporte = os.path.join(reporte_dir, "reporte_detallado.txt")
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE FINAL DE OPTIMIZACIÓN CLAHE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total de experimentos analizados: {len(df)}\n")
        f.write(f"Mejores {len(top_ids)} iteraciones seleccionadas\n\n")
        
        f.write("─" * 80 + "\n")
        f.write("RANKING DE MEJORES RESULTADOS\n")
        f.write("─" * 80 + "\n\n")
        
        for idx, (_, row) in enumerate(df_top.iterrows(), 1):
            f.write(f"{idx}. EXPERIMENTO ID {int(row['ID_Experimento'])}\n")
            f.write(f"   {'─' * 70}\n")
            f.write(f"   Parámetros:\n")
            f.write(f"     • Clip Limit (α): {row['ClipLimit']:.2f}\n")
            f.write(f"     • Tile Size (ω): {int(row['TileSize'])} x {int(row['TileSize'])}\n\n")
            f.write(f"   Métricas de Calidad:\n")
            f.write(f"     • Entropía: {row['Entropia']:.4f}\n")
            f.write(f"     • Contraste Local Promedio: {row['Contraste_Local_Promedio']:.4f}\n")
            f.write(f"     • Nitidez de Borde: {row['Nitidez_Borde']:.4f}\n")
            f.write(f"     • Contraste de Michelson: {row['Contraste_Michelson']:.4f}\n\n")
            
            # Evaluación cualitativa automática
            f.write(f"   Evaluación Cualitativa:\n")
            if row['Contraste_Local_Promedio'] > df['Contraste_Local_Promedio'].quantile(0.75):
                f.write(f"     ✓ Excelente contraste local - Delimitación clara de estructuras\n")
            else:
                f.write(f"     • Contraste local moderado\n")
            
            if row['Nitidez_Borde'] > df['Nitidez_Borde'].quantile(0.75):
                f.write(f"     ✓ Alta nitidez de bordes - Estructuras bien definidas\n")
            else:
                f.write(f"     • Nitidez de bordes moderada\n")
            
            if row['Entropia'] > df['Entropia'].quantile(0.75):
                f.write(f"     ✓ Alta entropía - Rica distribución de intensidades\n")
            else:
                f.write(f"     • Entropía moderada\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Reporte detallado guardado en: {ruta_reporte}")
    
    return reporte_dir, df_top

def determinar_combinacion_optima(df_top):
    """E. CONCLUSIÓN - Determinar y reportar la combinación óptima."""
    print(f"\n{'=' * 70}")
    print("CONCLUSIÓN: COMBINACIÓN ÓPTIMA")
    print(f"{'=' * 70}")
    
    # Seleccionar el mejor resultado (primera fila ya está ordenada)
    mejor = df_top.iloc[0]
    
    alpha_optimo = mejor['ClipLimit']
    omega_optimo = int(mejor['TileSize'])
    
    print(f"\n🏆 PARÁMETROS ÓPTIMOS IDENTIFICADOS:")
    print(f"   α* (Clip Limit): {alpha_optimo:.2f}")
    print(f"   ω* (Tile Size): {omega_optimo} x {omega_optimo}")
    print(f"\n📊 MÉTRICAS DEL RESULTADO ÓPTIMO:")
    print(f"   • Entropía: {mejor['Entropia']:.4f}")
    print(f"   • Contraste Local Promedio: {mejor['Contraste_Local_Promedio']:.4f}")
    print(f"   • Nitidez de Borde: {mejor['Nitidez_Borde']:.4f}")
    print(f"   • Contraste de Michelson: {mejor['Contraste_Michelson']:.4f}")
    
    print(f"\n📁 Experimento ID: {int(mejor['ID_Experimento'])}")
    print(f"   Ruta: ./resultados_clahe/iteracion_{int(mejor['ID_Experimento']):04d}/")
    
    print(f"\n{'=' * 70}")
    print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print(f"{'=' * 70}\n")
    
    return {
        'alpha_optimo': alpha_optimo,
        'omega_optimo': omega_optimo,
        'id_experimento': int(mejor['ID_Experimento']),
        'metricas': {
            'entropia': mejor['Entropia'],
            'contraste_local': mejor['Contraste_Local_Promedio'],
            'nitidez_borde': mejor['Nitidez_Borde'],
            'contraste_michelson': mejor['Contraste_Michelson']
        }
    }

def main():
    """Función principal que ejecuta todo el flujo de análisis."""
    
    # Configuración
    RUTA_CSV = "./resultados_clahe/resultados_maestros.csv"
    RESULTADOS_DIR = "./resultados_clahe/"
    OUTPUT_DIR = "./resultados_clahe/"
    METRICA_PRINCIPAL = 'Contraste_Local_Promedio'
    TOP_N_PRESELECCION = 10
    TOP_N_EVALUACION = 5
    
    # A. CARGA DE DATOS
    df = cargar_datos_maestros(RUTA_CSV)
    if df is None:
        return
    
    # B. PRESELECCIÓN OBJETIVA
    top_resultados = preseleccion_objetiva(df, METRICA_PRINCIPAL, TOP_N_PRESELECCION)
    
    # C. VISUALIZACIÓN DE RESULTADOS
    graficos_dir = visualizar_relaciones_parametros(df, OUTPUT_DIR)
    
    # D. EVALUACIÓN SUBJETIVA
    top_ids = top_resultados.head(TOP_N_EVALUACION)['ID_Experimento'].tolist()
    reporte_dir, df_top = evaluar_subjetivamente(df, top_ids, RESULTADOS_DIR, OUTPUT_DIR)
    
    # E. CONCLUSIÓN
    resultado_optimo = determinar_combinacion_optima(df_top)
    
    # Guardar resultado óptimo en JSON
    ruta_optimo = os.path.join(OUTPUT_DIR, "parametros_optimos.json")
    with open(ruta_optimo, 'w', encoding='utf-8') as f:
        json.dump(resultado_optimo, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ Parámetros óptimos guardados en: {ruta_optimo}")
    print(f"✓ Gráficos de análisis en: {graficos_dir}")
    print(f"✓ Reporte final en: {reporte_dir}")

if __name__ == "__main__":
    main()