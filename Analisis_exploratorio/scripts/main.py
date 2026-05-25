import os
import sys

# Permite ejecutar main.py desde cualquier directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from descarga_masiva import descargar_comentarios
from limpieza_texto import limpiar_base
from extract_locations import extraer_ubicaciones
from tokenizacion import tokenizar_textos
from clasificador_sentimientos import clasificar_textos
from export_para_looker import exportar_a_looker

# --- CONFIGURACIÓN ---
# La API Key se lee desde variable de entorno para no exponerla en el código.
# Antes de ejecutar: export YOUTUBE_API_KEY="tu_clave_aqui"
API_KEY = os.getenv("YOUTUBE_API_KEY", "")
VIDEO_ID = "pWnndG1K5Hg"
OBJETIVO = 5000

# Paths resueltos relativos a la ubicación de este script
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_ORIGINAL = os.path.join(_BASE_DIR, "../data/original.db")
DB_LIMPIA   = os.path.join(_BASE_DIR, "../data/base_limpia.db")

def main():
    print("=== INICIANDO PIPELINE DE PROCESAMIENTO ===")

    os.makedirs(os.path.join(_BASE_DIR, "../data"), exist_ok=True)

    # --- PASO 1: Descarga de comentarios ---
    if not os.path.exists(DB_ORIGINAL):
        if not API_KEY:
            print("❌ No se encontró YOUTUBE_API_KEY en las variables de entorno. "
                  "Ejecuta: export YOUTUBE_API_KEY='tu_clave'")
            return
        descargar_comentarios(VIDEO_ID, API_KEY, OBJETIVO, DB_ORIGINAL)
    else:
        print(f"✅ {DB_ORIGINAL} ya existe, omitiendo descarga.")

    # --- PASO 2: Limpieza de Texto ---
    limpiar_base(DB_ORIGINAL, DB_LIMPIA)

    # --- PASO 3: Extracción Geográfica ---
    extraer_ubicaciones(DB_LIMPIA)

    # --- PASO 4: Tokenización ---
    tokenizar_textos(DB_LIMPIA)

    # --- PASO 5: Análisis de Sentimientos ---
    clasificar_textos(DB_LIMPIA)

    # --- PASO 6: Exportar CSV para Looker Studio ---
    exportar_a_looker()

    print("=== PIPELINE FINALIZADO CON ÉXITO ===")

if __name__ == "__main__":
    main()
