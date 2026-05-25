import sqlite3
import pandas as pd
import os

def exportar_a_looker():
    db_path = '../data/base_limpia.db'
    csv_path = '../data/base_limpia_looker.csv'
    
    if not os.path.exists(db_path):
        print(f"❌ Error: La base de datos {db_path} no existe.")
        return

    print("📊 Preparando datos para Looker Studio...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM comentarios", conn)
    conn.close()

    # 1. Transformar Fecha (Eliminar T y Z de ISO 8601)
    if 'fecha' in df.columns:
        # Convertir a datetime y luego formatear a string estándar (YYYY-MM-DD HH:MM:SS)
        df['fecha_formateada'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d %H:%M:%S')
        # Alternativa para Looker (solo la fecha sin hora si se prefiere agrupar por días)
        df['solo_dia'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')
        
    # 2. Arreglar Ubicación (Borrar 'sin info')
    if 'ubicacion' in df.columns:
        # Reemplazar 'sin info' con nulo puro para que Looker lo ignore en el mapa
        df['ubicacion'] = df['ubicacion'].replace('sin info', None)
        
    # Guardar en CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ Exportación completada. Archivo guardado exitosamente en: {csv_path}")

if __name__ == "__main__":
    exportar_a_looker()
