import sqlite3
import os
from geotext import GeoText

STOPWORDS_GEO = {
    'como', 'un', 'una', 'el', 'la', 'los', 'las', 'de', 'del',
    'en', 'por', 'para', 'con', 'sin', 'que', 'a', 'o', 'y', 'e', 'u',
    'si', 'no', 'san', 'santa', 'es', 'al', 'su', 'sus', 'te', 'me', 'se',
    'lo', 'le', 'nos', 'os', 'mi', 'mis', 'tu', 'tus', 'ya', 'ha', 'he',
    'has', 'han', 'hay', 'muy', 'más', 'mas', 'pero', 'porque', 'cuando',
    'donde', 'quien', 'cual', 'este', 'esta', 'estos', 'estas', 'ese', 'esa',
    'esos', 'esas', 'aquel', 'aquella', 'todo', 'toda', 'todos', 'todas',
    'nada', 'algo', 'bien', 'mal', 'tan', 'tal', 'ser', 'estar', 'ir', 'ver',
    'hacer', 'tener', 'decir', 'poder', 'dar', 'saber', 'querer', 'creer',
    'parecer', 'venir', 'pasar', 'llegar', 'dejar', 'poner', 'tomar', 'vivir',
    'hablar', 'dar', 'solo', 'solamente', 'entonces', 'luego', 'antes',
    'después', 'siempre', 'nunca', 'hoy', 'ayer', 'mañana', 'ahora',
    'aquí', 'ahí', 'allí', 'acá', 'allá', 'arriba', 'abajo', 'cerca',
    'lejos', 'dentro', 'fuera', 'dos', 'tres', 'hola', 'saludos', 'saludo',
    'asi', 'así', 'amo', 'mar', 'paz', 'sur', 'norte', 'este', 'oeste', 'fin',
    'dios', 'rey', 'ley', 'luz', 'sol', 'flor', 'florida', 'pan', 'oro',
    'mira', 'mirar', 'hace', 'fue', 'soy', 'eres', 'somos', 'son'
}

# Diccionario de normalización para agrupar variaciones y aliases repetitivos
ALIASES_GEO = {
    'Salvador': 'El Salvador',
    'San Salvador': 'El Salvador',
    'San Salvador, El Salvador': 'El Salvador',
    'E.S.': 'El Salvador',
    'E. S.': 'El Salvador',
    'San Salvador City': 'El Salvador',
    'Sv': 'El Salvador'
    # NOTA: 'El' y 'San' fueron eliminados — son palabras comunes en español
    # que generaban falsos positivos en prácticamente cualquier comentario.
    # NOTA: 'Bukele' y 'Nayib' fueron eliminados — son nombres de persona,
    # no entidades geográficas reconocibles por GeoText.
}

def normalizar_lugar(lugar):
    lugar = lugar.strip()
    lugar_title = lugar.title()
    # Buscar en el diccionario de mapeo
    if lugar_title in ALIASES_GEO:
        return ALIASES_GEO[lugar_title]
    if lugar in ALIASES_GEO:
        return ALIASES_GEO[lugar]
    return lugar_title

def extraer_ubicaciones(db_path):
    if not os.path.exists(db_path):
        print(f"❌ Error: La base de datos {db_path} no existe.")
        return

    print(f"🌍 Extrayendo ubicaciones geográficas en {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE comentarios ADD COLUMN ubicacion TEXT")
    except sqlite3.OperationalError:
        pass

    # Solo procesa filas sin ubicación asignada
    cursor.execute("SELECT rowid, texto FROM comentarios WHERE ubicacion IS NULL")
    rows = cursor.fetchall()

    if not rows:
        print("✅ Todas las ubicaciones ya están extraídas.")
        conn.close()
        return

    updates = []

    for rowid, texto in rows:
        ubicacion = "sin info"
        if texto:
            places = GeoText(str(texto))
            cities_countries = list(set(places.cities + places.countries))

            valid_places = []
            for place in cities_countries:
                # Normalizar a minúsculas para comparar contra STOPWORDS_GEO
                if place.lower() not in STOPWORDS_GEO:
                    norm_place = normalizar_lugar(place)
                    valid_places.append(norm_place)

            if valid_places:
                ubicacion = ', '.join(list(set(valid_places)))

        updates.append((ubicacion, rowid))
    
    cursor.executemany("UPDATE comentarios SET ubicacion = ? WHERE rowid = ?", updates)
    conn.commit()
    conn.close()
    
    print(f"✅ Extracción de ubicaciones completada. Se actualizaron {len(updates)} registros.")
