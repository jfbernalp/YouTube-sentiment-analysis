import sqlite3
import pandas as pd
import re
import os
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('spanish'))

# spaCy para lematización en español
try:
    _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
except OSError:
    _nlp = None
    print("⚠️  Modelo spaCy 'es_core_news_sm' no encontrado. "
          "Instálalo con: python -m spacy download es_core_news_sm")

# Patrón que preserva letras españolas (tildes, eñe, diéresis)
_PATRON_NO_ALFA = re.compile(r'[^a-záéíóúüñ\s]', re.IGNORECASE)
_PATRON_ESPACIOS = re.compile(r'\s+')
_PATRON_URLS = re.compile(r'https?://\S+|www\.\S+', re.MULTILINE)

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = _PATRON_URLS.sub('', texto)
    texto = _PATRON_NO_ALFA.sub(' ', texto)
    texto = _PATRON_ESPACIOS.sub(' ', texto).strip()

    palabras = [w for w in texto.split() if w not in stop_words and len(w) > 2]

    if _nlp is None:
        return " ".join(palabras)

    # Lematización: convierte cada palabra a su forma base (ej. "corriendo" → "correr")
    doc = _nlp(" ".join(palabras))
    lemas = [token.lemma_ for token in doc if token.lemma_ not in stop_words and len(token.lemma_) > 2]
    return " ".join(lemas)

def limpiar_base(db_origen, db_destino):
    if not os.path.exists(db_origen):
        print(f"❌ Error: La base de origen {db_origen} no existe.")
        return

    print(f"🧹 Iniciando limpieza desde {db_origen} hacia {db_destino}...")
    conn_origen = sqlite3.connect(db_origen)
    df = pd.read_sql_query("SELECT * FROM comentarios", conn_origen)
    conn_origen.close()

    columnas_requeridas = ['autor', 'texto', 'likes', 'fecha']
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        print(f"❌ La base de origen no tiene las columnas requeridas: {faltantes}")
        return

    df['texto_limpio'] = df['texto'].apply(limpiar_texto)

    conn_destino = sqlite3.connect(db_destino)

    # Si ya hay datos procesados (tokens, sentimiento), los preservamos
    columnas_a_preservar = {}
    if os.path.exists(db_destino):
        try:
            df_existente = pd.read_sql_query(
                "SELECT rowid, tokens, sentimiento, probabilidad, ubicacion FROM comentarios",
                conn_destino
            )
            for col in ['tokens', 'sentimiento', 'probabilidad', 'ubicacion']:
                if col in df_existente.columns:
                    columnas_a_preservar[col] = df_existente[col].values
        except Exception:
            pass

    df.to_sql('comentarios', conn_destino, if_exists='replace', index=False)

    # Restaurar columnas procesadas previamente
    for col, valores in columnas_a_preservar.items():
        if len(valores) == len(df):
            conn_destino.execute(f"ALTER TABLE comentarios ADD COLUMN {col} TEXT") if col not in df.columns else None
            for i, val in enumerate(valores):
                conn_destino.execute(
                    f"UPDATE comentarios SET {col} = ? WHERE rowid = ?", (val, i + 1)
                )
            conn_destino.commit()

    conn_destino.close()
    print(f"✅ Limpieza y lematización completadas. Datos guardados en {db_destino}.")