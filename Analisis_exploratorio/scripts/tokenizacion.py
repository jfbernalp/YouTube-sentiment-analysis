import sqlite3
import os
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def tokenizar_textos(db_path):
    if not os.path.exists(db_path):
        print(f"❌ Error: La base de datos {db_path} no existe.")
        return

    print(f"🔠 Aplicando tokenización NLP en {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE comentarios ADD COLUMN tokens TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Solo procesa filas que aún no tienen tokens
    cursor.execute("SELECT rowid, texto_limpio FROM comentarios WHERE tokens IS NULL OR tokens = ''")
    rows = cursor.fetchall()

    if not rows:
        print("✅ Todos los comentarios ya están tokenizados.")
        conn.close()
        return

    updates = []
    for rowid, texto_limpio in rows:
        tokens_str = ""
        if texto_limpio and str(texto_limpio).strip():
            # Tokenización real con nltk: separa correctamente puntuación y contracciones
            tokens = nltk.word_tokenize(str(texto_limpio), language='spanish')
            tokens_str = ",".join(tokens)
        updates.append((tokens_str, rowid))

    cursor.executemany("UPDATE comentarios SET tokens = ? WHERE rowid = ?", updates)
    conn.commit()
    conn.close()

    print(f"✅ Tokenización completada. Se procesaron {len(updates)} comentarios.")
