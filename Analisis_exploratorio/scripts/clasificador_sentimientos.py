import sqlite3
import os
import logging
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pysentimiento import create_analyzer

BATCH_SIZE = 32       # Comentarios por lote para inferencia paralela en el modelo
CHECKPOINT_CADA = 500  # Guardar en BD cada N resultados procesados

MAPA_SENTIMIENTO = {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}

def _guardar_checkpoint(cursor, conn, updates):
    cursor.executemany(
        "UPDATE comentarios SET sentimiento = ?, probabilidad = ? WHERE rowid = ?",
        updates
    )
    conn.commit()

def clasificar_textos(db_path):
    if not os.path.exists(db_path):
        print(f"❌ Error: La base de datos {db_path} no existe.")
        return

    print("🧠 Inicializando modelo RoBERTuito (puede demorar la primera vez)...")
    analyzer = create_analyzer(task="sentiment", lang="es")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for col in ["sentimiento", "probabilidad"]:
        try:
            cursor.execute(f"ALTER TABLE comentarios ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    # Usa texto_limpio (ya procesado por limpieza_texto.py) para consistencia con el pipeline
    cursor.execute("""
        SELECT rowid, texto_limpio, texto
        FROM comentarios
        WHERE sentimiento IS NULL OR sentimiento = 'sin info'
    """)
    rows = cursor.fetchall()

    if not rows:
        print("✅ Todos los comentarios ya están clasificados.")
        conn.close()
        return

    print(f"🤖 Clasificando {len(rows)} comentarios en lotes de {BATCH_SIZE}...")

    updates_pendientes = []
    total_procesados = 0

    with tqdm(total=len(rows), desc="Clasificando", unit="comentarios") as barra:
        for i in range(0, len(rows), BATCH_SIZE):
            lote = rows[i:i + BATCH_SIZE]
            rowids = [r[0] for r in lote]

            # Usa texto_limpio si existe y no está vacío, si no cae a texto original
            textos = [
                str(r[1]).strip() if r[1] and str(r[1]).strip() else str(r[2] or "")
                for r in lote
            ]

            try:
                resultados = analyzer.predict(textos)
                for rowid, resultado in zip(rowids, resultados):
                    sentimiento = MAPA_SENTIMIENTO.get(resultado.output, "neutro")
                    probabilidad = float(resultado.probas.get(resultado.output, 0.0))
                    updates_pendientes.append((sentimiento, probabilidad, rowid))
            except Exception as e:
                logging.warning(f"Error procesando lote en rowid {rowids[0]}-{rowids[-1]}: {e}")
                for rowid in rowids:
                    updates_pendientes.append(("neutro", 0.0, rowid))

            barra.update(len(lote))
            total_procesados += len(lote)

            # Checkpointing: guardar periódicamente para no perder trabajo
            if total_procesados % CHECKPOINT_CADA == 0:
                _guardar_checkpoint(cursor, conn, updates_pendientes)
                updates_pendientes = []
                barra.set_postfix({"guardados": total_procesados})

    # Guardar lo que quedó en el último lote parcial
    if updates_pendientes:
        _guardar_checkpoint(cursor, conn, updates_pendientes)

    conn.close()
    print(f"✅ Clasificación completada. {total_procesados} comentarios procesados.")

if __name__ == "__main__":
    clasificar_textos('../data/base_limpia.db')
