import sqlite3
import time
import random
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

CHECKPOINT_CADA = 500  # Guarda en BD cada N comentarios descargados

def _inicializar_tabla(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comentarios (
            autor TEXT,
            texto TEXT,
            likes INTEGER,
            fecha TEXT,
            UNIQUE(autor, fecha)
        )
    """)
    conn.commit()

def _guardar_lote(conn, lote):
    conn.executemany(
        "INSERT OR IGNORE INTO comentarios (autor, texto, likes, fecha) VALUES (?,?,?,?)",
        lote
    )
    conn.commit()

def descargar_comentarios(video_id, api_key, objetivo, db_dest='data/original.db'):
    youtube = build('youtube', 'v3', developerKey=api_key)
    next_page_token = None
    total_guardados = 0
    lote_actual = []

    print(f"🚀 Iniciando descarga de hasta {objetivo} comentarios...")
    print(f"📺 Video: https://www.youtube.com/watch?v={video_id}")

    conn = sqlite3.connect(db_dest)
    _inicializar_tabla(conn)

    try:
        while total_guardados + len(lote_actual) < objetivo:
            try:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()
            except HttpError as e:
                if e.resp.status == 403:
                    print(f"❌ Cuota de API agotada o acceso denegado: {e}")
                elif e.resp.status == 404:
                    print(f"❌ Video no encontrado o comentarios desactivados: {e}")
                else:
                    print(f"❌ Error HTTP {e.resp.status}: {e}")
                break

            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                lote_actual.append((
                    snippet['authorDisplayName'],
                    snippet['textOriginal'],
                    snippet['likeCount'],
                    snippet['publishedAt']
                ))
                if total_guardados + len(lote_actual) >= objetivo:
                    break

            # Checkpointing: guardar cada CHECKPOINT_CADA comentarios
            if len(lote_actual) >= CHECKPOINT_CADA:
                _guardar_lote(conn, lote_actual)
                total_guardados += len(lote_actual)
                lote_actual = []
                print(f"💾 Checkpoint: {total_guardados} comentarios guardados...")

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                print("ℹ️  Fin de los comentarios disponibles en el video.")
                break

            time.sleep(random.uniform(1, 2))

    finally:
        # Guardar lo que quede en el lote aunque el proceso se haya interrumpido
        if lote_actual:
            _guardar_lote(conn, lote_actual)
            total_guardados += len(lote_actual)
        conn.close()

    if total_guardados > 0:
        print(f"🎉 Descarga finalizada. {total_guardados} comentarios guardados en {db_dest}.")
    else:
        print("⚠️  No se descargaron comentarios.")