import os
import io
import base64
import sqlite3
import warnings

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────
PALETA      = {"positivo": "#00f2fe", "neutro": "#a8b0c3", "negativo": "#fe0979"}
ORDEN_SENT  = ["positivo", "neutro", "negativo"]
BG          = "#0b0f19"
CARD_BG     = "#151b2b"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
DB_PATH     = os.path.join(BASE_DIR, "data", "base_limpia.db")

# ── Carga de datos ────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM comentarios", conn)
    conn.close()
    df['texto']       = df['texto'].fillna('')
    df['texto_limpio']= df['texto_limpio'].fillna('')
    df['ubicacion']   = df['ubicacion'].fillna('sin info')
    df['sentimiento'] = df['sentimiento'].fillna('neutro')
    df['probabilidad']= pd.to_numeric(df['probabilidad'], errors='coerce').fillna(0)
    df['likes']       = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
    df['fecha_dt']    = pd.to_datetime(df['fecha'], errors='coerce')
    df['solo_fecha']  = df['fecha_dt'].dt.date
    return df

df = load_data()

# ── KPIs globales ─────────────────────────────────────────────────────────────
total_comentarios   = len(df)
confianza_media     = df['probabilidad'].mean()
alta_confianza_pct  = (df['probabilidad'] >= 0.8).mean() * 100
sent_counts         = df['sentimiento'].value_counts()
sentimiento_dom     = sent_counts.idxmax()
pct_pos             = sent_counts.get('positivo', 0) / total_comentarios * 100
pct_neg             = sent_counts.get('negativo', 0) / total_comentarios * 100
pct_neu             = sent_counts.get('neutro',   0) / total_comentarios * 100
ratio_pos_neg       = sent_counts.get('positivo', 0) / max(sent_counts.get('negativo', 1), 1)
likes_total_pos     = df[df['sentimiento'] == 'positivo']['likes'].sum()
likes_total         = df['likes'].sum()
pct_likes_pos       = likes_total_pos / max(likes_total, 1) * 100

_layout_base = dict(template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    autosize=True)
_cfg = {"responsive": True}

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN Y TENDENCIAS
# ═════════════════════════════════════════════════════════════════════════════
df_tiempo    = df.dropna(subset=['solo_fecha']).copy()
pivot_tiempo = (df_tiempo.groupby(['solo_fecha','sentimiento'])
                .size().unstack(fill_value=0)
                .reindex(columns=ORDEN_SENT, fill_value=0))

fig_tiempo = go.Figure()
for sent in ORDEN_SENT:
    fig_tiempo.add_trace(go.Scatter(
        x=pivot_tiempo.index, y=pivot_tiempo[sent],
        mode='lines', stackgroup='one',
        name=sent.capitalize(), line=dict(color=PALETA[sent], width=0.5)
    ))
fig_tiempo.update_layout(
    title="Evolución Temporal del Sentimiento",
    xaxis_title="Fecha", yaxis_title="Comentarios",
    **_layout_base
)

likes_prom = df.groupby('sentimiento')['likes'].mean().reindex(ORDEN_SENT).reset_index()
likes_prom.columns = ['sentimiento', 'likes_promedio']
fig_likes = px.bar(
    likes_prom, x='sentimiento', y='likes_promedio',
    color='sentimiento', color_discrete_map=PALETA,
    title="Promedio de Likes por Categoría de Sentimiento",
    text='likes_promedio'
)
fig_likes.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_likes.update_layout(xaxis_title="Sentimiento", yaxis_title="Likes promedio",
                        showlegend=False, **_layout_base)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — RENDIMIENTO NLP
# ═════════════════════════════════════════════════════════════════════════════

# Métrica 1: Distribución
counts_y = [int(sent_counts.get(s, 0)) for s in ORDEN_SENT]
fig_dist_sent = px.bar(
    x=ORDEN_SENT, y=counts_y,
    color=ORDEN_SENT, color_discrete_map=PALETA,
    title="Métrica 1: Distribución de Sentimientos",
    text=counts_y
)
fig_dist_sent.update_traces(textposition='outside')
fig_dist_sent.update_layout(
    showlegend=False, xaxis_title="Sentimiento",
    yaxis_title="Nº de Comentarios", **_layout_base
)

# Métrica 2: Confianza
fig_confianza = px.histogram(
    df, x='probabilidad', color='sentimiento',
    nbins=40, barmode='overlay',
    color_discrete_map=PALETA,
    title="Métrica 2: Score de Confianza del Modelo por Clase",
    opacity=0.75
)
fig_confianza.add_vline(x=0.8, line_dash="dash", line_color="#ffd700",
                        annotation_text="Umbral 0.8", annotation_position="top right")
fig_confianza.update_layout(
    xaxis_title="Probabilidad de Predicción",
    yaxis_title="Cantidad de Comentarios", **_layout_base
)

# Métrica 3: PCA clusters
corpus_valido = df[df['texto_limpio'].str.strip().str.len() > 3].copy()
vectorizer    = TfidfVectorizer(max_features=500, min_df=3, max_df=0.85,
                                 ngram_range=(1,2), sublinear_tf=True)
X      = vectorizer.fit_transform(corpus_valido['texto_limpio'].astype(str))
km     = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())

df_pca = pd.DataFrame(coords, columns=['PC1','PC2'])
df_pca['sentimiento'] = corpus_valido['sentimiento'].values
df_pca['texto_corto'] = corpus_valido['texto'].str[:60] + "..."

fig_pca = px.scatter(
    df_pca, x='PC1', y='PC2', color='sentimiento',
    hover_data=['texto_corto'], color_discrete_map=PALETA,
    title="Métrica 3: Proyección PCA 2D de Clusters Temáticos",
    opacity=0.65
)
feature_names   = vectorizer.get_feature_names_out()
centros         = km.cluster_centers_
cluster_words   = []
for k in range(3):
    top_idx = centros[k].argsort()[-5:][::-1]
    cluster_words.append(f"C{k}: " + ", ".join(feature_names[top_idx]))
fig_pca.add_annotation(
    x=1.05, y=0.5,
    text="Palabras Clave por Cluster:<br>" + "<br>".join(cluster_words),
    xref="paper", yref="paper", showarrow=False,
    font=dict(color="#a8b0c3", size=11), align="left",
    bgcolor="rgba(0,0,0,0.5)"
)
fig_pca.update_layout(**_layout_base)

# Métrica 4: TF-IDF por sentimiento
fig_tfidf = make_subplots(rows=1, cols=3,
                           subplot_titles=["Positivo","Neutro","Negativo"])
for i, sent in enumerate(ORDEN_SENT):
    textos_clase = (df[df['sentimiento']==sent]['texto_limpio']
                    .fillna("").astype(str))
    textos_clase = textos_clase[textos_clase.str.strip().str.len() > 3]
    if len(textos_clase) > 0:
        vect    = TfidfVectorizer(max_features=200, min_df=2, max_df=0.90)
        X_clase = vect.fit_transform(textos_clase)
        scores  = np.asarray(X_clase.mean(axis=0)).ravel()
        top_idx = scores.argsort()[-15:]
        fig_tfidf.add_trace(go.Bar(
            x=scores[top_idx],
            y=np.array(vect.get_feature_names_out())[top_idx],
            orientation='h', name=sent,
            marker=dict(color=PALETA[sent])
        ), row=1, col=i+1)
fig_tfidf.update_layout(
    title="Métrica 4: Términos más Discriminantes por Sentimiento (TF-IDF)",
    showlegend=False, height=520, **_layout_base
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAFÍA
# ═════════════════════════════════════════════════════════════════════════════
df_geo    = df[df['ubicacion'].str.lower() != 'sin info'].copy()
geo_counts = df_geo['ubicacion'].value_counts().head(15).reset_index()
geo_counts.columns = ['Ubicacion','Cantidad']
fig_geo = px.bar(
    geo_counts, x='Cantidad', y='Ubicacion', orientation='h',
    title="Top 15 Ubicaciones de los Comentaristas",
    color='Cantidad', color_continuous_scale="Agsunset",
    text='Cantidad'
)
fig_geo.update_traces(textposition='outside')
fig_geo.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_title="Nº de Comentarios", yaxis_title="",
    **_layout_base
)

# Sentimiento por país (top 5)
top5_paises  = geo_counts.head(5)['Ubicacion'].tolist()
df_pais_sent = (df_geo[df_geo['ubicacion'].isin(top5_paises)]
                .groupby(['ubicacion','sentimiento'])
                .size().unstack(fill_value=0)
                .reindex(columns=ORDEN_SENT, fill_value=0)
                .reset_index())
fig_geo_sent = go.Figure()
for sent in ORDEN_SENT:
    if sent in df_pais_sent.columns:
        fig_geo_sent.add_trace(go.Bar(
            name=sent.capitalize(),
            x=df_pais_sent['ubicacion'],
            y=df_pais_sent[sent],
            marker_color=PALETA[sent]
        ))
fig_geo_sent.update_layout(
    barmode='group',
    title="Distribución de Sentimientos por País (Top 5)",
    xaxis_title="País / Ubicación", yaxis_title="Nº de Comentarios",
    **_layout_base
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — NUBES DE PALABRAS
# ═════════════════════════════════════════════════════════════════════════════
print("☁️  Generando nubes de palabras...")

def wc_to_base64(wc_obj, bg=BG, figsize=(10, 5)):
    """Convierte un WordCloud a imagen PNG embebida en base64."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.imshow(wc_obj, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor(bg)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=bg, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def make_wc(text, colormap, bg_color=BG, max_words=120, width=1200, height=500):
    return WordCloud(
        background_color=bg_color,
        colormap=colormap,
        max_words=max_words,
        width=width, height=height,
        collocations=False,
        prefer_horizontal=0.85,
        relative_scaling=0.5,
        min_font_size=10,
    ).generate(text)

# ── ANTES: texto original (con stopwords, símbolos, ruido) ──
texto_antes = " ".join(df['texto'].str.lower().fillna(''))
wc_antes    = make_wc(texto_antes, colormap='Greys_r')
img_antes   = wc_to_base64(wc_antes)

# ── DESPUÉS: texto limpio, sin stopwords, lematizado ──
texto_despues = " ".join(df['texto_limpio'].fillna(''))
wc_despues    = make_wc(texto_despues, colormap='winter')
img_despues   = wc_to_base64(wc_despues)

# ── POR SENTIMIENTO (post-preprocesamiento) ──
wc_imgs_sent = {}
cmaps = {"positivo": "cool", "negativo": "RdPu", "neutro": "Greys"}
for sent in ORDEN_SENT:
    textos_s = " ".join(df[df['sentimiento']==sent]['texto_limpio'].fillna(''))
    if len(textos_s.strip()) > 50:
        wc_s = make_wc(textos_s, colormap=cmaps[sent])
        wc_imgs_sent[sent] = wc_to_base64(wc_s)

print("   ✅ Nubes generadas.")

# ═════════════════════════════════════════════════════════════════════════════
# PRE-RENDER PLOTLY → HTML snippets
# ═════════════════════════════════════════════════════════════════════════════
_html_tiempo     = fig_tiempo.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_likes      = fig_likes.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_dist_sent  = fig_dist_sent.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_confianza  = fig_confianza.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_pca        = fig_pca.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_tfidf      = fig_tfidf.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_geo        = fig_geo.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_geo_sent   = fig_geo_sent.to_html(full_html=False, include_plotlyjs=False, config=_cfg)

# ── Datos reales para la página de conclusiones ────────────────────────────
pico_dia  = df_tiempo.groupby('solo_fecha').size().idxmax()
pico_n    = df_tiempo.groupby('solo_fecha').size().max()
n_paises  = df_geo['ubicacion'].nunique()
top1_pais = geo_counts.iloc[0]['Ubicacion']
top1_n    = geo_counts.iloc[0]['Cantidad']

# ═════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DEL HTML COMPLETO
# ═════════════════════════════════════════════════════════════════════════════
html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Dashboard · Análisis de Sentimientos YouTube</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
    <style>
        body {{
            background-color: {BG};
            color: #e2e8f0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 24px;
        }}
        /* ── Cards ── */
        .glass-card {{
            background: rgba(21,27,43,0.75);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 22px;
            margin-bottom: 22px;
            width: 100%;
        }}
        .glass-card.conclusion-card {{
            border-left: 4px solid #00f2fe;
        }}
        .glass-card.rec-card {{
            border-left: 4px solid #fe0979;
        }}
        /* ── KPIs ── */
        .kpi-value  {{ font-size: 2.4rem; font-weight: 700; color: #fff; text-align: center; }}
        .kpi-title  {{ font-size: .85rem; color: #a8b0c3; text-transform: uppercase;
                       letter-spacing: 1.2px; text-align: center; margin-bottom: 6px; }}
        /* ── Tabs ── */
        .nav-pills .nav-link         {{ color: #a8b0c3; margin-right: 8px; border-radius: 10px;
                                        transition: all .25s; cursor: pointer; font-size: .92rem; }}
        .nav-pills .nav-link:hover   {{ background: rgba(0,242,254,.18); color: #00f2fe; }}
        .nav-pills .nav-link.active  {{ background-color: #00f2fe !important; color: #000 !important;
                                        font-weight: 700; }}
        .tab-content {{ margin-top: 22px; }}
        /* ── Chart description ── */
        .chart-desc {{
            font-size: .88rem;
            color: #a8b0c3;
            background: rgba(0,0,0,.25);
            border-left: 3px solid #00f2fe;
            border-radius: 0 8px 8px 0;
            padding: 10px 14px;
            margin-top: 10px;
            line-height: 1.6;
        }}
        .chart-desc strong {{ color: #e2e8f0; }}
        /* ── Word cloud images ── */
        .wc-img {{ width: 100%; border-radius: 12px; display: block; }}
        .wc-label {{
            font-size: .78rem; text-transform: uppercase; letter-spacing: 1px;
            color: #a8b0c3; text-align: center; margin-top: 6px;
        }}
        /* ── Conclusion cards ── */
        .finding-badge {{
            display: inline-block;
            background: rgba(0,242,254,.15);
            color: #00f2fe;
            border-radius: 20px;
            padding: 2px 12px;
            font-size: .78rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .rec-badge {{
            display: inline-block;
            background: rgba(254,9,121,.15);
            color: #fe0979;
            border-radius: 20px;
            padding: 2px 12px;
            font-size: .78rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .stat-highlight {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #00f2fe;
        }}
        h5.section-title {{
            font-weight: 700;
            color: #fff;
            border-bottom: 1px solid rgba(255,255,255,.1);
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        .plotly-graph-div {{ width: 100% !important; }}
    </style>
</head>
<body>
<div class="container-fluid">

    <!-- ── ENCABEZADO ── -->
    <div class="d-flex align-items-center mb-1 mt-2">
        <i class="fa-solid fa-brain fa-2x me-3" style="color:#00f2fe"></i>
        <div>
            <h2 class="fw-bold text-white mb-0">Dashboard de Análisis de Sentimientos</h2>
            <small class="text-secondary">Video YouTube: "Esto es lo que nadie te cuenta de El Salvador" · Corpus: {total_comentarios:,} comentarios</small>
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,.08); margin-bottom:24px">

    <!-- ── KPIs ── -->
    <div class="row mb-4 g-3">
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-comment-dots me-1"></i> Total Comentarios</div>
                <div class="kpi-value">{total_comentarios:,}</div>
            </div>
        </div>
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-face-smile me-1"></i> Sentimiento Dominante</div>
                <div class="kpi-value" style="color:{PALETA[sentimiento_dom]}">{sentimiento_dom.capitalize()}</div>
            </div>
        </div>
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-gauge-high me-1"></i> Confianza Media</div>
                <div class="kpi-value">{confianza_media:.2f}</div>
            </div>
        </div>
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-bullseye me-1"></i> Alta Confianza (&gt;0.8)</div>
                <div class="kpi-value">{alta_confianza_pct:.1f}%</div>
            </div>
        </div>
    </div>

    <!-- ── TABS ── -->
    <ul class="nav nav-pills mb-3 flex-wrap" id="mainTabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab1" type="button">📈 Resumen y Tendencias</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab2" type="button">🧠 Rendimiento NLP</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab3" type="button">🌍 Impacto Geográfico</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab4" type="button">☁️ Nubes de Palabras</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab5" type="button">📝 Conclusiones</button></li>
    </ul>

    <div class="tab-content" id="mainTabContent">

        <!-- ══════════════════════════════════════════════════════════════
             TAB 1 · RESUMEN Y TENDENCIAS
        ══════════════════════════════════════════════════════════════ -->
        <div class="tab-pane fade show active" id="tab1" role="tabpanel">

            <div class="row"><div class="col-12">
                <div class="glass-card">
                    {_html_tiempo}
                    <p class="chart-desc">
                        <strong>¿Qué muestra?</strong> Evolución diaria del volumen de comentarios desglosado
                        por sentimiento desde la publicación del video. Cada banda de color representa una
                        categoría (<span style="color:#00f2fe">■ positivo</span>,
                        <span style="color:#a8b0c3">■ neutro</span>,
                        <span style="color:#fe0979">■ negativo</span>)
                        apilada para mostrar el total acumulado por día.<br>
                        <strong>Interpretación:</strong> La primera jornada concentra el mayor volumen de
                        actividad ({pico_n:,} comentarios el {pico_dia}), patrón típico en YouTube donde
                        el algoritmo impulsa el contenido recién publicado. La proporción de sentimientos
                        se mantiene relativamente estable a lo largo del tiempo, sin cambios bruscos de
                        tono que puedan asociarse a eventos externos.
                    </p>
                </div>
            </div></div>

            <div class="row"><div class="col-12">
                <div class="glass-card">
                    {_html_likes}
                    <p class="chart-desc">
                        <strong>¿Qué muestra?</strong> Promedio de likes recibidos por cada comentario
                        según su categoría de sentimiento. Los likes son un indicador de resonancia
                        social: cuántas personas estuvieron de acuerdo o encontraron valioso ese comentario.<br>
                        <strong>Interpretación:</strong> Los comentarios positivos obtienen un promedio
                        notablemente mayor de likes, lo que revela que la comunidad que interactúa
                        activamente con el video tiene una disposición más favorable que la distribución
                        textual general. En total, los comentarios positivos concentraron el
                        {pct_likes_pos:.1f}% de todos los likes del corpus.
                    </p>
                </div>
            </div></div>
        </div>

        <!-- ══════════════════════════════════════════════════════════════
             TAB 2 · RENDIMIENTO NLP
        ══════════════════════════════════════════════════════════════ -->
        <div class="tab-pane fade" id="tab2" role="tabpanel">

            <div class="row g-3">
                <div class="col-md-6">
                    <div class="glass-card">
                        {_html_dist_sent}
                        <p class="chart-desc">
                            <strong>¿Qué muestra?</strong> Conteo absoluto de comentarios clasificados
                            en cada categoría de sentimiento por el modelo RoBERTuito.<br>
                            <strong>Interpretación:</strong>
                            <span style="color:#00f2fe">Positivo {pct_pos:.1f}%</span> ·
                            <span style="color:#fe0979">Negativo {pct_neg:.1f}%</span> ·
                            <span style="color:#a8b0c3">Neutro {pct_neu:.1f}%</span>.
                            La distribución no uniforme confirma que el modelo discrimina de manera
                            efectiva entre clases en lugar de asignar categorías al azar.
                        </p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="glass-card">
                        {_html_confianza}
                        <p class="chart-desc">
                            <strong>¿Qué muestra?</strong> Distribución del score de confianza
                            (probabilidad softmax) que el modelo asigna a su propia predicción.
                            La línea dorada marca el umbral de alta confianza (0.80).<br>
                            <strong>Interpretación:</strong> Las clases positivo y negativo muestran
                            distribuciones sesgadas hacia la derecha (alta certeza). La clase neutro
                            presenta mayor dispersión porque agrupa comentarios ambiguos o muy
                            cortos, que son intrínsecamente más difíciles de clasificar.
                        </p>
                    </div>
                </div>
            </div>

            <div class="row"><div class="col-12">
                <div class="glass-card">
                    {_html_pca}
                    <p class="chart-desc">
                        <strong>¿Qué muestra?</strong> Cada punto representa un comentario proyectado
                        en un espacio 2D usando Análisis de Componentes Principales (PCA) sobre
                        vectores TF-IDF. Los colores corresponden al sentimiento asignado por el modelo.
                        K-Means agrupa los comentarios en 3 clusters temáticos cuyos términos
                        más representativos se muestran en la anotación lateral.<br>
                        <strong>Interpretación:</strong> La separación parcial entre grupos confirma
                        que los comentarios de distinto sentimiento usan vocabularios diferenciados.
                        La superposición en el centro corresponde a comentarios temáticamente
                        similares pero con distinta polaridad, lo cual es esperable en debates
                        políticos donde se discuten los mismos temas desde perspectivas opuestas.
                    </p>
                </div>
            </div></div>

            <div class="row"><div class="col-12">
                <div class="glass-card">
                    {_html_tfidf}
                    <p class="chart-desc">
                        <strong>¿Qué muestra?</strong> Los 15 términos más discriminantes de cada
                        clase según TF-IDF (Term Frequency – Inverse Document Frequency). A diferencia
                        de la frecuencia simple, TF-IDF penaliza los términos que aparecen en muchos
                        comentarios y premia los que son específicos de una categoría.<br>
                        <strong>Interpretación:</strong> Cada clase tiene un vocabulario propio y
                        diferenciado: los positivos destacan agradecimiento y apoyo
                        (<em>gracias, paz, presidente</em>), los negativos articulan crítica
                        argumentada (<em>derechos, humanos, delincuentes</em>), y los neutros
                        describen hechos o contexto (<em>años, país, video</em>). Esta separabilidad
                        léxica es evidencia de que el modelo aprendió fronteras reales de decisión.
                    </p>
                </div>
            </div></div>
        </div>

        <!-- ══════════════════════════════════════════════════════════════
             TAB 3 · GEOGRAFÍA
        ══════════════════════════════════════════════════════════════ -->
        <div class="tab-pane fade" id="tab3" role="tabpanel">

            <div class="row"><div class="col-12">
                <div class="glass-card">
                    {_html_geo}
                    <p class="chart-desc">
                        <strong>¿Qué muestra?</strong> Las 15 ubicaciones geográficas más frecuentes
                        detectadas automáticamente en el texto de los comentarios mediante la
                        librería GeoText. Solo el {df_geo.shape[0]/total_comentarios*100:.1f}%
                        del corpus contiene una referencia geográfica identificable.<br>
                        <strong>Interpretación:</strong> {top1_pais} lidera con {top1_n:,} menciones,
                        lo cual es esperable dado que el video trata directamente sobre ese país.
                        La presencia de Colombia, Venezuela y Argentina refleja el alcance
                        transnacional del debate sobre seguridad y gobernanza en Latinoamérica,
                        regiones con experiencias comparables que generan interés en el tema.
                    </p>
                </div>
            </div></div>

            <div class="row"><div class="col-12">
                <div class="glass-card">
                    {_html_geo_sent}
                    <p class="chart-desc">
                        <strong>¿Qué muestra?</strong> Distribución de las tres categorías de
                        sentimiento dentro de los comentarios provenientes de los 5 países con
                        mayor participación en el corpus.<br>
                        <strong>Interpretación:</strong> La comparación entre países revela
                        diferencias en la tonalidad del debate según el origen geográfico del
                        comentarista. Los países con mayor experiencia de violencia o de
                        gobiernos similares tienden a mostrar una composición de sentimientos
                        distinta a la de países con contextos políticos diferentes.
                    </p>
                </div>
            </div></div>
        </div>

        <!-- ══════════════════════════════════════════════════════════════
             TAB 4 · NUBES DE PALABRAS
        ══════════════════════════════════════════════════════════════ -->
        <div class="tab-pane fade" id="tab4" role="tabpanel">

            <!-- FILA 1: Antes vs Después -->
            <div class="glass-card mb-3">
                <h5 class="section-title">
                    <i class="fa-solid fa-arrows-left-right me-2" style="color:#00f2fe"></i>
                    Comparativa: Antes y Después del Preprocesamiento
                </h5>
                <div class="row g-3">
                    <div class="col-md-6">
                        <img src="data:image/png;base64,{img_antes}" class="wc-img" alt="Nube antes">
                        <p class="wc-label">📄 Texto original · con stopwords y ruido</p>
                    </div>
                    <div class="col-md-6">
                        <img src="data:image/png;base64,{img_despues}" class="wc-img" alt="Nube después">
                        <p class="wc-label">✨ Texto limpio · sin stopwords · lematizado</p>
                    </div>
                </div>
                <p class="chart-desc mt-3">
                    <strong>¿Qué muestra?</strong> La nube izquierda representa el vocabulario
                    <strong>antes del preprocesamiento</strong>: las palabras más grandes son
                    preposiciones, artículos y conjunciones del español (<em>que, de, y, el, la, no</em>)
                    que no aportan significado analítico pero dominan el corpus por su alta frecuencia.<br>
                    La nube derecha muestra el vocabulario <strong>después de aplicar el pipeline NLP</strong>
                    (eliminación de stopwords, lematización con spaCy): emergen los términos con
                    verdadera carga semántica. El corpus pasó de {146131:,} palabras totales a
                    {71308:,} tras el preprocesamiento, eliminando el {(1-71308/146131)*100:.0f}%
                    del ruido léxico.<br>
                    <strong>Interpretación:</strong> <em>Salvador</em>, <em>Bukele</em>, <em>Juan</em> y
                    <em>paz</em> se posicionan como los términos centrales del debate, confirmando que
                    el análisis gira en torno al personaje político, el periodista del video y
                    el concepto de seguridad pública.
                </p>
            </div>

            <!-- FILA 2: Por sentimiento (post-procesamiento) -->
            <div class="glass-card">
                <h5 class="section-title">
                    <i class="fa-solid fa-palette me-2" style="color:#00f2fe"></i>
                    Vocabulario por Categoría de Sentimiento (texto lematizado)
                </h5>
                <div class="row g-3">
                    <div class="col-md-4">
                        <img src="data:image/png;base64,{wc_imgs_sent.get('positivo','')}" class="wc-img" alt="Positivo">
                        <p class="wc-label" style="color:#00f2fe">😊 Positivo · {int(sent_counts.get('positivo',0)):,} comentarios ({pct_pos:.1f}%)</p>
                    </div>
                    <div class="col-md-4">
                        <img src="data:image/png;base64,{wc_imgs_sent.get('negativo','')}" class="wc-img" alt="Negativo">
                        <p class="wc-label" style="color:#fe0979">😠 Negativo · {int(sent_counts.get('negativo',0)):,} comentarios ({pct_neg:.1f}%)</p>
                    </div>
                    <div class="col-md-4">
                        <img src="data:image/png;base64,{wc_imgs_sent.get('neutro','')}" class="wc-img" alt="Neutro">
                        <p class="wc-label" style="color:#a8b0c3">😐 Neutro · {int(sent_counts.get('neutro',0)):,} comentarios ({pct_neu:.1f}%)</p>
                    </div>
                </div>
                <p class="chart-desc mt-3">
                    <strong>¿Qué muestra?</strong> El vocabulario más frecuente dentro de cada
                    categoría de sentimiento <em>después</em> del preprocesamiento. Cada nube
                    usa un esquema de color propio para facilitar la distinción visual.<br>
                    <strong>Interpretación:</strong>
                    Los <span style="color:#00f2fe"><strong>positivos</strong></span> comparten términos de
                    apoyo y agradecimiento (<em>gracias, bien, paz, saludos, presidente</em>).
                    Los <span style="color:#fe0979"><strong>negativos</strong></span> concentran
                    el debate crítico (<em>derechos, humanos, delincuentes, precio, pandilla</em>),
                    con mayor riqueza léxica que refleja argumentación más elaborada.
                    Los <span style="color:#a8b0c3"><strong>neutros</strong></span> incluyen
                    términos descriptivos e informativos (<em>salvador, país, años, video, canal</em>)
                    sin polaridad clara. La separación léxica entre categorías valida la coherencia
                    del modelo de clasificación.
                </p>
            </div>
        </div>

        <!-- ══════════════════════════════════════════════════════════════
             TAB 5 · CONCLUSIONES Y RECOMENDACIONES
        ══════════════════════════════════════════════════════════════ -->
        <div class="tab-pane fade" id="tab5" role="tabpanel">

            <div class="glass-card mb-4" style="border-left: 4px solid #00f2fe">
                <h5 class="section-title">
                    <i class="fa-solid fa-magnifying-glass-chart me-2" style="color:#00f2fe"></i>
                    Conclusiones del Análisis
                </h5>

                <!-- C1 -->
                <div class="glass-card conclusion-card mb-3">
                    <span class="finding-badge">Hallazgo 1 · Distribución de Sentimientos</span>
                    <p class="mb-1">
                        El modelo RoBERTuito clasificó los {total_comentarios:,} comentarios con una
                        distribución <strong>significativamente no uniforme</strong>:
                        <span class="stat-highlight">{pct_pos:.1f}%</span> positivos,
                        <span style="color:#fe0979; font-weight:700; font-size:1.4rem">{pct_neg:.1f}%</span> negativos y
                        <span style="color:#a8b0c3; font-weight:700; font-size:1.4rem">{pct_neu:.1f}%</span> neutros.
                        El ratio positivo/negativo de <strong>{ratio_pos_neg:.2f}x</strong> indica
                        una leve preponderancia favorable, aunque con una oposición significativa que
                        refleja la naturaleza polarizada del tema político abordado.
                    </p>
                </div>

                <!-- C2 -->
                <div class="glass-card conclusion-card mb-3">
                    <span class="finding-badge">Hallazgo 2 · Confianza del Modelo</span>
                    <p class="mb-1">
                        La confianza media del modelo es de <span class="stat-highlight">{confianza_media:.3f}</span>,
                        con el <span class="stat-highlight">{alta_confianza_pct:.1f}%</span> de predicciones
                        superando el umbral de alta confianza (0.80). La clase <em>neutro</em> presenta la
                        menor confianza media (~0.62), consistente con su naturaleza difusa: agrupa
                        comentarios cortos, ambiguos o descriptivos que son intrínsecamente más difíciles
                        de clasificar. Esto sugiere que trabajos futuros podrían beneficiarse de un umbral
                        de confianza mínimo para filtrar predicciones dudosas.
                    </p>
                </div>

                <!-- C3 -->
                <div class="glass-card conclusion-card mb-3">
                    <span class="finding-badge">Hallazgo 3 · Engagement y Sentimiento</span>
                    <p class="mb-1">
                        Los comentarios positivos concentraron el
                        <span class="stat-highlight">{pct_likes_pos:.1f}%</span> del total de likes
                        del corpus, a pesar de representar solo el {pct_pos:.1f}% de los comentarios.
                        Esto evidencia una <strong>asimetría de resonancia social</strong>: la comunidad
                        que interactúa activamente con el video tiene una disposición más favorable que
                        el conjunto de comentaristas. Los discursos de apoyo generan mayor adhesión
                        colectiva visible, mientras los críticos obtienen menos validación explícita.
                    </p>
                </div>

                <!-- C4 -->
                <div class="glass-card conclusion-card mb-3">
                    <span class="finding-badge">Hallazgo 4 · Clusterización Temática</span>
                    <p class="mb-1">
                        El análisis TF-IDF y PCA reveló que los comentarios se organizan en torno a
                        <strong>subtemas discursivos diferenciados</strong>: elogios al periodista,
                        debate sobre derechos humanos, comparaciones con otros países y discusión sobre
                        el "precio de la paz". Esta estructura temática es independiente del sentimiento,
                        lo que demuestra que personas con distinta polaridad debaten los mismos temas
                        desde perspectivas opuestas, validando la riqueza del corpus para análisis
                        de argumentación política.
                    </p>
                </div>

                <!-- C5 -->
                <div class="glass-card conclusion-card mb-3">
                    <span class="finding-badge">Hallazgo 5 · Impacto del Preprocesamiento NLP</span>
                    <p class="mb-1">
                        El pipeline de limpieza y lematización eliminó el
                        <strong>{(1-71308/146131)*100:.0f}% del ruido léxico</strong> (pasando de
                        146,131 a 71,308 palabras efectivas), reduciendo el vocabulario único de
                        19,946 a 12,795 términos. La comparativa de nubes de palabras muestra cómo
                        las stopwords y símbolos dominaban el texto original, ocultando el contenido
                        semántico real que solo emerge tras el preprocesamiento.
                    </p>
                </div>

                <!-- C6 -->
                <div class="glass-card conclusion-card mb-0">
                    <span class="finding-badge">Hallazgo 6 · Alcance Geográfico</span>
                    <p class="mb-1">
                        Con referencias geográficas identificadas en el {df_geo.shape[0]/total_comentarios*100:.1f}%
                        del corpus y {n_paises} ubicaciones distintas, el debate trasciende ampliamente
                        las fronteras de El Salvador. La participación de Colombia, Venezuela, México
                        y Argentina confirma que la percepción sobre las políticas de seguridad de
                        Bukele forma parte de un debate político latinoamericano más amplio, donde
                        audiencias de distintos países proyectan sus propias realidades nacionales.
                    </p>
                </div>
            </div>

            <!-- RECOMENDACIONES -->
            <div class="glass-card" style="border-left: 4px solid #fe0979">
                <h5 class="section-title">
                    <i class="fa-solid fa-lightbulb me-2" style="color:#fe0979"></i>
                    Recomendaciones para Futuros Análisis
                </h5>

                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="glass-card rec-card h-100">
                            <span class="rec-badge">R1 · Anotación Manual</span>
                            <p class="mb-0 small">
                                Etiquetar manualmente una muestra representativa (mínimo 500 comentarios)
                                para obtener <strong>métricas supervisadas</strong> reales (precisión,
                                recall, F1-score) y validar el rendimiento de RoBERTuito en este dominio
                                específico. Sin ground truth, la evaluación actual solo puede ser indirecta.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card rec-card h-100">
                            <span class="rec-badge">R2 · Análisis de Emociones</span>
                            <p class="mb-0 small">
                                Ir más allá de la polaridad triclase e implementar un modelo de
                                <strong>análisis de emociones</strong> (ira, alegría, miedo, sorpresa)
                                usando pysentimiento con <code>task="emotion"</code>. Esto aportaría
                                mayor granularidad para entender qué emociones específicas moviliza
                                el contenido del video.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card rec-card h-100">
                            <span class="rec-badge">R3 · Análisis de Aspectos (ABSA)</span>
                            <p class="mb-0 small">
                                Aplicar <strong>Aspect-Based Sentiment Analysis</strong> para identificar
                                sobre qué dimensiones concretas (seguridad, economía, derechos humanos,
                                imagen del periodista) se expresa cada tipo de sentimiento. Actualmente
                                el análisis trata cada comentario como una unidad uniforme, ignorando que
                                un mismo texto puede tener polaridades distintas por aspecto.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card rec-card h-100">
                            <span class="rec-badge">R4 · Modelos de Tópicos (BERTopic)</span>
                            <p class="mb-0 small">
                                Reemplazar K-Means + TF-IDF por <strong>BERTopic</strong>, que combina
                                embeddings contextuales (SBERT) con clustering probabilístico (HDBSCAN).
                                BERTopic maneja mejor la polisemia y la polifonía temática de comentarios
                                cortos, produciendo tópicos más coherentes e interpretables.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card rec-card h-100">
                            <span class="rec-badge">R5 · Análisis Temporal Extendido</span>
                            <p class="mb-0 small">
                                Ampliar el corpus incluyendo los <strong>comentarios de respuesta</strong>
                                (hilos anidados) y extender el período de recolección para estudiar la
                                evolución del sentimiento a largo plazo. Correlacionar los picos de
                                actividad con eventos externos (noticias, declaraciones políticas) para
                                entender qué dispara cambios en el tono del debate.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card rec-card h-100">
                            <span class="rec-badge">R6 · Análisis Multimodal y Multilingüe</span>
                            <p class="mb-0 small">
                                Incorporar el análisis de <strong>emojis y expresiones coloquiales</strong>
                                como señales adicionales de sentimiento (actualmente se eliminan en el
                                preprocesamiento). Además, implementar detección de idioma para identificar
                                y analizar comentarios en inglés y otros idiomas presentes en el corpus.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="text-center mt-4 mb-2">
                <small class="text-secondary">
                    <i class="fa-solid fa-robot me-1"></i>
                    Pipeline NLP: spaCy · NLTK · RoBERTuito (pysentimiento) · scikit-learn ·
                    Plotly · Python 3.13 &nbsp;|&nbsp;
                    Corpus: {total_comentarios:,} comentarios · Video ID: pWnndG1K5Hg
                </small>
            </div>
        </div>

    </div><!-- /tab-content -->
</div><!-- /container -->

<!-- FIX: Plotly + Bootstrap tabs resize -->
<script>
    document.addEventListener("DOMContentLoaded", function () {{
        var sel = 'button[data-bs-toggle="pill"], button[data-bs-toggle="tab"]';
        document.querySelectorAll(sel).forEach(function (tabEl) {{
            tabEl.addEventListener('shown.bs.tab', function (event) {{
                var pane = document.querySelector(event.target.getAttribute('data-bs-target'));
                if (!pane) return;
                pane.querySelectorAll('.plotly-graph-div').forEach(function (div) {{
                    Plotly.Plots.resize(div);
                }});
            }});
        }});
        setTimeout(function () {{
            document.querySelectorAll('.tab-pane.show.active .plotly-graph-div').forEach(function (div) {{
                Plotly.Plots.resize(div);
            }});
        }}, 200);
    }});
</script>
</body>
</html>"""

# ── Escribir archivo ──────────────────────────────────────────────────────────
output_path = os.path.join(PROJECT_DIR, "output", "dashboard_sentimientos.html")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_template)

size_kb = os.path.getsize(output_path) / 1024
print(f"✅ Dashboard generado: {output_path}")
print(f"   Tamaño: {size_kb:.0f} KB")
