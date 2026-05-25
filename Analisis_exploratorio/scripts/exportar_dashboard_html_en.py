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

# ── Constants ─────────────────────────────────────────────────────────────────
PALETA      = {"positivo": "#00f2fe", "neutro": "#a8b0c3", "negativo": "#fe0979"}
ORDEN_SENT  = ["positivo", "neutro", "negativo"]
LABEL_SENT  = {"positivo": "Positive", "neutro": "Neutral", "negativo": "Negative"}
PALETA_EN   = {"Positive": "#00f2fe", "Neutral": "#a8b0c3", "Negative": "#fe0979"}
ORDEN_EN    = ["Positive", "Neutral", "Negative"]
BG          = "#0b0f19"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
DB_PATH     = os.path.join(BASE_DIR, "data", "base_limpia.db")

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM comentarios", conn)
    conn.close()
    df['texto']        = df['texto'].fillna('')
    df['texto_limpio'] = df['texto_limpio'].fillna('')
    df['ubicacion']    = df['ubicacion'].fillna('sin info')
    df['sentimiento']  = df['sentimiento'].fillna('neutro')
    df['probabilidad'] = pd.to_numeric(df['probabilidad'], errors='coerce').fillna(0)
    df['likes']        = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
    df['fecha_dt']     = pd.to_datetime(df['fecha'], errors='coerce')
    df['solo_fecha']   = df['fecha_dt'].dt.date
    df['sentiment_en'] = df['sentimiento'].map(LABEL_SENT)
    return df

df = load_data()

# ── Global KPIs ───────────────────────────────────────────────────────────────
total_comments     = len(df)
avg_confidence     = df['probabilidad'].mean()
high_conf_pct      = (df['probabilidad'] >= 0.8).mean() * 100
sent_counts        = df['sentimiento'].value_counts()
dominant_sentiment = sent_counts.idxmax()
dominant_label     = LABEL_SENT[dominant_sentiment]
pct_pos            = sent_counts.get('positivo', 0) / total_comments * 100
pct_neg            = sent_counts.get('negativo', 0) / total_comments * 100
pct_neu            = sent_counts.get('neutro',   0) / total_comments * 100
ratio_pos_neg      = sent_counts.get('positivo', 0) / max(sent_counts.get('negativo', 1), 1)
likes_total_pos    = df[df['sentimiento'] == 'positivo']['likes'].sum()
likes_total        = df['likes'].sum()
pct_likes_pos      = likes_total_pos / max(likes_total, 1) * 100

_layout_base = dict(template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    autosize=True)
_cfg = {"responsive": True}

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW & TRENDS
# ═════════════════════════════════════════════════════════════════════════════
df_tiempo    = df.dropna(subset=['solo_fecha']).copy()
pivot_tiempo = (df_tiempo.groupby(['solo_fecha', 'sentimiento'])
                .size().unstack(fill_value=0)
                .reindex(columns=ORDEN_SENT, fill_value=0))

fig_tiempo = go.Figure()
for sent in ORDEN_SENT:
    fig_tiempo.add_trace(go.Scatter(
        x=pivot_tiempo.index, y=pivot_tiempo[sent],
        mode='lines', stackgroup='one',
        name=LABEL_SENT[sent], line=dict(color=PALETA[sent], width=0.5)
    ))
fig_tiempo.update_layout(title="Sentiment Evolution Over Time",
                         xaxis_title="Date", yaxis_title="Comments",
                         **_layout_base)

likes_prom = (df.groupby('sentimiento')['likes'].mean()
              .reindex(ORDEN_SENT).reset_index())
likes_prom.columns = ['sentimiento', 'avg_likes']
likes_prom['label'] = likes_prom['sentimiento'].map(LABEL_SENT)
fig_likes = px.bar(likes_prom, x='label', y='avg_likes',
                   color='label', color_discrete_map=PALETA_EN,
                   title="Average Likes per Sentiment Category",
                   text='avg_likes')
fig_likes.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_likes.update_layout(xaxis_title="Sentiment", yaxis_title="Average Likes",
                        showlegend=False, **_layout_base)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — NLP PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

# Metric 1: Distribution
counts_y = [int(sent_counts.get(s, 0)) for s in ORDEN_SENT]
fig_dist = px.bar(x=ORDEN_EN, y=counts_y,
                  color=ORDEN_EN, color_discrete_map=PALETA_EN,
                  title="Metric 1: Sentiment Distribution", text=counts_y)
fig_dist.update_traces(textposition='outside')
fig_dist.update_layout(showlegend=False, xaxis_title="Sentiment",
                       yaxis_title="Number of Comments", **_layout_base)

# Metric 2: Confidence
fig_conf = px.histogram(df, x='probabilidad', color='sentiment_en',
                        nbins=40, barmode='overlay',
                        color_discrete_map=PALETA_EN, opacity=0.75,
                        title="Metric 2: Model Confidence Score by Class",
                        labels={'probabilidad': 'Confidence Score',
                                'sentiment_en': 'Sentiment'})
fig_conf.add_vline(x=0.8, line_dash="dash", line_color="#ffd700",
                   annotation_text="Threshold 0.8", annotation_position="top right")
fig_conf.update_layout(xaxis_title="Prediction Probability",
                       yaxis_title="Number of Comments", **_layout_base)

# Metric 3: PCA Clustering
corpus_valido = df[df['texto_limpio'].str.strip().str.len() > 3].copy()
vectorizer    = TfidfVectorizer(max_features=500, min_df=3, max_df=0.85,
                                ngram_range=(1, 2), sublinear_tf=True)
X      = vectorizer.fit_transform(corpus_valido['texto_limpio'].astype(str))
km     = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())

df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'])
df_pca['sentiment'] = corpus_valido['sentimiento'].map(LABEL_SENT).values
df_pca['excerpt']   = corpus_valido['texto'].str[:60] + "..."

fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='sentiment',
                     hover_data=['excerpt'], color_discrete_map=PALETA_EN,
                     title="Metric 3: 2D PCA Projection of Thematic Clusters",
                     opacity=0.65,
                     labels={'sentiment': 'Sentiment', 'excerpt': 'Excerpt'})
feature_names = vectorizer.get_feature_names_out()
centros       = km.cluster_centers_
cluster_words = []
for k in range(3):
    top_idx = centros[k].argsort()[-5:][::-1]
    cluster_words.append(f"C{k}: " + ", ".join(feature_names[top_idx]))
fig_pca.add_annotation(
    x=1.05, y=0.5,
    text="Key Words by Cluster:<br>" + "<br>".join(cluster_words),
    xref="paper", yref="paper", showarrow=False,
    font=dict(color="#a8b0c3", size=11), align="left",
    bgcolor="rgba(0,0,0,0.5)")
fig_pca.update_layout(**_layout_base)

# Metric 4: TF-IDF per sentiment
fig_tfidf = make_subplots(rows=1, cols=3,
                          subplot_titles=["Positive", "Neutral", "Negative"])
for i, sent in enumerate(ORDEN_SENT):
    textos_clase = (df[df['sentimiento'] == sent]['texto_limpio']
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
            orientation='h', name=LABEL_SENT[sent],
            marker=dict(color=PALETA[sent])
        ), row=1, col=i + 1)
fig_tfidf.update_layout(
    title="Metric 4: Most Discriminant Terms per Sentiment Class (TF-IDF)",
    showlegend=False, height=520, **_layout_base)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC IMPACT
# ═════════════════════════════════════════════════════════════════════════════
df_geo     = df[df['ubicacion'].str.lower() != 'sin info'].copy()
geo_counts = df_geo['ubicacion'].value_counts().head(15).reset_index()
geo_counts.columns = ['Location', 'Count']
fig_geo = px.bar(geo_counts, x='Count', y='Location', orientation='h',
                 title="Top 15 Commenter Locations",
                 color='Count', color_continuous_scale="Agsunset", text='Count')
fig_geo.update_traces(textposition='outside')
fig_geo.update_layout(yaxis={'categoryorder': 'total ascending'},
                      xaxis_title="Number of Comments", yaxis_title="",
                      **_layout_base)

top5_locs    = geo_counts.head(5)['Location'].tolist()
df_pais_sent = (df_geo[df_geo['ubicacion'].isin(top5_locs)]
                .groupby(['ubicacion', 'sentimiento'])
                .size().unstack(fill_value=0)
                .reindex(columns=ORDEN_SENT, fill_value=0)
                .reset_index())
fig_geo_sent = go.Figure()
for sent in ORDEN_SENT:
    if sent in df_pais_sent.columns:
        fig_geo_sent.add_trace(go.Bar(
            name=LABEL_SENT[sent],
            x=df_pais_sent['ubicacion'],
            y=df_pais_sent[sent],
            marker_color=PALETA[sent]))
fig_geo_sent.update_layout(barmode='group',
                            title="Sentiment Distribution by Country (Top 5)",
                            xaxis_title="Country / Location",
                            yaxis_title="Number of Comments",
                            **_layout_base)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — WORD CLOUDS
# ═════════════════════════════════════════════════════════════════════════════
print("Generating word clouds...")

def wc_to_base64(wc_obj, bg=BG, figsize=(10, 5)):
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
    return WordCloud(background_color=bg_color, colormap=colormap,
                     max_words=max_words, width=width, height=height,
                     collocations=False, prefer_horizontal=0.85,
                     relative_scaling=0.5, min_font_size=10).generate(text)

img_antes   = wc_to_base64(make_wc(" ".join(df['texto'].str.lower().fillna('')),
                                    colormap='Greys_r'))
img_despues = wc_to_base64(make_wc(" ".join(df['texto_limpio'].fillna('')),
                                    colormap='winter'))
wc_imgs_sent = {}
for sent, cmap in [("positivo","cool"), ("negativo","RdPu"), ("neutro","Greys")]:
    txt = " ".join(df[df['sentimiento'] == sent]['texto_limpio'].fillna(''))
    if len(txt.strip()) > 50:
        wc_imgs_sent[sent] = wc_to_base64(make_wc(txt, colormap=cmap))

print("  Word clouds done.")

# ═════════════════════════════════════════════════════════════════════════════
# PRE-RENDER Plotly → HTML snippets
# ═════════════════════════════════════════════════════════════════════════════
_html_tiempo   = fig_tiempo.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_likes    = fig_likes.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_dist     = fig_dist.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_conf     = fig_conf.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_pca      = fig_pca.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_tfidf    = fig_tfidf.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_geo      = fig_geo.to_html(full_html=False, include_plotlyjs=False, config=_cfg)
_html_geo_sent = fig_geo_sent.to_html(full_html=False, include_plotlyjs=False, config=_cfg)

# Stats for conclusions
pico_dia    = df_tiempo.groupby('solo_fecha').size().idxmax()
pico_n      = int(df_tiempo.groupby('solo_fecha').size().max())
n_locations = df_geo['ubicacion'].nunique()
top1_loc    = geo_counts.iloc[0]['Location']
top1_n      = int(geo_counts.iloc[0]['Count'])
geo_cov_pct = df_geo.shape[0] / total_comments * 100
noise_pct   = int((1 - 71308/146131) * 100)

# ═════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═════════════════════════════════════════════════════════════════════════════
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>YouTube Sentiment Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
    <style>
        body {{ background-color:{BG}; color:#e2e8f0; font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; padding:24px; }}
        .glass-card {{ background:rgba(21,27,43,.75); border-radius:16px; box-shadow:0 4px 30px rgba(0,0,0,.5);
                       backdrop-filter:blur(10px); border:1px solid rgba(255,255,255,.06);
                       padding:22px; margin-bottom:22px; width:100%; }}
        .kpi-value {{ font-size:2.4rem; font-weight:700; color:#fff; text-align:center; }}
        .kpi-title {{ font-size:.85rem; color:#a8b0c3; text-transform:uppercase; letter-spacing:1.2px;
                      text-align:center; margin-bottom:6px; }}
        .nav-pills .nav-link {{ color:#a8b0c3; margin-right:8px; border-radius:10px;
                                transition:all .25s; cursor:pointer; font-size:.92rem; }}
        .nav-pills .nav-link:hover {{ background:rgba(0,242,254,.18); color:#00f2fe; }}
        .nav-pills .nav-link.active {{ background-color:#00f2fe !important; color:#000 !important; font-weight:700; }}
        .tab-content {{ margin-top:22px; }}
        .chart-desc {{ font-size:.88rem; color:#a8b0c3; background:rgba(0,0,0,.25);
                       border-left:3px solid #00f2fe; border-radius:0 8px 8px 0;
                       padding:10px 14px; margin-top:10px; line-height:1.6; }}
        .chart-desc strong {{ color:#e2e8f0; }}
        .wc-img {{ width:100%; border-radius:12px; display:block; }}
        .wc-label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:1px;
                     color:#a8b0c3; text-align:center; margin-top:6px; }}
        .finding-badge {{ display:inline-block; background:rgba(0,242,254,.15); color:#00f2fe;
                          border-radius:20px; padding:2px 12px; font-size:.78rem;
                          font-weight:600; margin-bottom:8px; }}
        .rec-badge {{ display:inline-block; background:rgba(254,9,121,.15); color:#fe0979;
                      border-radius:20px; padding:2px 12px; font-size:.78rem;
                      font-weight:600; margin-bottom:8px; }}
        .stat-highlight {{ font-size:1.4rem; font-weight:700; color:#00f2fe; }}
        h5.section-title {{ font-weight:700; color:#fff; border-bottom:1px solid rgba(255,255,255,.1);
                            padding-bottom:8px; margin-bottom:16px; }}
        .plotly-graph-div {{ width:100% !important; }}
    </style>
</head>
<body>
<div class="container-fluid">

    <!-- HEADER -->
    <div class="d-flex align-items-center mb-1 mt-2">
        <i class="fa-solid fa-brain fa-2x me-3" style="color:#00f2fe"></i>
        <div>
            <h2 class="fw-bold text-white mb-0">YouTube Sentiment Analysis Dashboard</h2>
            <small class="text-secondary">Video: "Esto es lo que nadie te cuenta de El Salvador" &nbsp;·&nbsp; Corpus: {total_comments:,} comments</small>
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,.08);margin-bottom:24px">

    <!-- KPIs -->
    <div class="row mb-4 g-3">
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-comment-dots me-1"></i> Total Comments</div>
                <div class="kpi-value">{total_comments:,}</div>
            </div>
        </div>
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-face-smile me-1"></i> Dominant Sentiment</div>
                <div class="kpi-value" style="color:{PALETA[dominant_sentiment]}">{dominant_label}</div>
            </div>
        </div>
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-gauge-high me-1"></i> Avg. Model Confidence</div>
                <div class="kpi-value">{avg_confidence:.2f}</div>
            </div>
        </div>
        <div class="col-6 col-md-3">
            <div class="glass-card py-3">
                <div class="kpi-title"><i class="fa-solid fa-bullseye me-1"></i> High Confidence (&gt;0.8)</div>
                <div class="kpi-value">{high_conf_pct:.1f}%</div>
            </div>
        </div>
    </div>

    <!-- TABS -->
    <ul class="nav nav-pills mb-3 flex-wrap" id="mainTabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab1" type="button">📈 Overview &amp; Trends</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab2" type="button">🧠 NLP Performance</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab3" type="button">🌍 Geographic Impact</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab4" type="button">☁️ Word Clouds</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab5" type="button">📝 Conclusions</button></li>
    </ul>

    <div class="tab-content" id="mainTabContent">

        <!-- TAB 1 · OVERVIEW & TRENDS -->
        <div class="tab-pane fade show active" id="tab1" role="tabpanel">
            <div class="row"><div class="col-12"><div class="glass-card">
                {_html_tiempo}
                <p class="chart-desc">
                    <strong>What does it show?</strong> Daily comment volume broken down by sentiment since the video's publication.
                    Each colored band (<span style="color:#00f2fe">■ Positive</span>, <span style="color:#a8b0c3">■ Neutral</span>,
                    <span style="color:#fe0979">■ Negative</span>) is stacked to show the cumulative daily total.<br>
                    <strong>Interpretation:</strong> The largest spike occurred on {pico_dia} with {pico_n:,} comments —
                    a typical YouTube pattern driven by the recommendation algorithm at publication.
                    Sentiment proportions remain stable over time, with no abrupt tone shifts linked to external events.
                </p>
            </div></div></div>
            <div class="row"><div class="col-12"><div class="glass-card">
                {_html_likes}
                <p class="chart-desc">
                    <strong>What does it show?</strong> Average number of likes per comment by sentiment category.
                    Likes serve as a social resonance indicator — how many users agreed with or valued that comment.<br>
                    <strong>Interpretation:</strong> Positive comments receive far more likes on average, showing that
                    the actively engaging community leans more favorable than the overall commenter pool.
                    Positive comments alone captured {pct_likes_pos:.1f}% of all corpus likes.
                </p>
            </div></div></div>
        </div>

        <!-- TAB 2 · NLP PERFORMANCE -->
        <div class="tab-pane fade" id="tab2" role="tabpanel">
            <div class="row g-3">
                <div class="col-md-6"><div class="glass-card">
                    {_html_dist}
                    <p class="chart-desc">
                        <strong>What does it show?</strong> Absolute count of comments classified into each sentiment
                        category by the RoBERTuito model.<br>
                        <strong>Interpretation:</strong>
                        <span style="color:#00f2fe">Positive {pct_pos:.1f}%</span> ·
                        <span style="color:#fe0979">Negative {pct_neg:.1f}%</span> ·
                        <span style="color:#a8b0c3">Neutral {pct_neu:.1f}%</span>.
                        The non-uniform distribution confirms the model effectively discriminates
                        between classes rather than assigning categories randomly.
                    </p>
                </div></div>
                <div class="col-md-6"><div class="glass-card">
                    {_html_conf}
                    <p class="chart-desc">
                        <strong>What does it show?</strong> Distribution of the softmax confidence score the model
                        assigns to each prediction. The golden dashed line marks the high-confidence threshold (0.80).<br>
                        <strong>Interpretation:</strong> Positive and Negative classes skew right (high certainty).
                        Neutral shows greater spread because it groups ambiguous or very short comments that are
                        intrinsically harder to classify with confidence.
                    </p>
                </div></div>
            </div>
            <div class="row"><div class="col-12"><div class="glass-card">
                {_html_pca}
                <p class="chart-desc">
                    <strong>What does it show?</strong> Each dot is a comment projected into 2D via PCA on TF-IDF vectors.
                    Colors correspond to the model-assigned sentiment. K-Means finds 3 thematic clusters whose
                    top terms appear in the side annotation.<br>
                    <strong>Interpretation:</strong> Partial separation between sentiment groups confirms that different
                    polarities use distinct vocabularies. Overlap in the center reflects thematically similar comments
                    with opposing polarity — expected in political debates where the same topics are argued from
                    contrasting viewpoints.
                </p>
            </div></div></div>
            <div class="row"><div class="col-12"><div class="glass-card">
                {_html_tfidf}
                <p class="chart-desc">
                    <strong>What does it show?</strong> Top 15 most discriminant terms per sentiment class by TF-IDF
                    (Term Frequency – Inverse Document Frequency). TF-IDF penalizes common terms across all
                    documents and highlights those specific to each category.<br>
                    <strong>Interpretation:</strong> Each class has a distinct vocabulary: positives feature gratitude
                    and support (<em>gracias, paz, presidente</em>), negatives articulate criticism
                    (<em>derechos, humanos, delincuentes</em>), and neutrals describe context (<em>años, país, video</em>).
                    This lexical separability validates the model's genuine decision boundaries.
                </p>
            </div></div></div>
        </div>

        <!-- TAB 3 · GEOGRAPHIC IMPACT -->
        <div class="tab-pane fade" id="tab3" role="tabpanel">
            <div class="row"><div class="col-12"><div class="glass-card">
                {_html_geo}
                <p class="chart-desc">
                    <strong>What does it show?</strong> Top 15 geographic locations automatically detected in comment
                    text using the GeoText library. Only {geo_cov_pct:.1f}% of the corpus contains an
                    identifiable geographic reference.<br>
                    <strong>Interpretation:</strong> {top1_loc} leads with {top1_n:,} mentions, expected since the
                    video directly discusses that country. Colombia, Venezuela, and Argentina reflect the transnational
                    reach of the debate on Latin American security and governance.
                </p>
            </div></div></div>
            <div class="row"><div class="col-12"><div class="glass-card">
                {_html_geo_sent}
                <p class="chart-desc">
                    <strong>What does it show?</strong> Sentiment breakdown within comments from the top 5 most active
                    countries in the corpus.<br>
                    <strong>Interpretation:</strong> Comparing countries reveals that the tone of the debate varies
                    by geographic origin. Countries with similar experiences of political violence or government style
                    tend to display distinct sentiment compositions compared to those with different political contexts.
                </p>
            </div></div></div>
        </div>

        <!-- TAB 4 · WORD CLOUDS -->
        <div class="tab-pane fade" id="tab4" role="tabpanel">

            <div class="glass-card mb-3">
                <h5 class="section-title">
                    <i class="fa-solid fa-arrows-left-right me-2" style="color:#00f2fe"></i>
                    Before vs. After NLP Preprocessing
                </h5>
                <div class="row g-3">
                    <div class="col-md-6">
                        <img src="data:image/png;base64,{img_antes}" class="wc-img" alt="Before preprocessing">
                        <p class="wc-label">📄 Raw text &nbsp;·&nbsp; with stopwords &amp; noise</p>
                    </div>
                    <div class="col-md-6">
                        <img src="data:image/png;base64,{img_despues}" class="wc-img" alt="After preprocessing">
                        <p class="wc-label">✨ Cleaned text &nbsp;·&nbsp; stopwords removed &amp; lemmatized</p>
                    </div>
                </div>
                <p class="chart-desc mt-3">
                    <strong>What does it show?</strong> The left cloud represents vocabulary
                    <strong>before preprocessing</strong>: the largest words are Spanish stopwords
                    (<em>que, de, y, el, la, no</em>) that carry no analytical meaning but dominate
                    by raw frequency. The right cloud shows vocabulary
                    <strong>after the NLP pipeline</strong> (stopword removal, spaCy lemmatization):
                    semantically meaningful terms emerge. The corpus shrank from {146131:,} to
                    {71308:,} words — removing {noise_pct}% of lexical noise.<br>
                    <strong>Interpretation:</strong> <em>Salvador</em>, <em>Bukele</em>, <em>Juan</em>
                    and <em>paz</em> (peace) stand out as the central terms, confirming the debate revolves
                    around the political figure, the video journalist, and the concept of public security.
                </p>
            </div>

            <div class="glass-card">
                <h5 class="section-title">
                    <i class="fa-solid fa-palette me-2" style="color:#00f2fe"></i>
                    Vocabulary by Sentiment Category (lemmatized text)
                </h5>
                <div class="row g-3">
                    <div class="col-md-4">
                        <img src="data:image/png;base64,{wc_imgs_sent.get('positivo','')}" class="wc-img" alt="Positive">
                        <p class="wc-label" style="color:#00f2fe">😊 Positive &nbsp;·&nbsp; {int(sent_counts.get('positivo',0)):,} comments ({pct_pos:.1f}%)</p>
                    </div>
                    <div class="col-md-4">
                        <img src="data:image/png;base64,{wc_imgs_sent.get('negativo','')}" class="wc-img" alt="Negative">
                        <p class="wc-label" style="color:#fe0979">😠 Negative &nbsp;·&nbsp; {int(sent_counts.get('negativo',0)):,} comments ({pct_neg:.1f}%)</p>
                    </div>
                    <div class="col-md-4">
                        <img src="data:image/png;base64,{wc_imgs_sent.get('neutro','')}" class="wc-img" alt="Neutral">
                        <p class="wc-label" style="color:#a8b0c3">😐 Neutral &nbsp;·&nbsp; {int(sent_counts.get('neutro',0)):,} comments ({pct_neu:.1f}%)</p>
                    </div>
                </div>
                <p class="chart-desc mt-3">
                    <strong>What does it show?</strong> Most frequent vocabulary within each sentiment category after
                    preprocessing. Each cloud uses its own color scheme for easy visual distinction.<br>
                    <strong>Interpretation:</strong>
                    <span style="color:#00f2fe"><strong>Positive</strong></span> comments share supportive and grateful
                    language (<em>gracias, bien, paz, saludos, presidente</em>).
                    <span style="color:#fe0979"><strong>Negative</strong></span> comments concentrate critical discourse
                    (<em>derechos, humanos, delincuentes, precio, pandilla</em>), with richer vocabulary reflecting
                    more elaborate argumentation.
                    <span style="color:#a8b0c3"><strong>Neutral</strong></span> comments feature descriptive and
                    informational terms (<em>salvador, país, años, video, canal</em>) without clear polarity.
                    The lexical separation between categories validates the classification model's coherence.
                </p>
            </div>
        </div>

        <!-- TAB 5 · CONCLUSIONS & RECOMMENDATIONS -->
        <div class="tab-pane fade" id="tab5" role="tabpanel">

            <div class="glass-card mb-4" style="border-left:4px solid #00f2fe">
                <h5 class="section-title">
                    <i class="fa-solid fa-magnifying-glass-chart me-2" style="color:#00f2fe"></i>
                    Key Findings
                </h5>

                <div class="glass-card" style="border-left:4px solid #00f2fe;margin-bottom:14px">
                    <span class="finding-badge">Finding 1 · Sentiment Distribution</span>
                    <p class="mb-1">
                        RoBERTuito classified all {total_comments:,} comments with a
                        <strong>significantly non-uniform distribution</strong>:
                        <span class="stat-highlight">{pct_pos:.1f}%</span> positive,
                        <span style="color:#fe0979;font-weight:700;font-size:1.4rem">{pct_neg:.1f}%</span> negative, and
                        <span style="color:#a8b0c3;font-weight:700;font-size:1.4rem">{pct_neu:.1f}%</span> neutral.
                        The positive-to-negative ratio of <strong>{ratio_pos_neg:.2f}x</strong> indicates a slight
                        favorable preponderance, though with substantial opposition reflecting the polarized nature of
                        the political subject matter.
                    </p>
                </div>

                <div class="glass-card" style="border-left:4px solid #00f2fe;margin-bottom:14px">
                    <span class="finding-badge">Finding 2 · Model Confidence</span>
                    <p class="mb-1">
                        Average model confidence is <span class="stat-highlight">{avg_confidence:.3f}</span>,
                        with <span class="stat-highlight">{high_conf_pct:.1f}%</span> of predictions exceeding the
                        high-confidence threshold (0.80). The <em>Neutral</em> class shows the lowest average
                        confidence (~0.62), consistent with its diffuse nature: it groups short, ambiguous, or
                        purely descriptive comments. Future work could benefit from applying a minimum confidence
                        threshold to filter uncertain predictions before analysis.
                    </p>
                </div>

                <div class="glass-card" style="border-left:4px solid #00f2fe;margin-bottom:14px">
                    <span class="finding-badge">Finding 3 · Engagement Asymmetry</span>
                    <p class="mb-1">
                        Positive comments captured <span class="stat-highlight">{pct_likes_pos:.1f}%</span> of all
                        corpus likes despite representing only {pct_pos:.1f}% of comments. This
                        <strong>social resonance asymmetry</strong> reveals that the actively engaging community
                        leans more favorable than the overall commenter population. Supportive discourse generates
                        greater visible collective endorsement, while critical comments receive less explicit validation.
                    </p>
                </div>

                <div class="glass-card" style="border-left:4px solid #00f2fe;margin-bottom:14px">
                    <span class="finding-badge">Finding 4 · Thematic Clustering</span>
                    <p class="mb-1">
                        TF-IDF and PCA analysis revealed comments organizing around
                        <strong>distinct discursive subtopics</strong>: praise for the journalist, human rights debate,
                        cross-country comparisons, and discussion of the "price of peace." This thematic structure
                        is independent of sentiment, showing that people with opposing polarities argue the same
                        topics from different angles — validating the corpus's value for political argumentation analysis.
                    </p>
                </div>

                <div class="glass-card" style="border-left:4px solid #00f2fe;margin-bottom:14px">
                    <span class="finding-badge">Finding 5 · NLP Preprocessing Impact</span>
                    <p class="mb-1">
                        The cleaning and lemmatization pipeline removed <strong>{noise_pct}% of lexical noise</strong>
                        (from 146,131 down to 71,308 effective words), reducing unique vocabulary from 19,946 to
                        12,795 terms. The word cloud comparison visually demonstrates how stopwords and symbols
                        dominated the raw text, concealing the real semantic content that only emerges after
                        preprocessing.
                    </p>
                </div>

                <div class="glass-card" style="border-left:4px solid #00f2fe;margin-bottom:0">
                    <span class="finding-badge">Finding 6 · Geographic Reach</span>
                    <p class="mb-1">
                        With geographic references identified in {geo_cov_pct:.1f}% of the corpus and
                        {n_locations} distinct locations detected, the debate extends well beyond El Salvador.
                        Participation from Colombia, Venezuela, Mexico, and Argentina confirms that perceptions of
                        Bukele's security policies are part of a broader Latin American political conversation,
                        where audiences from different countries project their own national realities onto the discussion.
                    </p>
                </div>
            </div>

            <!-- RECOMMENDATIONS -->
            <div class="glass-card" style="border-left:4px solid #fe0979">
                <h5 class="section-title">
                    <i class="fa-solid fa-lightbulb me-2" style="color:#fe0979"></i>
                    Recommendations for Future Work
                </h5>
                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="glass-card" style="border-left:4px solid #fe0979;height:100%">
                            <span class="rec-badge">R1 · Manual Annotation</span>
                            <p class="mb-0 small">
                                Manually label a representative sample (minimum 500 comments) to obtain real
                                <strong>supervised metrics</strong> (precision, recall, F1-score) and validate
                                RoBERTuito's performance in this specific domain. Without ground truth, the
                                current evaluation can only be indirect.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card" style="border-left:4px solid #fe0979;height:100%">
                            <span class="rec-badge">R2 · Emotion Analysis</span>
                            <p class="mb-0 small">
                                Go beyond three-class polarity and implement an <strong>emotion detection</strong>
                                model (anger, joy, fear, surprise) using pysentimiento with
                                <code>task="emotion"</code>, providing greater granularity to understand
                                which specific emotions the video content triggers.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card" style="border-left:4px solid #fe0979;height:100%">
                            <span class="rec-badge">R3 · Aspect-Based Sentiment (ABSA)</span>
                            <p class="mb-0 small">
                                Apply <strong>Aspect-Based Sentiment Analysis</strong> to identify on which
                                specific dimensions (security, economy, human rights, journalist image) each
                                sentiment type is expressed. The current approach treats each comment as a
                                uniform unit, ignoring that one text may carry different polarities per aspect.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card" style="border-left:4px solid #fe0979;height:100%">
                            <span class="rec-badge">R4 · Topic Modeling (BERTopic)</span>
                            <p class="mb-0 small">
                                Replace K-Means + TF-IDF with <strong>BERTopic</strong>, which combines contextual
                                embeddings (SBERT) with probabilistic clustering (HDBSCAN). BERTopic handles
                                polysemy and thematic complexity in short comments far better, producing more
                                coherent and interpretable topics.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card" style="border-left:4px solid #fe0979;height:100%">
                            <span class="rec-badge">R5 · Extended Temporal Analysis</span>
                            <p class="mb-0 small">
                                Expand the corpus to include <strong>reply threads</strong> and extend the
                                collection window to study long-term sentiment trends. Correlate activity spikes
                                with external events (news, political statements) to understand what triggers
                                shifts in the debate's tone.
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card" style="border-left:4px solid #fe0979;height:100%">
                            <span class="rec-badge">R6 · Multimodal &amp; Multilingual</span>
                            <p class="mb-0 small">
                                Incorporate <strong>emoji and colloquial expression analysis</strong> as additional
                                sentiment signals (currently removed during preprocessing). Also implement language
                                detection to identify and analyze comments written in English and other languages
                                present in the corpus.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="text-center mt-4 mb-2">
                <small class="text-secondary">
                    <i class="fa-solid fa-robot me-1"></i>
                    NLP Pipeline: spaCy · NLTK · RoBERTuito (pysentimiento) · scikit-learn · Plotly · Python 3.13
                    &nbsp;|&nbsp; Corpus: {total_comments:,} comments · Video ID: pWnndG1K5Hg
                </small>
            </div>
        </div>

    </div><!-- /tab-content -->
</div><!-- /container -->

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

# ── Write output ──────────────────────────────────────────────────────────────
output_path = os.path.join(PROJECT_DIR, "output", "dashboard_sentiment_analysis_en.html")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(output_path) / 1024
print(f"✅ English dashboard generated: {output_path}")
print(f"   Size: {size_kb:.0f} KB")
