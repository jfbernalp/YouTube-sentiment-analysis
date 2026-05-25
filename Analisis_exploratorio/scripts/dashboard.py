import os
import sqlite3
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

PALETA = {"positivo": "#00f2fe", "neutro": "#a8b0c3", "negativo": "#fe0979"}
ORDEN_SENT = ["positivo", "neutro", "negativo"]
BG_COLOR = "#0b0f19"
CARD_BG = "#151b2b"
TEXT_COLOR = "#e2e8f0"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "base_limpia.db")

def load_and_prepare_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM comentarios", conn)
    conn.close()

    df['texto'] = df['texto'].fillna('')
    df['texto_limpio'] = df['texto_limpio'].fillna('')
    df['ubicacion'] = df['ubicacion'].fillna('sin info')
    df['sentimiento'] = df['sentimiento'].fillna('neutro')
    df['probabilidad'] = pd.to_numeric(df['probabilidad'], errors='coerce').fillna(0)
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype(int)
    
    df['fecha_dt'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['solo_fecha'] = df['fecha_dt'].dt.date
    
    return df

df = load_and_prepare_data()

# --- PRE-CÁLCULOS ANALÍTICOS ---

# 1. KPIs
total_comentarios = len(df)
confianza_media = df['probabilidad'].mean()
alta_confianza_pct = (df['probabilidad'] >= 0.8).mean() * 100
sent_counts = df['sentimiento'].value_counts()
sentimiento_dominante = sent_counts.idxmax()

# 📈 Resumen y Tendencias
df_tiempo = df.dropna(subset=['solo_fecha']).copy()
pivot_tiempo = df_tiempo.groupby(['solo_fecha', 'sentimiento']).size().unstack(fill_value=0).reindex(columns=ORDEN_SENT, fill_value=0)

fig_tiempo = go.Figure()
for sent in ORDEN_SENT:
    fig_tiempo.add_trace(go.Scatter(
        x=pivot_tiempo.index, y=pivot_tiempo[sent], mode='lines', stackgroup='one',
        name=sent.capitalize(), line=dict(color=PALETA[sent], width=0.5)
    ))
fig_tiempo.update_layout(title="Evolución Temporal del Sentimiento", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

likes_prom = df.groupby('sentimiento')['likes'].mean().reset_index()
fig_likes = px.bar(likes_prom, x='sentimiento', y='likes', color='sentimiento', color_discrete_map=PALETA, title="Promedio de Likes por Sentimiento")
fig_likes.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# 🧠 Rendimiento NLP (Las 4 Métricas)
# Métrica 1: Distribución
fig_dist_sent = px.bar(x=ORDEN_SENT, y=[sent_counts.get(s, 0) for s in ORDEN_SENT], color=ORDEN_SENT, color_discrete_map=PALETA, title="Métrica 1: Distribución de Sentimientos")
fig_dist_sent.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis_title="Sentimiento", yaxis_title="Comentarios")

# Métrica 2: Confianza
fig_confianza = px.histogram(df, x='probabilidad', color='sentimiento', nbins=40, barmode='overlay', color_discrete_map=PALETA, title="Métrica 2: Score de Confianza del Modelo")
fig_confianza.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# Métrica 3: Coherencia de Clusterización (KMeans + PCA)
corpus_valido = df[df['texto_limpio'].str.strip().str.len() > 3].copy()
vectorizer = TfidfVectorizer(max_features=500, min_df=3, max_df=0.85, ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(corpus_valido['texto_limpio'].astype(str))
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())

df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'])
df_pca['sentimiento'] = corpus_valido['sentimiento'].values
df_pca['texto_corto'] = corpus_valido['texto'].str[:60] + "..."

fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='sentimiento', hover_data=['texto_corto'], color_discrete_map=PALETA, title="Métrica 3: PCA 2D de Clusters Temáticos", opacity=0.7)
fig_pca.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

feature_names = vectorizer.get_feature_names_out()
centros = km.cluster_centers_
cluster_words = []
for k in range(3):
    top_idx = centros[k].argsort()[-5:][::-1]
    cluster_words.append(f"C{k}: " + ", ".join(feature_names[top_idx]))
titulo_clusters = "Palabras Clave por Cluster:<br>" + "<br>".join(cluster_words)
fig_pca.add_annotation(
    x=1.05, y=0.5, text=titulo_clusters, xref="paper", yref="paper",
    showarrow=False, font=dict(color="#a8b0c3", size=11), align="left", bgcolor="rgba(0,0,0,0.5)"
)

# Métrica 4: Análisis de Palabras Clave TF-IDF por Sentimiento
fig_tfidf = make_subplots(rows=1, cols=3, subplot_titles=["Positivo", "Neutro", "Negativo"])
for i, sent in enumerate(ORDEN_SENT):
    textos_clase = df[df['sentimiento'] == sent]['texto_limpio'].fillna("").astype(str)
    textos_clase = textos_clase[textos_clase.str.strip().str.len() > 3]
    if len(textos_clase) > 0:
        vect = TfidfVectorizer(max_features=200, min_df=2, max_df=0.90)
        X_clase = vect.fit_transform(textos_clase)
        scores = np.asarray(X_clase.sum(axis=0)).ravel()
        top_indices = scores.argsort()[-15:]
        top_words = np.array(vect.get_feature_names_out())[top_indices]
        top_scores = scores[top_indices]
        fig_tfidf.add_trace(go.Bar(
            x=top_scores, y=top_words, orientation='h', name=sent, marker=dict(color=PALETA[sent])
        ), row=1, col=i+1)

fig_tfidf.update_layout(
    title="Métrica 4: Palabras más discriminantes (TF-IDF) por Sentimiento",
    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False, height=500
)

# 🌍 Geografía
df_geo = df[df['ubicacion'].str.lower() != 'sin info'].copy()
geo_counts = df_geo['ubicacion'].value_counts().head(15).reset_index()
geo_counts.columns = ['Ubicacion', 'Cantidad']
fig_geo = px.bar(geo_counts, x='Cantidad', y='Ubicacion', orientation='h', title="Top 15 Ubicaciones", color='Cantidad', color_continuous_scale="Agsunset")
fig_geo.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'})


# --- APP DASH ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"])
app.title = "Analítica Profunda de Sentimientos"

custom_css = """
.glass-card {
    background: rgba(21, 27, 43, 0.7); border-radius: 16px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; margin-bottom: 20px;
}
.kpi-value { font-size: 2.5rem; font-weight: bold; margin-bottom: 0; color: #fff; text-align: center; }
.kpi-title { font-size: 1rem; color: #a8b0c3; text-transform: uppercase; letter-spacing: 1px; text-align: center; }
.nav-pills .nav-link.active { background-color: #00f2fe !important; color: #000 !important; font-weight: bold; }
.nav-pills .nav-link { color: #a8b0c3; margin-right: 10px; border-radius: 10px; transition: all 0.3s; }
.nav-pills .nav-link:hover { background-color: rgba(0, 242, 254, 0.2); }
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{custom_css}</style>
    </head>
    <body style="background-color: {BG_COLOR}; color: {TEXT_COLOR}; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

kpi_row = dbc.Row([
    dbc.Col(html.Div(className="glass-card", children=[html.Div("Total Comentarios", className="kpi-title"), html.Div(f"{total_comentarios:,}", className="kpi-value")]), md=3),
    dbc.Col(html.Div(className="glass-card", children=[html.Div("Sentimiento Dominante", className="kpi-title"), html.Div(sentimiento_dominante.capitalize(), className="kpi-value", style={"color": PALETA[sentimiento_dominante]})]), md=3),
    dbc.Col(html.Div(className="glass-card", children=[html.Div("Confianza Media", className="kpi-title"), html.Div(f"{confianza_media:.2f}", className="kpi-value")]), md=3),
    dbc.Col(html.Div(className="glass-card", children=[html.Div("% Alta Confianza (>0.8)", className="kpi-title"), html.Div(f"{alta_confianza_pct:.1f}%", className="kpi-value")]), md=3),
], className="mb-4")

tabs = dbc.Tabs(
    [
        dbc.Tab(label="📈 Resumen y Tendencias", tab_id="tab-1", children=[
            dbc.Row([dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_tiempo, config={'displayModeBar': False})]), md=12)]),
            dbc.Row([dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_likes, config={'displayModeBar': False})]), md=12)])
        ]),
        dbc.Tab(label="🧠 Rendimiento NLP (Las 4 Métricas)", tab_id="tab-2", children=[
            dbc.Row([
                dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_dist_sent, config={'displayModeBar': False})]), md=6),
                dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_confianza, config={'displayModeBar': False})]), md=6),
            ]),
            dbc.Row([dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_pca, config={'displayModeBar': False})]), md=12)]),
            dbc.Row([dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_tfidf, config={'displayModeBar': False})]), md=12)])
        ]),
        dbc.Tab(label="🌍 Impacto Geográfico", tab_id="tab-3", children=[
            dbc.Row([dbc.Col(html.Div(className="glass-card", children=[dcc.Graph(figure=fig_geo, config={'displayModeBar': False})]), md=12)])
        ])
    ],
    id="tabs", active_tab="tab-1", className="nav-pills mb-4"
)

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1([html.I(className="fa-solid fa-chart-network me-3"), "Analítica Profunda de Sentimientos"], className="my-4 fw-bold text-white"))),
    kpi_row, tabs
], fluid=True, className="p-4")

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
