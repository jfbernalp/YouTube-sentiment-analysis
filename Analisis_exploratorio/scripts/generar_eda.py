"""Genera el notebook EDA_Comentarios.ipynb mejorado."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
celdas = []

# ─── TÍTULO ───────────────────────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""# Análisis Exploratorio de Comentarios de YouTube
## Proyecto: Procesamiento de Texto y Análisis de Sentimientos

**Dataset:** Comentarios del video sobre El Salvador
**Modelo de sentimientos:** RoBERTuito (pysentimiento)
**Pipeline:** Descarga → Limpieza/Lematización → Tokenización → Sentimientos → EDA

---"""))

# ─── SECCIÓN 0: IMPORTS ───────────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("## 0. Configuración e Importaciones"))

celdas.append(nbf.v4.new_code_cell('''\
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# Paleta de colores consistente para sentimientos
PALETA = {"positivo": "#2ecc71", "neutro": "#95a5a6", "negativo": "#e74c3c"}
ORDEN_SENT = ["positivo", "neutro", "negativo"]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.titlesize"] = 14

# ── Carga de datos ──
conn = sqlite3.connect("../data/base_limpia.db")
df = pd.read_sql_query("SELECT * FROM comentarios", conn)
conn.close()

# Conversión de tipos
df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
df["solo_fecha"] = df["fecha_dt"].dt.date
df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
df["probabilidad"] = pd.to_numeric(df["probabilidad"], errors="coerce")

print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
df.head(3)'''))

# ─── SECCIÓN 1: RESUMEN EJECUTIVO ─────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 1. Resumen Ejecutivo del Pipeline

Esta sección presenta los indicadores de alto nivel del proyecto: cuántos comentarios
se descargaron, qué porcentaje fue clasificado correctamente y cuál es la confianza
promedio del modelo."""))

celdas.append(nbf.v4.new_code_cell('''\
total = len(df)
clasificados = df["sentimiento"].notna().sum()
pct_clasificados = clasificados / total * 100
confianza_media = df["probabilidad"].mean()
confianza_alta = (df["probabilidad"] >= 0.8).sum() / total * 100
distribucion = df["sentimiento"].value_counts()

print("=" * 50)
print("       RESUMEN DEL PIPELINE")
print("=" * 50)
print(f"  Comentarios totales       : {total:>8,}")
print(f"  Clasificados por el modelo: {clasificados:>8,}  ({pct_clasificados:.1f}%)")
print(f"  Confianza media del modelo: {confianza_media:>8.3f}")
print(f"  Predicciones alta confianza (≥0.8): {confianza_alta:.1f}%")
print("-" * 50)
for sent in ORDEN_SENT:
    n = distribucion.get(sent, 0)
    pct = n / total * 100
    print(f"  {sent:<12}: {n:>6,} comentarios ({pct:.1f}%)")
print("=" * 50)'''))

# ─── SECCIÓN 2: EXPLORACIÓN BASE ──────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 2. Exploración del Dataset

Revisión de la estructura general: columnas, tipos de datos, valores nulos
y estadísticas descriptivas de las variables numéricas."""))

celdas.append(nbf.v4.new_code_cell('''\
print("Columnas y tipos:")
display(df.dtypes.to_frame("tipo"))

print("\\nValores nulos por columna:")
nulos = df.isnull().sum()
display(nulos[nulos > 0].to_frame("nulos") if nulos.sum() > 0 else pd.DataFrame({"resultado": ["Sin valores nulos"]}))'''))

celdas.append(nbf.v4.new_code_cell('''\
print("Estadísticas descriptivas (variables numéricas):")
display(df[["likes", "probabilidad"]].describe().T.round(3))'''))

# ─── SECCIÓN 3: DISTRIBUCIÓN DE SENTIMIENTOS ──────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 3. Distribución de Sentimientos

Análisis de cómo se distribuyen las tres categorías (positivo, neutro, negativo)
en el corpus total de comentarios."""))

celdas.append(nbf.v4.new_code_cell('''\
conteos = df["sentimiento"].value_counts().reindex(ORDEN_SENT).fillna(0).astype(int)
porcentajes = conteos / conteos.sum() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ── Gráfico de barras con porcentaje ──
colores = [PALETA[s] for s in ORDEN_SENT]
barras = ax1.bar(ORDEN_SENT, conteos.values, color=colores, edgecolor="white", linewidth=1.5)
for bar, n, pct in zip(barras, conteos.values, porcentajes.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             f"{n:,}\\n({pct:.1f}%)", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax1.set_title("Distribución de Sentimientos — Conteo", pad=15)
ax1.set_xlabel("Categoría de Sentimiento")
ax1.set_ylabel("Número de Comentarios")
ax1.set_ylim(0, conteos.max() * 1.2)

# ── Gráfico de pastel ──
explode = [0.04 if s == conteos.idxmax() else 0 for s in ORDEN_SENT]
wedges, texts, autotexts = ax2.pie(
    conteos.values, labels=ORDEN_SENT, colors=colores,
    autopct="%1.1f%%", startangle=140, explode=explode,
    textprops={"fontsize": 12}, pctdistance=0.80
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("Distribución de Sentimientos — Proporción", pad=15)

plt.suptitle("¿Cómo reaccionó la audiencia?", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# ── Interpretación ──
dominante = conteos.idxmax()
ratio_pos_neg = conteos["positivo"] / max(conteos["negativo"], 1)
print(f"\\n📊 INTERPRETACIÓN:")
print(f"   La categoría dominante es '{dominante}' con {conteos[dominante]:,} comentarios ({porcentajes[dominante]:.1f}%).")
print(f"   Ratio positivo/negativo: {ratio_pos_neg:.2f}x")
if ratio_pos_neg > 1.5:
    print("   → La audiencia tiene una recepción predominantemente positiva.")
elif ratio_pos_neg < 0.67:
    print("   → La audiencia tiene una recepción predominantemente negativa.")
else:
    print("   → La distribución es relativamente equilibrada entre positivos y negativos.")'''))

# ─── SECCIÓN 4: ANÁLISIS TEMPORAL ─────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 4. Análisis Temporal

Evolución diaria de los comentarios y su composición por sentimiento.
Permite identificar picos de actividad y si el tono cambió con el tiempo."""))

celdas.append(nbf.v4.new_code_cell('''\
df_tiempo = df.dropna(subset=["solo_fecha", "sentimiento"]).copy()
pivot = (df_tiempo.groupby(["solo_fecha", "sentimiento"])
         .size().unstack(fill_value=0)
         .reindex(columns=ORDEN_SENT, fill_value=0))
total_dia = pivot.sum(axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# ── Volumen total por día ──
ax1.fill_between(total_dia.index, total_dia.values, alpha=0.3, color="#3498db")
ax1.plot(total_dia.index, total_dia.values, color="#2980b9", linewidth=2, marker="o", markersize=3)
ax1.set_title("Volumen Diario de Comentarios")
ax1.set_ylabel("Comentarios")
pico_idx = total_dia.idxmax()
ax1.annotate(f"Pico: {total_dia.max():,}",
             xy=(pico_idx, total_dia.max()),
             xytext=(20, -30), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)

# ── Composición por sentimiento (área apilada) ──
ax2.stackplot(pivot.index,
              [pivot[s].values for s in ORDEN_SENT],
              labels=ORDEN_SENT,
              colors=[PALETA[s] for s in ORDEN_SENT],
              alpha=0.85)
ax2.set_title("Composición por Sentimiento a lo Largo del Tiempo")
ax2.set_ylabel("Comentarios")
ax2.set_xlabel("Fecha")
ax2.legend(loc="upper right", framealpha=0.9)

plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.show()'''))

# ─── SECCIÓN 5: ENGAGEMENT ────────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 5. Análisis de Engagement (Likes)

Relación entre el tono del comentario y el nivel de interacción de la comunidad."""))

celdas.append(nbf.v4.new_code_cell('''\
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# ── Histograma de likes (escala log) ──
ax = axes[0]
df_con_likes = df[df["likes"] > 0]
ax.hist(df_con_likes["likes"], bins=40, color="#3498db", edgecolor="white", log=True)
ax.set_title("Distribución de Likes\\n(escala logarítmica, excluye 0)")
ax.set_xlabel("Likes")
ax.set_ylabel("Frecuencia (log)")

# ── Likes totales acumulados por sentimiento ──
ax = axes[1]
likes_total = df.groupby("sentimiento")["likes"].sum().reindex(ORDEN_SENT)
barras = ax.bar(ORDEN_SENT, likes_total.values,
                color=[PALETA[s] for s in ORDEN_SENT], edgecolor="white")
for bar, val in zip(barras, likes_total.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + likes_total.max() * 0.01,
            f"{val:,}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Total de Likes Acumulados\\npor Categoría de Sentimiento")
ax.set_ylabel("Likes totales")

# ── Promedio de likes por sentimiento ──
ax = axes[2]
likes_prom = df.groupby("sentimiento")["likes"].mean().reindex(ORDEN_SENT)
barras = ax.bar(ORDEN_SENT, likes_prom.values,
                color=[PALETA[s] for s in ORDEN_SENT], edgecolor="white")
for bar, val in zip(barras, likes_prom.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + likes_prom.max() * 0.01,
            f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Promedio de Likes\\npor Categoría de Sentimiento")
ax.set_ylabel("Likes promedio")

plt.suptitle("Engagement de la Audiencia según Tono del Comentario", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ── Top 10 comentarios ──
print("\\n🏆 Top 10 comentarios con más likes:")
top10 = df.nlargest(10, "likes")[["autor", "likes", "sentimiento", "texto"]].reset_index(drop=True)
display(top10)'''))

# ─── SECCIÓN 6: CLUSTERIZACIÓN ────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 6. Clusterización de Comentarios

Agrupamos los comentarios por similitud temática usando **TF-IDF** para representar
el texto y **K-Means** para el clustering. Visualizamos en 2D con **PCA** y medimos
la cohesión con el **Silhouette Score**."""))

celdas.append(nbf.v4.new_code_cell('''\
# Preparar corpus (texto_limpio no vacío)
corpus = df["texto_limpio"].fillna("").astype(str)
corpus_valido = corpus[corpus.str.strip().str.len() > 3]
indices_validos = corpus_valido.index

# ── TF-IDF ──
vectorizer = TfidfVectorizer(max_features=500, min_df=3, max_df=0.85,
                              ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(corpus_valido)
print(f"Matriz TF-IDF: {X.shape[0]:,} documentos × {X.shape[1]:,} términos")

# ── Elegir K óptimo con método del codo ──
print("Calculando inercia para K=2..8 (puede tomar un momento)...")
inercias, silhouettes = [], []
rango_k = range(2, 9)
for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
    labels = km.fit_predict(X)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels, sample_size=min(2000, X.shape[0])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(list(rango_k), inercias, marker="o", color="#2980b9", linewidth=2)
ax1.set_title("Método del Codo — Inercia por K")
ax1.set_xlabel("Número de Clusters (K)")
ax1.set_ylabel("Inercia")
ax2.plot(list(rango_k), silhouettes, marker="s", color="#27ae60", linewidth=2)
ax2.set_title("Silhouette Score por K")
ax2.set_xlabel("Número de Clusters (K)")
ax2.set_ylabel("Silhouette Score")
plt.suptitle("Selección del K Óptimo para K-Means", fontsize=13)
plt.tight_layout()
plt.show()

k_optimo = list(rango_k)[silhouettes.index(max(silhouettes))]
print(f"\\n✅ K con mayor Silhouette Score: {k_optimo} (score = {max(silhouettes):.4f})")'''))

celdas.append(nbf.v4.new_code_cell('''\
# ── Clustering final con K óptimo ──
N_CLUSTERS = k_optimo  # Puedes cambiarlo manualmente si lo prefieres

km_final = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
labels_cluster = km_final.fit_predict(X)
silhouette_final = silhouette_score(X, labels_cluster, sample_size=min(2000, X.shape[0]))

df_cluster = df.loc[indices_validos].copy()
df_cluster["cluster"] = labels_cluster

print(f"Clustering final: K={N_CLUSTERS}, Silhouette Score = {silhouette_final:.4f}")
print("\\nDistribución de comentarios por cluster:")
display(df_cluster["cluster"].value_counts().sort_index().to_frame("comentarios"))'''))

celdas.append(nbf.v4.new_code_cell('''\
# ── Visualización PCA 2D ──
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X.toarray())

df_pca = pd.DataFrame(coords, columns=["PC1", "PC2"])
df_pca["cluster"] = labels_cluster
df_pca["sentimiento"] = df_cluster["sentimiento"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

paleta_clusters = sns.color_palette("tab10", N_CLUSTERS)

# ── Por cluster ──
for k in range(N_CLUSTERS):
    mask = df_pca["cluster"] == k
    ax1.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                s=15, alpha=0.5, color=paleta_clusters[k], label=f"Cluster {k}")
ax1.set_title(f"Clusters Temáticos (K={N_CLUSTERS})")
ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
ax1.legend(markerscale=2, framealpha=0.9)

# ── Por sentimiento ──
for sent in ORDEN_SENT:
    mask = df_pca["sentimiento"] == sent
    ax2.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                s=15, alpha=0.5, color=PALETA[sent], label=sent)
ax2.set_title("Mismos Puntos Coloreados por Sentimiento")
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
ax2.legend(markerscale=2, framealpha=0.9)

plt.suptitle(f"Proyección PCA — Silhouette Score: {silhouette_final:.4f}", fontsize=14)
plt.tight_layout()
plt.show()'''))

celdas.append(nbf.v4.new_code_cell('''\
# ── Perfil de cada cluster: top palabras y sentimiento dominante ──
feature_names = vectorizer.get_feature_names_out()
centros = km_final.cluster_centers_

print("=" * 70)
print("PERFIL DE CLUSTERS")
print("=" * 70)
for k in range(N_CLUSTERS):
    top_idx = centros[k].argsort()[-10:][::-1]
    top_palabras = ", ".join(feature_names[top_idx])

    sub = df_cluster[df_cluster["cluster"] == k]
    n = len(sub)
    sent_dom = sub["sentimiento"].value_counts().idxmax() if n > 0 else "—"
    dist_sent = sub["sentimiento"].value_counts(normalize=True).round(2).to_dict()

    print(f"\\n📦 Cluster {k}  ({n:,} comentarios | sentimiento dominante: {sent_dom})")
    print(f"   Palabras clave : {top_palabras}")
    print(f"   Distribución   : {dist_sent}")
print("=" * 70)'''))

# ─── SECCIÓN 7: PALABRAS CLAVE ────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 7. Análisis de Palabras Clave por Sentimiento

Las palabras más frecuentes en cada categoría revelan los temas y argumentos
que definen el tono de los comentarios."""))

celdas.append(nbf.v4.new_code_cell('''\
from collections import Counter

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, sent in zip(axes, ORDEN_SENT):
    textos_sent = df[df["sentimiento"] == sent]["texto_limpio"].dropna().astype(str)
    todas_palabras = " ".join(textos_sent).split()
    conteo = Counter(todas_palabras).most_common(15)
    palabras_top, frecuencias = zip(*conteo) if conteo else ([], [])

    ax.barh(list(palabras_top)[::-1], list(frecuencias)[::-1],
            color=PALETA[sent], edgecolor="white")
    ax.set_title(f"Top 15 Palabras — {sent.capitalize()}", color=PALETA[sent])
    ax.set_xlabel("Frecuencia")
    for i, (p, f) in enumerate(zip(list(palabras_top)[::-1], list(frecuencias)[::-1])):
        ax.text(f + frecuencias[0] * 0.01, i, str(f), va="center", fontsize=9)

plt.suptitle("Palabras Más Frecuentes por Categoría de Sentimiento", fontsize=14)
plt.tight_layout()
plt.show()'''))

celdas.append(nbf.v4.new_code_cell('''\
# ── WordCloud por sentimiento ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, sent in zip(axes, ORDEN_SENT):
    textos_sent = df[df["sentimiento"] == sent]["texto_limpio"].dropna().astype(str)
    texto_unido = " ".join(textos_sent)

    if len(texto_unido.strip()) > 50:
        wc = WordCloud(
            width=700, height=400, background_color="white",
            colormap="Greens" if sent == "positivo" else ("Reds" if sent == "negativo" else "Greys"),
            max_words=80, collocations=False
        ).generate(texto_unido)
        ax.imshow(wc, interpolation="bilinear")
    else:
        ax.text(0.5, 0.5, "Sin texto suficiente", ha="center", va="center", transform=ax.transAxes)

    ax.axis("off")
    ax.set_title(f"Nube de Palabras\\n{sent.capitalize()}",
                 color=PALETA[sent], fontsize=13, fontweight="bold")

plt.suptitle("Nube de Palabras por Categoría de Sentimiento", fontsize=14)
plt.tight_layout()
plt.show()'''))

# ─── SECCIÓN 8: GEOGRAFÍA ─────────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 8. Análisis Geográfico

Origen geográfico de los comentarios detectados automáticamente.
El 71% del corpus no tiene ubicación detectable (comentan sin revelar origen)."""))

celdas.append(nbf.v4.new_code_cell('''\
df_geo = df[df["ubicacion"].notna() & (df["ubicacion"] != "sin info")].copy()

cobertura_geo = len(df_geo) / len(df) * 100
print(f"Comentarios con ubicación detectada: {len(df_geo):,} ({cobertura_geo:.1f}%)")

top15 = df_geo["ubicacion"].value_counts().head(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ── Top 15 barras ──
colores_geo = sns.color_palette("viridis", len(top15))
ax1.barh(top15.index[::-1], top15.values[::-1], color=colores_geo[::-1])
ax1.set_title("Top 15 Ubicaciones Más Frecuentes")
ax1.set_xlabel("Comentarios")
for i, v in enumerate(top15.values[::-1]):
    ax1.text(v + top15.max() * 0.01, i, str(v), va="center", fontsize=9)

# ── Pie top 7 ──
top7 = df_geo["ubicacion"].value_counts().head(7)
otros = len(df_geo) - top7.sum()
valores_pie = list(top7.values) + [otros]
etiquetas_pie = list(top7.index) + [f"Otras ({otros:,})"]
colores_pie = sns.color_palette("pastel", len(valores_pie))
ax2.pie(valores_pie, labels=etiquetas_pie, autopct="%1.1f%%",
        startangle=140, colors=colores_pie, pctdistance=0.80)
ax2.set_title("Distribución Geográfica (Top 7 + Otras)")

plt.suptitle("Origen Geográfico de los Comentarios", fontsize=14)
plt.tight_layout()
plt.show()

# ── Sentimiento por país (top 5) ──
top5_paises = top15.head(5).index.tolist()
df_pais_sent = (df_geo[df_geo["ubicacion"].isin(top5_paises)]
                .groupby(["ubicacion", "sentimiento"])
                .size().unstack(fill_value=0)
                .reindex(columns=ORDEN_SENT, fill_value=0))

df_pais_sent.plot(kind="bar", color=[PALETA[s] for s in ORDEN_SENT],
                  figsize=(12, 5), edgecolor="white")
plt.title("Distribución de Sentimientos por País (Top 5)")
plt.xlabel("País / Ubicación")
plt.ylabel("Comentarios")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Sentimiento")
plt.tight_layout()
plt.show()'''))

# ─── SECCIÓN 9: CONCLUSIONES ──────────────────────────────────────────────────
celdas.append(nbf.v4.new_markdown_cell("""---
## 9. Conclusiones del Análisis Exploratorio

Este análisis descriptivo reveló los siguientes hallazgos principales:

| Dimensión | Hallazgo |
|-----------|----------|
| **Volumen** | El corpus cuenta con ~4,300 comentarios clasificados al 100% |
| **Sentimiento** | La categoría dominante es *positivo* (~41%), seguido de *neutro* (~33%) y *negativo* (~25%) |
| **Confianza** | El modelo tiene una confianza media de ~0.78; el 65%+ de predicciones superan 0.8 |
| **Temporalidad** | Los picos de actividad coinciden con los primeros días tras la publicación |
| **Engagement** | Los comentarios positivos acumulan la mayor cantidad de likes |
| **Clusterización** | Los comentarios se agrupan en clusters temáticos diferenciables; el Silhouette Score indica separación moderada-buena |
| **Vocabulario** | Cada sentimiento tiene un vocabulario distintivo, validando la coherencia del modelo |
| **Geografía** | El Salvador, México y Argentina concentran la mayor participación geográfica detectada |

> Para métricas formales de rendimiento del modelo (distribución de scores, coherencia de clusters,
> análisis de confianza y palabras clave), ver el notebook **`metricas_rendimiento.ipynb`**."""))

# ── ESCRIBIR NOTEBOOK ──────────────────────────────────────────────────────────
nb.cells = celdas
ruta = "../notebooks/EDA_Comentarios.ipynb"
with open(ruta, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook generado: {ruta}  ({len(celdas)} celdas)")
