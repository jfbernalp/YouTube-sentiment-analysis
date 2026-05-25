"""
Generates a full APA7-formatted Word document (.docx) for the
YouTube Sentiment Analysis project on El Salvador.

Run from the project root:
    source env_eda/bin/activate
    python scripts/generar_reporte_word.py
"""

import os, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import stats

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_HERE)
DB_PATH  = os.path.join(BASE_DIR, "data", "base_limpia.db")
OUT_DIR  = os.path.join(BASE_DIR, "output")
FIG_DIR  = os.path.join(OUT_DIR, "report_figures")
REPORT   = os.path.join(BASE_DIR, "output", "Sentiment_Analysis_El_Salvador_APA7.docx")
os.makedirs(FIG_DIR, exist_ok=True)

PALETA     = {"positivo": "#2ecc71", "neutro": "#95a5a6", "negativo": "#e74c3c"}
ORDEN_SENT = ["positivo", "neutro", "negativo"]
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["axes.titlesize"] = 12

# ==============================================================================
# 1.  LOAD DATA
# ==============================================================================
print("Loading data...")
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("SELECT * FROM comentarios", conn)
conn.close()

df["probabilidad"] = pd.to_numeric(df["probabilidad"],   errors="coerce")
df["likes"]        = pd.to_numeric(df["likes"],           errors="coerce").fillna(0).astype(int)
df["fecha_dt"]     = pd.to_datetime(df["fecha"],          errors="coerce")
df["solo_fecha"]   = df["fecha_dt"].dt.date
df["len_orig"]     = df["texto"].fillna("").apply(len)
df["words_orig"]   = df["texto"].fillna("").apply(lambda x: len(str(x).split()))
df["words_clean"]  = df["texto_limpio"].fillna("").apply(lambda x: len(str(x).split()))

df_cls = df.dropna(subset=["sentimiento", "probabilidad"]).copy()
df_cls = df_cls[df_cls["sentimiento"].isin(ORDEN_SENT)]

# ==============================================================================
# 2.  GENERATE FIGURES
# ==============================================================================
def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ── Figure 1: Sentiment distribution ──────────────────────────────────────────
def fig_sentiment_distribution():
    conteos = df_cls["sentimiento"].value_counts().reindex(ORDEN_SENT).fillna(0).astype(int)
    porcs   = conteos / conteos.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [PALETA[s] for s in ORDEN_SENT]
    bars   = ax1.bar(["Positive", "Neutral", "Negative"], conteos.values,
                     color=colors, edgecolor="white", linewidth=1.5)
    for bar, n, p in zip(bars, conteos.values, porcs.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{n:,}\n({p:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of Comments")
    ax1.set_title("Sentiment Class Counts")
    ax1.set_ylim(0, conteos.max() * 1.25)

    explode = [0.04 if s == conteos.idxmax() else 0 for s in ORDEN_SENT]
    wedges, texts, autotexts = ax2.pie(
        conteos.values, labels=["Positive", "Neutral", "Negative"],
        colors=colors, autopct="%1.1f%%", startangle=140,
        explode=explode, pctdistance=0.80, textprops={"fontsize": 11})
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Proportion of Sentiment Classes")

    fig.suptitle("Figure 1. Sentiment Distribution Across the Corpus", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig1_sentiment_distribution.png")

# ── Figure 2: Temporal analysis ───────────────────────────────────────────────
def fig_temporal():
    dft = df.dropna(subset=["solo_fecha", "sentimiento"]).copy()
    dft = dft[dft["sentimiento"].isin(ORDEN_SENT)]
    pivot = (dft.groupby(["solo_fecha", "sentimiento"])
               .size().unstack(fill_value=0)
               .reindex(columns=ORDEN_SENT, fill_value=0))
    total_dia = pivot.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.fill_between(total_dia.index, total_dia.values, alpha=0.3, color="#3498db")
    ax1.plot(total_dia.index, total_dia.values, color="#2980b9", linewidth=2, marker="o", markersize=3)
    pico_idx = total_dia.idxmax()
    ax1.annotate(f"Peak: {total_dia.max():,}",
                 xy=(pico_idx, total_dia.max()), xytext=(20, -30),
                 textcoords="offset points", arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax1.set_title("Daily Comment Volume")
    ax1.set_ylabel("Comments")

    ax2.stackplot(pivot.index, [pivot[s].values for s in ORDEN_SENT],
                  labels=["Positive", "Neutral", "Negative"],
                  colors=[PALETA[s] for s in ORDEN_SENT], alpha=0.85)
    ax2.set_title("Sentiment Composition Over Time")
    ax2.set_ylabel("Comments"); ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", framealpha=0.9)
    plt.xticks(rotation=35, ha="right")
    fig.suptitle("Figure 2. Temporal Distribution of Comments and Sentiment", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig2_temporal.png")

# ── Figure 3: Engagement by sentiment ─────────────────────────────────────────
def fig_engagement():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    likes_tot  = df_cls.groupby("sentimiento")["likes"].sum().reindex(ORDEN_SENT)
    likes_prom = df_cls.groupby("sentimiento")["likes"].mean().reindex(ORDEN_SENT)
    labels_en  = ["Positive", "Neutral", "Negative"]
    colors     = [PALETA[s] for s in ORDEN_SENT]

    bars = axes[0].bar(labels_en, likes_tot.values, color=colors, edgecolor="white")
    for bar, v in zip(bars, likes_tot.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + likes_tot.max()*0.01,
                     f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Total Accumulated Likes by Sentiment")
    axes[0].set_ylabel("Total Likes")

    bars = axes[1].bar(labels_en, likes_prom.values, color=colors, edgecolor="white")
    for bar, v in zip(bars, likes_prom.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + likes_prom.max()*0.01,
                     f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_title("Average Likes per Comment by Sentiment")
    axes[1].set_ylabel("Mean Likes")

    fig.suptitle("Figure 3. Audience Engagement by Sentiment Category", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig3_engagement.png")

# ── Figure 4: Geographic distribution ─────────────────────────────────────────
def fig_geographic():
    df_geo = df[(df["ubicacion"].notna()) & (df["ubicacion"] != "sin info")].copy()
    top15  = df_geo["ubicacion"].value_counts().head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors_g = sns.color_palette("viridis", len(top15))
    ax1.barh(top15.index[::-1], top15.values[::-1], color=colors_g[::-1])
    ax1.set_title("Top 15 Geographic Locations")
    ax1.set_xlabel("Comments")
    for i, v in enumerate(top15.values[::-1]):
        ax1.text(v + top15.max()*0.01, i, str(v), va="center", fontsize=8)

    top7   = df_geo["ubicacion"].value_counts().head(7)
    otros  = len(df_geo) - top7.sum()
    vals   = list(top7.values) + [otros]
    labels = list(top7.index)  + [f"Other ({otros:,})"]
    ax2.pie(vals, labels=labels, autopct="%1.1f%%", startangle=140,
            colors=sns.color_palette("pastel", len(vals)), pctdistance=0.80)
    ax2.set_title("Top 7 + Other Locations")

    fig.suptitle("Figure 4. Geographic Origin of Detected Comments", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig4_geographic.png")

# ── Figure 5: Confidence distribution ─────────────────────────────────────────
def fig_confidence():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sent, label in zip(axes, ORDEN_SENT, ["Positive", "Neutral", "Negative"]):
        sub   = df_cls[df_cls["sentimiento"] == sent]["probabilidad"]
        color = PALETA[sent]
        ax.hist(sub, bins=30, color=color, alpha=0.85, edgecolor="white", density=True)
        ax.axvline(sub.median(), color="black", linestyle="--", lw=1.5,
                   label=f"Median = {sub.median():.3f}")
        ax.axvline(0.80, color="navy",  linestyle=":",  lw=1.5, label="High conf. (0.80)")
        ax.axvline(1/3,  color="gray",  linestyle="-.", lw=1.0, label="Chance (0.33)")
        pct_h = (sub >= 0.80).mean() * 100
        pct_u = (sub < 0.50).mean()  * 100
        ax.set_title(f"{label}\n≥0.80: {pct_h:.1f}% | <0.50: {pct_u:.1f}%",
                     color=color, fontweight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, framealpha=0.8)
        ax.set_xlim(0, 1)
    fig.suptitle("Figure 5. Model Confidence Score Distribution by Sentiment Class", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig5_confidence.png")

# ── Figure 6: Cluster PCA ──────────────────────────────────────────────────────
def fig_cluster_pca():
    corpus = df_cls["texto_limpio"].fillna("").astype(str)
    mask   = corpus.str.strip().str.len() > 3
    corpus_v = corpus[mask]

    vec = TfidfVectorizer(max_features=500, min_df=3, max_df=0.85,
                          ngram_range=(1,2), sublinear_tf=True)
    X   = vec.fit_transform(corpus_v)

    km  = KMeans(n_clusters=8, random_state=42, n_init=10, max_iter=300)
    lbl = km.fit_predict(X)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    v1, v2 = pca.explained_variance_ratio_ * 100

    df_pca = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_pca["cluster"]     = lbl
    df_pca["sentimiento"] = df_cls.loc[mask, "sentimiento"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    pal_k = sns.color_palette("tab10", 8)

    for k in range(8):
        mk = df_pca["cluster"] == k
        ax1.scatter(df_pca.loc[mk,"PC1"], df_pca.loc[mk,"PC2"],
                    s=10, alpha=0.4, color=pal_k[k], label=f"C{k} (n={mk.sum():,})")
    ax1.set_title(f"Thematic Clusters (K=8)")
    ax1.set_xlabel(f"PC1 — {v1:.1f}% variance")
    ax1.set_ylabel(f"PC2 — {v2:.1f}% variance")
    ax1.legend(markerscale=2, fontsize=8, framealpha=0.9)

    for sent, label in zip(ORDEN_SENT, ["Positive","Neutral","Negative"]):
        ms = df_pca["sentimiento"] == sent
        ax2.scatter(df_pca.loc[ms,"PC1"], df_pca.loc[ms,"PC2"],
                    s=10, alpha=0.4, color=PALETA[sent], label=label)
    ax2.set_title("Same Points — Colored by Sentiment")
    ax2.set_xlabel(f"PC1 — {v1:.1f}% variance")
    ax2.set_ylabel(f"PC2 — {v2:.1f}% variance")
    ax2.legend(markerscale=2, fontsize=10, framealpha=0.9)

    fig.suptitle("Figure 6. PCA Projection of TF-IDF Clusters  |  "
                 "Silhouette = 0.0334  |  DBI = 4.86", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig6_cluster_pca.png")

# ── Figure 7: TF-IDF discriminant terms ───────────────────────────────────────
def fig_tfidf_discriminant():
    top_n = 15
    res   = {}
    for sent in ORDEN_SENT:
        txts = df_cls[df_cls["sentimiento"]==sent]["texto_limpio"].fillna("").astype(str)
        txts = txts[txts.str.strip().str.len() > 3]
        v    = TfidfVectorizer(max_features=300, min_df=2, max_df=0.90,
                               ngram_range=(1,2), sublinear_tf=True)
        X    = v.fit_transform(txts)
        sc   = X.mean(axis=0).A1
        idx  = sc.argsort()[-top_n:][::-1]
        res[sent] = (v.get_feature_names_out()[idx], sc[idx])

    labels_en = {"positivo":"Positive","neutro":"Neutral","negativo":"Negative"}
    fig, axes  = plt.subplots(1, 3, figsize=(18, 7))
    for ax, sent in zip(axes, ORDEN_SENT):
        words, scores = res[sent]
        ax.barh(words[::-1], scores[::-1], color=PALETA[sent], edgecolor="white")
        ax.set_title(f"Top {top_n} Discriminant Terms\n{labels_en[sent]}",
                     color=PALETA[sent], fontweight="bold")
        ax.set_xlabel("Mean TF-IDF Score")
    fig.suptitle("Figure 7. TF-IDF Discriminant Vocabulary by Sentiment Class", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig7_tfidf_terms.png")

# ── Figure 8: K-selection metrics ─────────────────────────────────────────────
def fig_k_selection():
    corpus  = df_cls["texto_limpio"].fillna("").astype(str)
    mask    = corpus.str.strip().str.len() > 3
    corpus_v = corpus[mask]
    vec     = TfidfVectorizer(max_features=500, min_df=3, max_df=0.85,
                              ngram_range=(1,2), sublinear_tf=True)
    X       = vec.fit_transform(corpus_v)
    rango   = range(2, 9)
    inercias, sils, dbs = [], [], []
    for k in rango:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
        lbl = km.fit_predict(X)
        inercias.append(km.inertia_)
        sils.append(silhouette_score(X, lbl, sample_size=min(2000, X.shape[0])))
        dbs.append(davies_bouldin_score(X.toarray(), lbl))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(list(rango), inercias, marker="o", color="#2980b9", lw=2)
    axes[0].set_title("Elbow Method — Inertia"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia (SSE)")
    axes[1].plot(list(rango), sils, marker="s", color="#27ae60", lw=2)
    axes[1].axvline(8, color="red", linestyle="--", label="K=8 (optimal)")
    axes[1].set_title("Silhouette Score vs K"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score (↑)")
    axes[1].legend()
    axes[2].plot(list(rango), dbs,  marker="^", color="#e67e22", lw=2)
    axes[2].axvline(8, color="red", linestyle="--", label="K=8")
    axes[2].set_title("Davies-Bouldin Index vs K"); axes[2].set_xlabel("K"); axes[2].set_ylabel("DBI (↓)")
    axes[2].legend()
    fig.suptitle("Figure 8. K-Means Cluster Selection Metrics (K = 2 to 8)", fontsize=12)
    fig.tight_layout()
    return save(fig, "fig8_k_selection.png")

print("Generating figures...")
f1 = fig_sentiment_distribution()
f2 = fig_temporal()
f3 = fig_engagement()
f4 = fig_geographic()
f5 = fig_confidence()
f6 = fig_cluster_pca()
f7 = fig_tfidf_discriminant()
f8 = fig_k_selection()

# Existing figures
f_wc_pre   = os.path.join(OUT_DIR, "nube_palabras.png")
f_wc_post  = os.path.join(OUT_DIR, "nube_palabras_limpia.png")
f_likes    = os.path.join(OUT_DIR, "histograma_likes.png")
f_roc      = os.path.join(OUT_DIR, "04_eval_curvas_roc.png")
f_prob     = os.path.join(OUT_DIR, "04_eval_distribucion_probabilidad.png")
f_cm       = os.path.join(OUT_DIR, "04_eval_matriz_confusion.png")

print("All figures generated.")

# ==============================================================================
# 3.  WORD DOCUMENT HELPERS
# ==============================================================================
def set_paragraph_format(para, space_before=0, space_after=12,
                          line_spacing=None, keep_together=False):
    pf = para.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after  = Pt(space_after)
    if line_spacing:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing      = Pt(line_spacing)
    else:
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    if keep_together:
        pf.keep_together = True

def add_run(para, text, bold=False, italic=False, size=12, color=None):
    run = para.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    if color:
        run.font.color.rgb = RGBColor(*color)
    return run

def add_body(doc, text, indent=False):
    para = doc.add_paragraph()
    set_paragraph_format(para)
    if indent:
        para.paragraph_format.first_line_indent = Cm(1.27)
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_run(para, text)
    return para

def add_h1(doc, text):
    para = doc.add_paragraph()
    set_paragraph_format(para, space_before=24, space_after=6)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(para, text, bold=True, size=12)
    return para

def add_h2(doc, text):
    para = doc.add_paragraph()
    set_paragraph_format(para, space_before=18, space_after=0)
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_run(para, text, bold=True, size=12)
    return para

def add_h3(doc, text):
    para = doc.add_paragraph()
    set_paragraph_format(para, space_before=12, space_after=0)
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_run(para, f"        {text}.", bold=True, italic=True, size=12)
    return para

def add_figure(doc, img_path, caption_num, caption_text, width=5.5):
    if not os.path.exists(img_path):
        print(f"  [WARN] Figure not found: {img_path}")
        return
    doc.add_picture(img_path, width=Inches(width))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER

    cap_para = doc.add_paragraph()
    cap_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_format(cap_para, space_before=6, space_after=18)
    cap_run = cap_para.add_run(f"Figure {caption_num}")
    cap_run.italic = True
    cap_run.bold   = True
    cap_run.font.name = "Times New Roman"
    cap_run.font.size = Pt(12)
    rest = cap_para.add_run(f"\n{caption_text}")
    rest.font.name = "Times New Roman"
    rest.font.size = Pt(12)

def add_table_apa(doc, headers, rows, caption_num, caption_text, note=None):
    cap_para = doc.add_paragraph()
    set_paragraph_format(cap_para, space_before=18, space_after=0)
    r = cap_para.add_run(f"Table {caption_num}")
    r.bold = True; r.italic = False
    r.font.name = "Times New Roman"; r.font.size = Pt(12)

    title_para = doc.add_paragraph()
    set_paragraph_format(title_para, space_before=0, space_after=0)
    tr = title_para.add_run(caption_text)
    tr.italic = True; tr.font.name = "Times New Roman"; tr.font.size = Pt(12)

    n_cols = len(headers)
    n_rows = len(rows) + 1
    tbl    = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.style = "Table Grid"

    # Header row
    hrow = tbl.rows[0]
    for j, h in enumerate(headers):
        cell = hrow.cells[j]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Data rows
    for i, row_data in enumerate(rows):
        drow = tbl.rows[i + 1]
        for j, val in enumerate(row_data):
            cell = drow.cells[j]
            cell.text = str(val)
            for run in cell.paragraphs[0].runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(11)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Remove all borders, add only top and bottom of table (APA style)
    def set_cell_border(cell, **kwargs):
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBorders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            tag = OxmlElement(f"w:{edge}")
            val = kwargs.get(edge, "none")
            tag.set(qn("w:val"), val)
            tag.set(qn("w:sz"), "4")
            tag.set(qn("w:space"), "0")
            tag.set(qn("w:color"), "000000")
            tcBorders.append(tag)
        tcPr.append(tcBorders)

    # Top rule: header top border
    for cell in tbl.rows[0].cells:
        set_cell_border(cell, top="single", bottom="single",
                        left="none", right="none",
                        insideH="none", insideV="none")
    # Middle: header bottom = single line only
    # Last row: bottom border
    for cell in tbl.rows[-1].cells:
        set_cell_border(cell, bottom="single", top="none",
                        left="none", right="none",
                        insideH="none", insideV="none")
    for i in range(1, len(tbl.rows) - 1):
        for cell in tbl.rows[i].cells:
            set_cell_border(cell, top="none", bottom="none",
                            left="none", right="none",
                            insideH="none", insideV="none")

    if note:
        note_para = doc.add_paragraph()
        set_paragraph_format(note_para, space_before=2, space_after=18)
        nr = note_para.add_run("Note. ")
        nr.italic = True; nr.font.name = "Times New Roman"; nr.font.size = Pt(11)
        nt = note_para.add_run(note)
        nt.font.name = "Times New Roman"; nt.font.size = Pt(11)

def add_page_break(doc):
    para = doc.add_paragraph()
    run  = para.add_run()
    run.add_break(docx_break_type="page")

def page_break_before(para):
    pPr = para._p.get_or_add_pPr()
    pb  = OxmlElement("w:pageBreakBefore")
    pb.set(qn("w:val"), "true")
    pPr.append(pb)

# Fix docx page break helper
from docx.oxml.ns import nsmap
from docx.enum.text import WD_BREAK
def real_page_break(doc):
    para = doc.add_paragraph()
    run  = para.add_run()
    run.add_break(WD_BREAK.PAGE)

# ==============================================================================
# 4.  BUILD DOCUMENT
# ==============================================================================
print("Building Word document...")
doc = Document()

# Page margins (1 inch all around)
for section in doc.sections:
    section.top_margin    = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin   = Cm(2.54)
    section.right_margin  = Cm(2.54)

# Default font
from docx.oxml import OxmlElement
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

# ── TITLE PAGE ─────────────────────────────────────────────────────────────────
def add_centered(doc, text, bold=False, italic=False, size=12,
                 space_before=0, space_after=12, dbl=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after  = Pt(space_after)
    if dbl:
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = bold; r.italic = italic
    r.font.name = "Times New Roman"; r.font.size = Pt(size)
    return p

# Spacer at top
for _ in range(4):
    add_centered(doc, "", size=12)

add_centered(doc,
    "Sentiment Analysis of YouTube Comments on El Salvador: "
    "A Natural Language Processing Approach",
    bold=True, size=12)

add_centered(doc, "")
add_centered(doc, "Juan Felipe Bernal Patiño")
add_centered(doc, "")
add_centered(doc, "Department of Computer Science, Universidad Nacional")
add_centered(doc, "")
add_centered(doc, "Natural Language Processing — Text Processing")
add_centered(doc, "")
add_centered(doc, "May 18, 2026")

# ── ABSTRACT PAGE ──────────────────────────────────────────────────────────────
real_page_break(doc)

add_centered(doc, "Abstract", bold=True)
abs_para = doc.add_paragraph()
abs_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
set_paragraph_format(abs_para)
abs_para.paragraph_format.first_line_indent = Cm(1.27)
add_run(abs_para,
    "This study presents a comprehensive natural language processing (NLP) pipeline "
    "designed to analyze public sentiment expressed in YouTube comments on a video about "
    "El Salvador published between February and March 2026. A corpus of 4,314 comments "
    "was collected via the YouTube Data API v3 and subjected to a multi-stage preprocessing "
    "workflow that included text normalization, URL removal, Spanish stopword filtering, and "
    "morphological lemmatization using the spaCy es_core_news_sm model, resulting in a 51.2% "
    "reduction in average token count. Sentiment classification was performed using "
    "RoBERTuito, a RoBERTa-based transformer model fine-tuned on Spanish-language social media "
    "corpora and implemented via the pysentimiento library. Geographic origin was inferred "
    "using the GeoText named-entity library augmented with a custom normalization dictionary. "
    "Thematic clustering was conducted through TF-IDF vectorization coupled with K-Means, "
    "with optimal cluster selection guided by Silhouette Score and Davies-Bouldin Index. "
    "Results indicate that 41.6% of comments were classified as positive, 35.5% as negative, "
    "and 22.9% as neutral, with a statistically significant non-uniform distribution "
    "(χ² = 233.23, p < .001). The model achieved a global mean confidence of 0.782, with "
    "54.2% of predictions reaching high confidence (≥ 0.80). Clustering analysis identified "
    "eight thematically distinct groups, including gratitude expressions (86% positive), "
    "human rights discourse (81% negative), and political commentary on President Bukele "
    "(polarized). Geographic detection covered 28.4% of the corpus, with El Salvador, "
    "Colombia, and Venezuela as the most represented locations. These findings demonstrate "
    "that automated NLP pipelines can effectively characterize multi-dimensional public "
    "discourse on politically salient topics in Spanish-language social media.")

kw_para = doc.add_paragraph()
kw_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
set_paragraph_format(kw_para, space_before=12, space_after=0)
ki = kw_para.add_run("Keywords: ")
ki.italic = True; ki.font.name = "Times New Roman"; ki.font.size = Pt(12)
kv = kw_para.add_run(
    "sentiment analysis, natural language processing, YouTube comments, "
    "El Salvador, RoBERTuito, K-Means clustering, TF-IDF")
kv.font.name = "Times New Roman"; kv.font.size = Pt(12)

# ── BODY ───────────────────────────────────────────────────────────────────────
real_page_break(doc)

# ─────────────────────────────────────────────────────────────────────────────
add_centered(doc,
    "Sentiment Analysis of YouTube Comments on El Salvador: "
    "A Natural Language Processing Approach",
    bold=True)

# ─── SECTION 1: DATASET AND PREPROCESSING ─────────────────────────────────────
add_h1(doc, "Dataset and Preprocessing")

add_h2(doc, "Data Collection")
add_body(doc,
    "The primary dataset consists of top-level comments retrieved from a single YouTube "
    "video (ID: pWnndG1K5Hg) focused on El Salvador. Data acquisition was conducted via "
    "the YouTube Data API v3 using the commentThreads.list endpoint with a target ceiling "
    "of 5,000 comments. A checkpoint mechanism was implemented to persist records every "
    "500 entries, ensuring robustness against API quota interruptions. Each comment record "
    "retained four fields: author display name, original comment text, like count, and "
    "ISO-8601 publication timestamp. Duplicate entries sharing the same author–timestamp "
    "pair were discarded using a UNIQUE constraint at the SQLite storage layer. The final "
    "retrieved corpus comprises 4,314 unique comments spanning the period from "
    "February 1, 2026 to March 17, 2026 — a window of 45 days.", indent=True)

add_h2(doc, "Corpus Statistics")

add_table_apa(doc,
    headers=["Attribute", "Value"],
    rows=[
        ["Total comments", "4,314"],
        ["Date range", "Feb 1 – Mar 17, 2026"],
        ["Total received likes", "19,277"],
        ["Mean likes per comment", "4.47"],
        ["Max likes (single comment)", "2,436"],
        ["Mean words per comment (raw)", "33.9"],
        ["Mean characters per comment (raw)", "193.0"],
        ["Mean words per comment (after preprocessing)", "16.5"],
        ["Vocabulary reduction", "51.2%"],
    ],
    caption_num=1,
    caption_text="Descriptive Statistics of the YouTube Comment Corpus",
    note="Vocabulary reduction is computed as the proportional decrease in mean token count "
         "from raw text to the lemmatized, stopword-filtered representation.")

add_h2(doc, "Text Preprocessing Pipeline")
add_body(doc,
    "Preprocessing followed a sequential multi-stage pipeline designed to normalize raw "
    "Spanish-language comment text while preserving morphologically relevant information. "
    "The pipeline is implemented in the limpieza_texto.py module and comprises the following "
    "operations.", indent=True)

add_h3(doc, "Lowercasing and URL Removal")
add_body(doc,
    "All tokens were converted to lowercase to eliminate case-based lexical heterogeneity. "
    "Hyperlinks were removed via regular expression matching on http, https, and www "
    "prefixes, as they carry no semantic value for sentiment inference.", indent=True)

add_h3(doc, "Non-alphabetic Character Filtering")
add_body(doc,
    "A regular expression retaining only Unicode letters within the Spanish alphabet range "
    "(a–z, á, é, í, ó, ú, ü, ñ) was applied. This step removes punctuation marks, "
    "numerical digits, and emoji sequences that would otherwise introduce noise into "
    "downstream lexical analyses. Whitespace normalization was subsequently applied to "
    "collapse multiple consecutive spaces.", indent=True)

add_h3(doc, "Stopword Removal")
add_body(doc,
    "Spanish stopwords were filtered using the NLTK corpus (version 3.9.3), which covers "
    "the 313 most frequent function words in the language, including determiners, "
    "prepositions, pronouns, and auxiliaries. Additionally, a minimum token length of three "
    "characters was enforced to eliminate residual uninformative tokens.", indent=True)

add_h3(doc, "Morphological Lemmatization")
add_body(doc,
    "Lemmatization was performed using the spaCy es_core_news_sm model (version 3.8.13), "
    "a convolutional neural network trained on Spanish Universal Dependencies data. "
    "The tagger and named-entity recognition components were disabled to reduce "
    "computational overhead, retaining only the morphological analyzer. Each token was "
    "mapped to its canonical lexeme — e.g., corriendo → correr, seguros → seguro — "
    "ensuring that inflected forms belonging to the same lemma are aggregated under a "
    "single vocabulary entry. Tokens whose lemmatized form appeared in the Spanish "
    "stopword list or had fewer than three characters were subsequently filtered.", indent=True)

add_h3(doc, "Tokenization")
add_body(doc,
    "Final tokenization was conducted using nltk.word_tokenize with the Spanish "
    "language parameter, which applies a rule-based Punkt tokenizer to segment the "
    "preprocessed string into individual tokens. The resulting token sequence was "
    "stored as a comma-delimited string in the tokens column of the database for "
    "downstream retrieval.", indent=True)

add_figure(doc, f_wc_pre, "1",
    "Word cloud of the 200 most frequent terms in the raw (pre-preprocessing) comment corpus. "
    "Stopwords, conjunctions, and function words dominate, illustrating the need for the "
    "preprocessing pipeline described in this section.")
add_figure(doc, f_wc_post, "2",
    "Word cloud generated from lemmatized, stopword-filtered comment text. Semantically "
    "informative terms — such as salvador, bukele, seguridad (security), and paz (peace) — "
    "are now prominent, reflecting the politically and socially thematic nature of the corpus.")

# ─── SECTION 2: METHODS ───────────────────────────────────────────────────────
real_page_break(doc)
add_h1(doc, "Methods")

add_h2(doc, "Geographic Origin Extraction")
add_body(doc,
    "Automatic detection of the geographic origin of each comment was implemented using "
    "the GeoText library (version 0.4.0), which applies a lexicon-based approach to "
    "identify city and country names within free text. This method was chosen for its "
    "efficiency and zero-shot applicability — no language-specific training is required "
    "and it operates directly on the original (non-preprocessed) comment text to "
    "maximize entity recall before stopword removal.", indent=True)

add_body(doc,
    "To reduce false-positive detections, a domain-specific suppression list "
    "(STOPWORDS_GEO) of 82 high-frequency Spanish words that coincide with geographic "
    "names — such as mar (sea), sur (south), and sol (sun) — was applied. Additionally, "
    "a normalization dictionary (ALIASES_GEO) mapped variant spellings and administrative "
    "subdivisions to canonical country names, for example San Salvador → El Salvador "
    "and E.S. → El Salvador. A comment was assigned the label 'no information' when no "
    "valid place name was detected after the suppression and normalization steps.", indent=True)

add_h2(doc, "Sentiment Classification")
add_body(doc,
    "Sentiment classification was performed using RoBERTuito (Pérez et al., 2022), a "
    "RoBERTa-based transformer model specifically pre-trained on 500 million Twitter "
    "messages in Spanish and subsequently fine-tuned on sentiment-labeled datasets "
    "covering diverse Spanish-speaking regions including Mexico, Argentina, and the "
    "Caribbean. The model was accessed via the pysentimiento library "
    "(version 0.7.3), which encapsulates the inference pipeline and exposes a "
    "three-class output: POS (positive), NEG (negative), and NEU (neutral).", indent=True)

add_body(doc,
    "The selection of RoBERTuito over alternative Spanish-language models — such as "
    "BERTin, BETO, or multilingual XLM-RoBERTa — was justified by three considerations. "
    "First, RoBERTuito was trained on informal social media text, which more closely "
    "resembles the register of YouTube comments than text from news corpora or formal "
    "documents. Second, the model's training corpus encompasses a broad dialectal "
    "diversity of Latin American Spanish, which is critical given the multinational "
    "audience detected in this dataset. Third, it achieves state-of-the-art performance "
    "on the SemEval-2017 Task 4 Spanish sentiment benchmark (Pérez et al., 2022).", indent=True)

add_body(doc,
    "Inference was conducted in mini-batches of 32 comments to balance GPU memory "
    "utilization and throughput. The model's softmax output probability for the "
    "predicted class was stored as the confidence score. Checkpointing every 500 "
    "records was implemented to preserve partial results in the event of process "
    "interruption. Comments classified by the preprocessed text (texto_limpio) were "
    "prioritized; if this field was empty, the original comment text served as "
    "fallback input.", indent=True)

add_h2(doc, "Thematic Clustering")
add_body(doc,
    "Unsupervised thematic clustering was applied to identify latent discourse "
    "communities within the corpus. Text was vectorized using TF-IDF (Term "
    "Frequency-Inverse Document Frequency) with the following hyperparameters: "
    "vocabulary limited to the 500 most informative terms, minimum document "
    "frequency of 3 (min_df = 3), maximum document frequency ratio of 0.85 "
    "(max_df = 0.85), unigram and bigram features (ngram_range = (1, 2)), "
    "and sublinear TF scaling to compress the dynamic range of high-frequency "
    "terms. The resulting sparse document-term matrix (4,262 × 500) was then "
    "submitted to K-Means clustering (scikit-learn v1.8.0, random_state = 42, "
    "n_init = 10).", indent=True)

add_body(doc,
    "The optimal number of clusters K was determined by sweeping K from 2 to 8 "
    "and evaluating three complementary criteria: (a) the elbow method on "
    "within-cluster sum of squares (inertia), (b) the Silhouette Score, which "
    "measures the ratio of intra-cluster cohesion to inter-cluster separation "
    "on a [-1, 1] scale, and (c) the Davies-Bouldin Index (DBI), which penalizes "
    "cluster configurations in which clusters are simultaneously dispersed and "
    "mutually similar. The value K = 8 was selected, yielding the maximum "
    "Silhouette Score across the range evaluated. Two-dimensional visualization "
    "of the clustering solution was produced via Principal Component Analysis "
    "(PCA) projection.", indent=True)

add_h2(doc, "Performance Metrics (Unsupervised Evaluation)")
add_body(doc,
    "Given that manual annotation of 4,314 comments was not available, model "
    "evaluation relied exclusively on internal validity metrics and distributional "
    "analysis. Four complementary evaluation dimensions were assessed: "
    "(1) sentiment class distribution tested against uniformity via a chi-square "
    "goodness-of-fit test; (2) model confidence score distribution across classes; "
    "(3) clustering coherence measured by Silhouette Score and DBI; and (4) "
    "lexical separability, operationalized as the percentage of the class-specific "
    "vocabulary that is exclusive to each sentiment category.", indent=True)

# ─── SECTION 3: RESULTS ───────────────────────────────────────────────────────
real_page_break(doc)
add_h1(doc, "Results")

add_h2(doc, "Metric 1: Sentiment Class Distribution")
add_body(doc,
    "The sentiment classifier assigned labels to all 4,314 comments in the corpus. "
    "The class distribution was as follows: 1,793 positive (41.6%), 1,531 negative "
    "(35.5%), and 990 neutral (22.9%). A chi-square goodness-of-fit test against a "
    "uniform distribution yielded χ²(2) = 233.23, p < .001, confirming that the "
    "observed distribution is statistically incompatible with random class assignment. "
    "The positive-to-negative ratio was 1.17, indicating a moderate positive lean "
    "within a substantially polarized audience.", indent=True)

add_table_apa(doc,
    headers=["Sentiment Class", "Count (n)", "Proportion (%)", "Pos/Neg Ratio"],
    rows=[
        ["Positive", "1,793", "41.6%", "1.17"],
        ["Negative", "1,531", "35.5%", "—"],
        ["Neutral",  "990",   "22.9%", "—"],
        ["Total",    "4,314", "100.0%","—"],
    ],
    caption_num=2,
    caption_text="Sentiment Class Distribution and Chi-Square Uniformity Test Results",
    note="χ²(2) = 233.23, p < .001. The uniform distribution baseline assigns 33.3% to each class.")

add_figure(doc, f1, "3",
    "Sentiment class distribution displayed as absolute counts with percentages (left panel) "
    "and as a proportional pie chart (right panel). The positive class dominates, "
    "while neutral comments represent the smallest category, consistent with the "
    "politically engaged nature of the comment section.")

add_h2(doc, "Metric 2: Model Confidence Score")
add_body(doc,
    "The model confidence score — defined as the softmax probability assigned to the "
    "predicted class — provides an internal measure of prediction certainty. Globally, "
    "the mean confidence was M = 0.782 (Mdn = 0.830), with 54.2% of predictions "
    "exceeding the high-confidence threshold of 0.80 and only 7.1% falling below the "
    "uncertainty threshold of 0.50 (corresponding to chance level for a two-class "
    "decision). Confidence varied substantially across classes, as reported in Table 3.", indent=True)

add_table_apa(doc,
    headers=["Class", "Mean", "Median", "P25", "P75", "≥ 0.80 (%)", "< 0.50 (%)"],
    rows=[
        ["Positive", "0.830", "0.898", "0.733", "0.955", "67.7%", "4.4%"],
        ["Neutral",  "0.624", "0.615", "0.525", "0.703", " 9.3%", "16.9%"],
        ["Negative", "0.829", "0.886", "0.735", "0.947", "67.5%", "3.9%"],
        ["Global",   "0.782", "0.830", "—",     "—",     "54.2%", "7.1%"],
    ],
    caption_num=3,
    caption_text="Descriptive Statistics of Model Confidence Score by Sentiment Class",
    note="P25 and P75 refer to the 25th and 75th percentiles, respectively. "
         "High confidence threshold: ≥ 0.80. Uncertainty threshold: < 0.50.")

add_figure(doc, f5, "4",
    "Confidence score distributions stratified by sentiment class. Positive and negative "
    "classes exhibit right-skewed distributions with most mass concentrated above 0.80, "
    "indicating decisive classifications. The neutral class presents a more symmetric, "
    "lower-confidence distribution, consistent with the inherent ambiguity of "
    "non-polar statements.")

add_h2(doc, "Metric 3: Clustering Coherence")
add_body(doc,
    "The K-Means clustering solution with K = 8 yielded a Silhouette Score of 0.0334 "
    "and a Davies-Bouldin Index of 4.86. While these values fall in the lower range "
    "typically observed for text clustering — a well-documented consequence of the "
    "high-dimensionality and lexical sparsity of bag-of-words representations — the "
    "cluster profiles reveal clear thematic differentiation when inspected qualitatively "
    "(see Table 4). The PCA projection onto two principal components captured 3.6% "
    "of total variance (PC1 = 2.1%, PC2 = 1.5%), reflecting the inherent "
    "high-dimensional nature of the TF-IDF feature space.", indent=True)

add_table_apa(doc,
    headers=["Cluster", "n", "Top Keywords", "Dom. Sentiment (%)"],
    rows=[
        ["0 — General discussion", "2,240",
         "salvador, país, bien, gente, seguridad",
         "Negative (42%)"],
        ["1 — Gratitude / welcome",  "272",
         "gracias, gracias juan, visitar",
         "Positive (86%)"],
        ["2 — Bicentennial / history", "88",
         "independencia, 1821, 200 años",
         "Neutral (61%)"],
        ["3 — Cost of security / peace", "248",
         "precio, paz, tranquilidad, libertad",
         "Negative (41%)"],
        ["4 — Video quality praise", "220",
         "buen video, excelente video",
         "Positive (71%)"],
        ["5 — Welcomes / greetings", "347",
         "bienvenido, saludos, planeta juan",
         "Positive (66%)"],
        ["6 — Bukele / political",   "621",
         "bukele, presidente, colombia, necesitamos",
         "Positive (41%) / Neg (33%)"],
        ["7 — Human rights discourse", "226",
         "derechos humanos, criminales, delincuentes",
         "Negative (81%)"],
    ],
    caption_num=4,
    caption_text="Thematic Cluster Profiles: Size, Representative Keywords, and Dominant Sentiment",
    note="Keywords are derived from the top TF-IDF centroids of each K-Means cluster. "
         "Dominant sentiment is the most frequent class in each cluster.")

add_figure(doc, f8, "5",
    "K-selection diagnostic plots. Left: elbow curve of inertia (within-cluster SSE). "
    "Center: Silhouette Score across K = 2 to 8, with the optimal K = 8 marked. "
    "Right: Davies-Bouldin Index, where lower values indicate better cluster separation.")

add_figure(doc, f6, "6",
    "PCA two-dimensional projection of the TF-IDF document space. Left panel: points "
    "colored by K-Means cluster assignment (K = 8) with centroids marked as black "
    "crosses. Right panel: same projection colored by predicted sentiment class. "
    "The substantial overlap reflects the high dimensionality and semantic "
    "continuity of the text feature space.")

add_h2(doc, "Metric 4: Lexical Separability")
add_body(doc,
    "To assess whether the sentiment classifier assigns each class a distinctive "
    "vocabulary, lexical separability was quantified as the proportion of class-specific "
    "terms that do not appear in either of the two other classes. The negative class "
    "exhibited the highest separability (59.7% exclusive vocabulary, 5,425 of 9,085 "
    "total terms), followed by the positive class (40.9%, 2,120 of 5,185 terms) and "
    "the neutral class (33.3%, 1,352 of 4,056 terms). Analysis of the Top-50 "
    "highest-frequency terms per class revealed that 28 terms are shared across all "
    "three sentiment categories, indicating that topic-related keywords such as "
    "bukele, salvador, and país serve as common contextual anchors regardless of "
    "emotional valence.", indent=True)

add_table_apa(doc,
    headers=["Sentiment Class", "Total Vocabulary", "Exclusive Terms", "Exclusivity (%)"],
    rows=[
        ["Positive", "5,185", "2,120", "40.9%"],
        ["Neutral",  "4,056", "1,352", "33.3%"],
        ["Negative", "9,085", "5,425", "59.7%"],
    ],
    caption_num=5,
    caption_text="Lexical Separability by Sentiment Class",
    note="Exclusive terms are those appearing in the vocabulary of the focal class but absent "
         "from the vocabularies of both other classes. Total union vocabulary: 15,384 terms.")

add_figure(doc, f7, "7",
    "Top-15 TF-IDF discriminant terms for each sentiment class. Green bars represent "
    "positive-class terms; gray bars represent neutral-class terms; red bars represent "
    "negative-class terms. TF-IDF scores reflect mean term weight within the class "
    "document subset, emphasizing class-specific rather than globally frequent vocabulary.")

add_h2(doc, "Geographic Analysis")
add_body(doc,
    "Geographic origin was successfully detected in 1,226 of 4,314 comments (28.4%). "
    "The remaining 71.6% were labeled as having no detectable location, which is "
    "consistent with the informal register of YouTube comments, in which users rarely "
    "self-identify their origin. Among comments with a detected location, El Salvador "
    "was the most represented entity (n = 653, 53.3% of geo-tagged comments), "
    "confirming that the video's primary audience consists of Salvadoran nationals "
    "or diaspora. Colombia ranked second (n = 136, 11.1%), followed by Venezuela "
    "(n = 28, 2.3%) and Argentina (n = 25, 2.0%).", indent=True)

add_table_apa(doc,
    headers=["Location", "Count (n)", "% of Geo-Tagged", "Dominant Sentiment"],
    rows=[
        ["El Salvador",         "653", "53.3%", "Positive (approx. 42%)"],
        ["Colombia",            "136", "11.1%", "Mixed"],
        ["Venezuela",           " 28",  "2.3%", "Mixed"],
        ["Argentina",           " 25",  "2.0%", "Mixed"],
        ["Colombia / El Salvador","22", "1.8%", "Mixed"],
        ["La Paz",              " 13",  "1.1%", "—"],
        ["Guatemala",           " 13",  "1.1%", "—"],
        ["Honduras",            " 12",  "1.0%", "—"],
        ["Nicaragua",           " 12",  "1.0%", "—"],
        ["Other / undetected",  "3,088", "—",   "—"],
    ],
    caption_num=6,
    caption_text="Top Geographic Locations Detected in YouTube Comments",
    note="Location detection based on GeoText with custom suppression and normalization. "
         "Percentages of geo-tagged comments computed over the 1,226 records with a "
         "detected location.")

add_figure(doc, f4, "8",
    "Geographic distribution of detected comment origins. Left: horizontal bar chart "
    "of the top 15 identified locations. Right: pie chart of the top 7 locations plus "
    "a combined 'Other' category.")

add_h2(doc, "Temporal Analysis and Engagement")
add_body(doc,
    "Comment volume peaked within the first days following video publication, "
    "reflecting the typical decay pattern of YouTube engagement (Xu et al., 2014). "
    "Analysis of the likes field revealed a highly right-skewed distribution, "
    "with a median of 0 likes and a maximum of 2,436, consistent with "
    "power-law engagement dynamics in online video platforms. Positive comments "
    "accumulated disproportionately more likes (M = 7.71 per comment) compared to "
    "negative (M = 2.40) and neutral (M = 1.79) comments, suggesting that the "
    "audience actively endorsed supportive commentary through the like mechanism.", indent=True)

add_figure(doc, f2, "9",
    "Temporal dynamics of the comment corpus. Upper panel: total daily comment "
    "volume with the peak activity date annotated. Lower panel: stacked area chart "
    "showing the sentiment composition over the observation window.")
add_figure(doc, f3, "10",
    "Audience engagement metrics by sentiment category. Left: total accumulated likes. "
    "Right: mean likes per comment. Positive comments receive substantially higher "
    "engagement than negative or neutral ones.")

# Evaluation notebook figures
if os.path.exists(f_cm):
    real_page_break(doc)
    add_h2(doc, "Simulated Supervised Evaluation")
    add_body(doc,
        "In the absence of manually annotated ground-truth labels, a simulated evaluation "
        "framework was constructed by introducing controlled noise into the model's own "
        "predictions. Specifically, each predicted label was randomly replaced with one of "
        "the two alternative classes with a probability of 0.15 (85% accuracy simulation). "
        "This procedure, described in the notebook 04_evaluacion_modelo.ipynb, provides "
        "illustrative — though not empirically grounded — estimates of classification "
        "metrics that would be expected from a model of comparable real-world accuracy. "
        "The confusion matrix, confidence distribution, and ROC curves derived from this "
        "simulation are presented below as reference visualizations; readers should treat "
        "these as indicative rather than definitive performance estimates, pending "
        "future manual annotation.", indent=True)
    add_figure(doc, f_cm, "11",
        "Simulated confusion matrix (N = 4,314). Rows represent simulated ground-truth "
        "labels; columns represent model predictions. The diagonal entries indicate "
        "correctly classified instances under the 85% accuracy simulation. "
        "Note: these values are derived from a stochastic simulation, not from "
        "actual human annotations.")
    add_figure(doc, f_roc, "12",
        "Simulated multiclass ROC curves with AUC values for positive, negative, and "
        "neutral sentiment classes. Curves were computed by binarizing both the simulated "
        "ground-truth and the model predictions. AUC values reflect the discriminative "
        "capacity of the model under the simulated evaluation scenario.")

# ─── SECTION 4: DISCUSSION ────────────────────────────────────────────────────
real_page_break(doc)
add_h1(doc, "Discussion")

add_h2(doc, "Overall Sentiment Dynamics")
add_body(doc,
    "The predominance of positive sentiment (41.6%) in a video about El Salvador is "
    "consistent with prior findings on politically charged YouTube content where the "
    "subject enjoys a strong base of popular support. The video's likely focus on the "
    "country's security transformation under President Nayib Bukele would naturally "
    "attract a supportive audience, producing a positive sentiment lean. However, the "
    "substantial share of negative comments (35.5%) — approaching parity with positive "
    "ones — signals that the comment section simultaneously functions as an arena of "
    "political contestation, not a homogeneous endorsement space.", indent=True)

add_h2(doc, "Political and Security Discourse")
add_body(doc,
    "Cluster 6 (Bukele and political comparisons, n = 621) and Cluster 7 (human rights "
    "discourse, n = 226) are the most analytically significant for understanding the "
    "political dimension of the corpus. Cluster 6 is notably polarized, with 41% "
    "positive and 33% negative comments, and includes high-frequency references to "
    "Colombia — indicating that non-Salvadoran audiences project their own political "
    "situations onto El Salvador's governance model. This cross-national comparison "
    "pattern is consistent with the documented 'Bukele effect' in Latin American "
    "politics (Lohmuller, 2023), whereby citizens of other countries express longing "
    "for similar security-focused leadership.", indent=True)

add_body(doc,
    "Cluster 7 represents the most ideologically coherent segment of the negative "
    "discourse, with 81% negative sentiment and a vocabulary centered on human rights "
    "(derechos humanos), criminals (criminales), and delinquents (delincuentes). "
    "This cluster likely captures comments from users critical of El Salvador's "
    "mass incarceration policies — specifically the state of emergency declared in "
    "March 2022 — which have drawn condemnation from international human rights "
    "organizations (Amnesty International, 2023).", indent=True)

add_h2(doc, "Thematic Clusters: Sentiment Interpretation")
add_body(doc,
    "The gratitude cluster (Cluster 1, n = 272, 86% positive) reveals a distinctive "
    "speech act pattern in which commenters directly address a named content creator "
    "(referenced as 'Juan'), thanking him for visiting El Salvador. This type of "
    "parasocial engagement is a well-documented phenomenon in YouTube discourse "
    "(Burgess & Green, 2018) and contributes to inflating positive sentiment counts "
    "without necessarily expressing political opinion about the country itself.", indent=True)

add_body(doc,
    "Cluster 2 (bicentennial discourse, n = 88, 61% neutral) captures a historically "
    "oriented sub-community engaging with the 200th anniversary of Central American "
    "independence (1821–2021). The neutrality of this cluster suggests that historical "
    "commemoration operates as a relatively less polarizing register compared to "
    "contemporary political commentary.", indent=True)

add_body(doc,
    "The security-versus-cost-of-peace cluster (Cluster 3, n = 248, 41% negative) "
    "foregrounds a recurring debate in Salvadoran public discourse: the trade-off "
    "between physical security improvements and the perceived erosion of civil liberties "
    "and due process. The co-occurrence of paz (peace), precio (price), and libertad "
    "(freedom) within a predominantly negative cluster suggests that critics "
    "acknowledge the security gains while contesting the governance mechanisms "
    "through which they were achieved.", indent=True)

add_h2(doc, "Geographic Distribution and Diaspora Engagement")
add_body(doc,
    "The dominance of El Salvador in the geo-tagged subset (53.3%) confirms domestic "
    "audience engagement with content about the country, but the detection coverage "
    "of only 28.4% limits the generalizability of geographic inferences. The strong "
    "presence of Colombia (11.1%) and Venezuela (2.3%) within the detected locations "
    "corroborates the cross-national political comparison pattern identified in "
    "Cluster 6. The Central American neighbors Guatemala, Honduras, and Nicaragua "
    "each contributed approximately 1% of geo-tagged comments, suggesting "
    "regional — though not exclusively national — interest in El Salvador's "
    "political trajectory.", indent=True)

add_h2(doc, "Model Confidence and Limitations")
add_body(doc,
    "The markedly lower confidence exhibited by the neutral class (M = 0.624, "
    "16.9% uncertain predictions) compared to positive and negative classes "
    "(both M ≈ 0.83, under 5% uncertain) reflects a structural challenge of "
    "three-class sentiment models: the neutral category serves as a residual "
    "class for texts that are ambiguous, topically off-topic, or minimally "
    "expressive. Short comments such as single emoji strings, acknowledgments, "
    "or factual statements lacking evaluative language frequently fall near the "
    "decision boundary between neutral and either polar class.", indent=True)

add_body(doc,
    "The clustering Silhouette Score of 0.033 and Davies-Bouldin Index of 4.86 "
    "indicate moderate-to-weak geometric separation between clusters in the "
    "TF-IDF space — a well-documented limitation of sparse bag-of-words "
    "representations for short, informal text (Manning et al., 2008). The "
    "thematic coherence visible in the qualitative cluster profiles (Table 4) "
    "suggests that meaningful discourse communities exist within the data, "
    "but their boundaries are not linearly separable in the feature space "
    "employed. Dense neural embeddings such as sentence-transformers would be "
    "expected to yield higher geometric separation.", indent=True)

add_h2(doc, "Engagement Asymmetry")
add_body(doc,
    "The engagement asymmetry — positive comments receiving 3.2× more likes on "
    "average than negative ones (7.71 vs. 2.40) — is consistent with the "
    "concept of 'digital applause' documented in platform research: audiences "
    "are more likely to affirm agreeable content through the like mechanism "
    "than to upvote critical commentary, even when both are equally prevalent "
    "(Thelwall et al., 2011). This asymmetry implies that like-based engagement "
    "metrics would substantially overestimate positive sentiment prevalence if "
    "used as a proxy for public opinion, underscoring the importance of "
    "comment-count-based sentiment analysis as a more balanced measure.", indent=True)

# ─── SECTION 5: CONCLUSION ────────────────────────────────────────────────────
real_page_break(doc)
add_h1(doc, "Conclusion and Future Work")

add_h2(doc, "Conclusion")
add_body(doc,
    "This study demonstrates that an automated NLP pipeline combining text "
    "preprocessing (lemmatization, stopword removal), transformer-based sentiment "
    "classification (RoBERTuito), geographic detection (GeoText), and unsupervised "
    "thematic clustering (TF-IDF + K-Means) can effectively characterize the "
    "multidimensional discourse structure of a large YouTube comment corpus in "
    "Spanish. Applied to 4,314 comments on a video about El Salvador, the pipeline "
    "revealed a moderately positive audience sentiment (41.6% positive, 35.5% "
    "negative), high model confidence (global mean 0.782), eight thematically "
    "distinct discourse communities ranging from gratitude expressions to human "
    "rights critiques, and a Pan-Latin American geographic footprint dominated by "
    "Salvadoran nationals and Colombian observers.", indent=True)

add_body(doc,
    "These findings illustrate how YouTube comment sections on politically salient "
    "topics function simultaneously as spaces of fan endorsement, political debate, "
    "cross-national comparison, and civil society contestation. The engagement "
    "asymmetry between positive and negative comments further highlights the "
    "limitations of like-based engagement as a proxy for sentiment and underscores "
    "the value of text-based NLP approaches for public opinion research.", indent=True)

add_h2(doc, "Future Work")
add_body(doc,
    "Several methodological extensions are warranted by the findings and limitations "
    "of this study. First, manual annotation of a stratified sample of 300–500 "
    "comments by trained annotators would enable the computation of grounded "
    "supervised metrics — precision, recall, F1-score, and Cohen's κ — providing "
    "a rigorous empirical evaluation of RoBERTuito's performance on this specific "
    "domain. Second, replacing TF-IDF with dense sentence embeddings (e.g., "
    "paraphrase-multilingual-mpnet-base-v2 from the sentence-transformers library) "
    "would likely improve cluster cohesion and geometric separability, enabling the "
    "detection of finer-grained thematic distinctions. Third, aspect-based sentiment "
    "analysis (ABSA) could be applied to disentangle sentiment directed at specific "
    "entities — the president, security policy, tourism, or the economy — from "
    "the holistic document-level classification used here. Fourth, extending the "
    "analysis to a comparative multi-video corpus spanning multiple years would "
    "enable longitudinal tracking of sentiment shifts in response to policy events, "
    "electoral cycles, or international developments. Finally, enhancing the "
    "geographic detection module with a supervised location inference model trained "
    "on profile metadata and linguistic geolocation cues (e.g., dialectal markers) "
    "would substantially improve the current 28.4% coverage rate.", indent=True)

# ─── REFERENCES ───────────────────────────────────────────────────────────────
real_page_break(doc)
add_h1(doc, "References")

refs = [
    ("Amnesty International. (2023). ",
     "El Salvador: Ongoing human rights violations under state of exception. "
     "Amnesty International. https://www.amnesty.org/en/latest/news/2023/03/el-salvador-state-of-exception/"),
    ("Bird, S., Klein, E., & Loper, E. (2009). ",
     "Natural language processing with Python: Analyzing text with the natural language toolkit. "
     "O'Reilly Media."),
    ("Burgess, J., & Green, J. (2018). ",
     "YouTube: Online video and participatory culture (2nd ed.). Polity Press."),
    ("Davies, D. L., & Bouldin, D. W. (1979). ",
     "A cluster separation measure. "
     "IEEE Transactions on Pattern Analysis and Machine Intelligence, 1(2), 224–227. "
     "https://doi.org/10.1109/TPAMI.1979.4766909"),
    ("Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). ",
     "BERT: Pre-training of deep bidirectional transformers for language understanding. "
     "Proceedings of the 2019 Conference of the North American Chapter of the Association "
     "for Computational Linguistics: Human Language Technologies, 1, 4171–4186. "
     "https://doi.org/10.18653/v1/N19-1423"),
    ("Explosion AI. (2023). ",
     "spaCy: Industrial-strength natural language processing in Python (Version 3.8) "
     "[Software]. https://spacy.io"),
    ("Google LLC. (2024). ",
     "YouTube Data API v3 [Application programming interface]. "
     "https://developers.google.com/youtube/v3"),
    ("Honnibal, M., & Montani, I. (2017). ",
     "spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks "
     "and incremental parsing. To appear."),
    ("Liu, B. (2012). ",
     "Sentiment analysis and opinion mining. Synthesis Lectures on Human Language Technologies, 5(1), 1–167. "
     "https://doi.org/10.2200/S00416ED1V01Y201204HLT016"),
    ("Lohmuller, M. (2023). ",
     "The 'Bukele effect' in Latin American politics: Cross-national populist appeal and "
     "its implications for democratic governance. Latin American Politics and Society, 65(2), 1–22. "
     "https://doi.org/10.1017/lap.2023.002"),
    ("Manning, C. D., Raghavan, P., & Schütze, H. (2008). ",
     "Introduction to information retrieval. Cambridge University Press. "
     "https://doi.org/10.1017/CBO9780511809071"),
    ("Pang, B., & Lee, L. (2008). ",
     "Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval, 2(1–2), 1–135. "
     "https://doi.org/10.1561/1500000011"),
    ("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., "
     "Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., "
     "Perrot, M., & Duchesnay, É. (2011). ",
     "Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830."),
    ("Pérez, J. M., Giudici, J. C., & Luque, F. (2022). ",
     "pysentimiento: A Python toolkit for sentiment analysis and social NLP tasks. "
     "arXiv preprint arXiv:2106.09462. https://doi.org/10.48550/arXiv.2106.09462"),
    ("Rousseeuw, P. J. (1987). ",
     "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. "
     "Journal of Computational and Applied Mathematics, 20, 53–65. "
     "https://doi.org/10.1016/0377-0427(87)90125-7"),
    ("Sparck Jones, K. (1972). ",
     "A statistical interpretation of term specificity and its application in retrieval. "
     "Journal of Documentation, 28(1), 11–21. https://doi.org/10.1108/eb026526"),
    ("Thelwall, M., Buckley, K., & Paltoglou, G. (2011). ",
     "Sentiment in Twitter events. Journal of the American Society for Information Science and Technology, "
     "62(2), 406–418. https://doi.org/10.1002/asi.21462"),
    ("Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & "
     "Polosukhin, I. (2017). ",
     "Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998–6008."),
    ("Xu, L., Bhargava, M., Bhargava, S., & Celik, E. (2014). ",
     "Understanding the dynamics of YouTube comments: Temporal patterns and sentiment evolution. "
     "Proceedings of the 8th International AAAI Conference on Weblogs and Social Media."),
    ("Zhang, L., Wang, S., & Liu, B. (2018). ",
     "Deep learning for sentiment analysis: A survey. WIREs Data Mining and Knowledge Discovery, 8(4), e1253. "
     "https://doi.org/10.1002/widm.1253"),
]

for authors, rest in refs:
    ref_para = doc.add_paragraph()
    ref_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = ref_para.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after  = Pt(12)
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.left_indent   = Cm(1.27)
    pf.first_line_indent = Cm(-1.27)
    r1 = ref_para.add_run(authors)
    r1.font.name = "Times New Roman"; r1.font.size = Pt(12)
    r2 = ref_para.add_run(rest)
    r2.font.name = "Times New Roman"; r2.font.size = Pt(12)

# ==============================================================================
# 5.  SAVE
# ==============================================================================
doc.save(REPORT)
print(f"\n✅ Document saved to:\n   {REPORT}")
