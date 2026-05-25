"""
Genera la guía completa en PDF:
'Cómo subir el proyecto de Análisis de Sentimientos a GitHub como portafolio'
"""

import os, re, base64, textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak,
    Table, TableStyle, Preformatted, KeepTogether, Image
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")
PDF_PATH    = os.path.join(OUTPUT_DIR, "Guia_GitHub_Portafolio.pdf")

# ── Colores ────────────────────────────────────────────────────────────────
C_BG       = colors.HexColor("#0b0f19")
C_CARD     = colors.HexColor("#151b2b")
C_CYAN     = colors.HexColor("#00f2fe")
C_PINK     = colors.HexColor("#fe0979")
C_GRAY     = colors.HexColor("#a8b0c3")
C_WHITE    = colors.white
C_DARK     = colors.HexColor("#1e2640")
C_STEP_BG  = colors.HexColor("#0d1829")
C_CODE_BG  = colors.HexColor("#111827")
C_YELLOW   = colors.HexColor("#ffd700")
C_GREEN    = colors.HexColor("#22c55e")

W, H = A4  # 595 x 842 pts

# ── Estilos ────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

sTitle = S("sTitle",
    fontName="Helvetica-Bold", fontSize=28, textColor=C_WHITE,
    leading=34, alignment=TA_CENTER, spaceAfter=6)

sSubtitle = S("sSubtitle",
    fontName="Helvetica", fontSize=13, textColor=C_CYAN,
    leading=18, alignment=TA_CENTER, spaceAfter=4)

sMeta = S("sMeta",
    fontName="Helvetica", fontSize=10, textColor=C_GRAY,
    leading=14, alignment=TA_CENTER, spaceAfter=20)

sH1 = S("sH1",
    fontName="Helvetica-Bold", fontSize=18, textColor=C_CYAN,
    leading=22, spaceBefore=18, spaceAfter=8)

sH2 = S("sH2",
    fontName="Helvetica-Bold", fontSize=14, textColor=C_WHITE,
    leading=18, spaceBefore=14, spaceAfter=6)

sH3 = S("sH3",
    fontName="Helvetica-Bold", fontSize=11, textColor=C_YELLOW,
    leading=15, spaceBefore=10, spaceAfter=4)

sBody = S("sBody",
    fontName="Helvetica", fontSize=10, textColor=C_GRAY,
    leading=16, alignment=TA_JUSTIFY, spaceAfter=6)

sBodyW = S("sBodyW",
    fontName="Helvetica", fontSize=10, textColor=C_WHITE,
    leading=16, alignment=TA_JUSTIFY, spaceAfter=6)

sBullet = S("sBullet",
    fontName="Helvetica", fontSize=10, textColor=C_GRAY,
    leading=15, leftIndent=16, spaceAfter=3,
    bulletIndent=4, bulletText="•")

sBulletCyan = S("sBulletCyan",
    fontName="Helvetica", fontSize=10, textColor=C_WHITE,
    leading=15, leftIndent=16, spaceAfter=3,
    bulletIndent=4, bulletText="▸")

sCode = S("sCode",
    fontName="Courier", fontSize=9, textColor=C_GREEN,
    leading=13, leftIndent=12, spaceAfter=2,
    backColor=C_CODE_BG)

sCodeComment = S("sCodeComment",
    fontName="Courier", fontSize=9, textColor=C_GRAY,
    leading=13, leftIndent=12, spaceAfter=2,
    backColor=C_CODE_BG)

sNote = S("sNote",
    fontName="Helvetica-Oblique", fontSize=9, textColor=C_YELLOW,
    leading=13, spaceAfter=4)

sCaption = S("sCaption",
    fontName="Helvetica-Oblique", fontSize=9, textColor=C_GRAY,
    leading=12, alignment=TA_CENTER, spaceAfter=8)

sCentered = S("sCentered",
    fontName="Helvetica", fontSize=10, textColor=C_GRAY,
    leading=14, alignment=TA_CENTER, spaceAfter=4)

sStepNum = S("sStepNum",
    fontName="Helvetica-Bold", fontSize=32, textColor=C_CYAN,
    leading=38, alignment=TA_CENTER)

sStepTitle = S("sStepTitle",
    fontName="Helvetica-Bold", fontSize=16, textColor=C_WHITE,
    leading=20, spaceBefore=0, spaceAfter=6)

# ── Helpers ────────────────────────────────────────────────────────────────
def hr(color=C_CYAN, thickness=1):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=8, spaceBefore=4)

def space(h=8):
    return Spacer(1, h)

def p(text, style=sBody):
    return Paragraph(text, style)

def h1(text): return p(f"<b>{text}</b>", sH1)
def h2(text): return p(text, sH2)
def h3(text): return p(text, sH3)

def step_header(n, title, subtitle=""):
    rows = [[
        Paragraph(str(n), sStepNum),
        [Paragraph(title, sStepTitle),
         Paragraph(subtitle, sBody) if subtitle else Spacer(1, 0)]
    ]]
    t = Table(rows, colWidths=[2.2*cm, 14*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_STEP_BG),
        ("ROUNDEDCORNERS", [8]),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING", (0,0), (0,-1), 12),
        ("LEFTPADDING", (1,0), (1,-1), 8),
        ("BOX",         (0,0), (-1,-1), 1.5, C_CYAN),
    ]))
    return t

def code_block(lines, title=""):
    items = []
    if title:
        items.append(p(f"<font color='#ffd700'>▶ {title}</font>", sH3))
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            items.append(p(line, sCodeComment))
        else:
            items.append(p(line if line.strip() else " ", sCode))
    bg_table = Table([[items]], colWidths=[16.2*cm])
    bg_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_CODE_BG),
        ("BOX",          (0,0), (-1,-1), 0.5, C_CYAN),
        ("ROUNDEDCORNERS", [6]),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
    ]))
    return bg_table

def info_box(text, color=C_CYAN):
    t = Table([[p(f"<b>ℹ</b>  {text}", sBodyW)]],
              colWidths=[16.2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_DARK),
        ("BOX",           (0,0), (-1,-1), 1.5, color),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return t

def warning_box(text):
    return info_box(f"⚠️  {text}", C_PINK)

def tip_box(text):
    return info_box(f"💡  {text}", C_YELLOW)

def kv_table(rows, col1=5*cm, col2=11.2*cm):
    data = []
    for k, v in rows:
        data.append([
            p(f"<b>{k}</b>", sBodyW),
            p(v, sBody)
        ])
    t = Table(data, colWidths=[col1, col2])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), C_STEP_BG),
        ("BACKGROUND",    (1,0), (1,-1), C_CARD),
        ("GRID",          (0,0), (-1,-1), 0.3, C_GRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t

def pipeline_table():
    headers = ["Paso", "Script", "Descripción", "Entrada → Salida"]
    rows = [
        ["1 · Descarga",   "descarga_masiva.py",    "YouTube Data API v3 → 4,314 comentarios", "API → original.db"],
        ["2 · Limpieza",   "limpieza_texto.py",     "Minúsculas, URLs, stopwords, lematización (spaCy)", "original.db → base_limpia.db"],
        ["3 · Geografía",  "extract_locations.py",  "GeoText detecta países/ciudades en el texto", "base_limpia.db (col. ubicacion)"],
        ["4 · Tokenización","tokenizacion.py",       "NLTK word_tokenize para español",          "base_limpia.db (col. tokens)"],
        ["5 · Sentimiento", "clasificador_sentimientos.py","RoBERTuito · lotes de 32 comentarios","base_limpia.db (col. sentimiento, probabilidad)"],
        ["6 · Dashboard",  "exportar_dashboard_html.py","Plotly + Bootstrap → HTML interactivo","base_limpia.db → dashboard.html"],
    ]
    all_rows = [headers] + rows
    col_w = [3.2*cm, 4.5*cm, 5.5*cm, 3.8*cm]
    t = Table(all_rows, colWidths=col_w, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_CYAN),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_BG),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), C_WHITE),
        ("BACKGROUND",    (0,1), (-1,-1), C_CARD),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_CARD, C_STEP_BG]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_GRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("ALIGN",         (0,0), (-1,0),  "CENTER"),
    ])
    t.setStyle(style)
    return t

# ── Background canvas ───────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(C_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Top accent bar
    canvas.setFillColor(C_CYAN)
    canvas.rect(0, H-3, W, 3, fill=1, stroke=0)
    # Bottom accent bar
    canvas.setFillColor(C_PINK)
    canvas.rect(0, 0, W, 3, fill=1, stroke=0)
    # Page number
    if doc.page > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_GRAY)
        canvas.drawCentredString(W/2, 18,
            f"Guía GitHub Portafolio · Análisis de Sentimientos YouTube · Pág. {doc.page}")
    canvas.restoreState()

# ═══════════════════════════════════════════════════════════════════════════
# CONTENIDO
# ═══════════════════════════════════════════════════════════════════════════
story = []

# ─────────────────────────────────────────────────────────────────────────
# PORTADA
# ─────────────────────────────────────────────────────────────────────────
story += [
    space(80),
    p("GUÍA COMPLETA", S("", fontName="Helvetica-Bold", fontSize=11,
                          textColor=C_CYAN, alignment=TA_CENTER, spaceAfter=8)),
    p("Cómo Publicar tu Proyecto de", sTitle),
    p("Análisis de Sentimientos en YouTube", sTitle),
    p("como Portafolio Profesional en GitHub", sTitle),
    space(20),
    hr(C_CYAN, 2),
    space(10),
    p("Pipeline NLP completo: Extracción · Transformación · Análisis · Dashboard interactivo", sSubtitle),
    space(8),
    p("RoBERTuito · spaCy · NLTK · scikit-learn · Plotly · Python 3.13", sMeta),
    space(60),
]

# Cuadro de datos del proyecto
cover_data = [
    ["📹 Video analizado",  '"Esto es lo que nadie te cuenta de El Salvador"'],
    ["💬 Corpus",           "4,314 comentarios clasificados al 100%"],
    ["🧠 Modelo NLP",       "RoBERTuito (pysentimiento) · Transformer bidireccional"],
    ["📊 Sentimiento dom.", "Positivo · 41.6% · ratio pos/neg: 1.17x"],
    ["🌍 Alcance",          "Más de 20 países · 1,226 comentarios geolocalizados"],
    ["⚙️  Lenguaje",         "Python 3.13 · entorno virtual (venv)"],
]
story += [
    kv_table(cover_data, col1=4.5*cm, col2=12*cm),
    space(40),
    hr(C_PINK, 1),
    space(8),
    p("Fundación Universitaria Cafam — Unicafam · Tecnología en Análisis de Datos · 2025",
      sMeta),
    PageBreak(),
]

# ─────────────────────────────────────────────────────────────────────────
# TABLA DE CONTENIDOS
# ─────────────────────────────────────────────────────────────────────────
story += [
    h1("Tabla de Contenidos"),
    hr(),
    space(4),
]
toc_items = [
    ("1", "Visión General del Proyecto",          "Qué construiste y por qué importa"),
    ("2", "Arquitectura del Pipeline NLP",         "Los 6 pasos del proceso completo"),
    ("3", "Preparar el repositorio local",         "Estructura de carpetas y .gitignore"),
    ("4", "Crear la cuenta de GitHub",             "Registro y configuración inicial"),
    ("5", "Configurar Git en tu computador",       "user.name, user.email, autenticación"),
    ("6", "Crear el repositorio en GitHub",        "Público, nombre profesional, licencia"),
    ("7", "Inicializar Git e Ignoring de archivos","git init, .gitignore, primer commit"),
    ("8", "Subir el código al repositorio",        "git add → commit → push"),
    ("9", "Publicar el Dashboard con GitHub Pages","URL pública del dashboard en 3 pasos"),
    ("10","Escribir un README profesional",        "La tarjeta de presentación del proyecto"),
    ("11","Verificar y mantener el repositorio",   "Buenas prácticas de mantenimiento"),
    ("12","Presentación en clase y portafolio",    "Cómo mostrar y explicar el proyecto"),
]
toc_data = [[p(f"<b>{n}.</b>", sBodyW), p(title, sBodyW), p(desc, sBody)]
            for n, title, desc in toc_items]
toc_t = Table(toc_data, colWidths=[0.8*cm, 7.5*cm, 8*cm])
toc_t.setStyle(TableStyle([
    ("FONTSIZE",      (0,0), (-1,-1), 10),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_CARD, C_STEP_BG]),
    ("GRID",          (0,0), (-1,-1), 0.2, C_GRAY),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
]))
story += [toc_t, PageBreak()]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — VISIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(1, "Visión General del Proyecto",
                "Qué construiste y por qué tiene valor como portafolio"),
    space(10),
    h2("¿Qué es este proyecto?"),
    p("Este proyecto implementa un <b>pipeline completo de Procesamiento de Lenguaje Natural (NLP)</b> "
      "para analizar la percepción pública expresada en los comentarios del video de YouTube "
      "<i>'Esto es lo que nadie te cuenta de El Salvador'</i>. El objetivo fue clasificar "
      "automáticamente 4,314 comentarios en tres categorías de sentimiento: positivo, negativo "
      "y neutro, utilizando el modelo de inteligencia artificial <b>RoBERTuito</b> (basado en "
      "la arquitectura Transformer de Meta/Facebook, entrenado sobre 500 millones de tweets "
      "en español latinoamericano).", sBody),
    space(6),
    h2("¿Por qué es valioso para tu portafolio?"),
]

value_rows = [
    ["✅ Pipeline end-to-end",
     "Cubre el ciclo completo de un proyecto de datos: recolección → limpieza → análisis → visualización. "
     "Demuestra que sabes trabajar con todas las capas de la cadena de valor de datos."],
    ["✅ API Real",
     "Usaste la YouTube Data API v3 para extraer datos reales, no un dataset predefinido. "
     "Esto muestra iniciativa y habilidad para trabajar con servicios externos."],
    ["✅ NLP avanzado",
     "RoBERTuito es un modelo Transformer de última generación. "
     "Saber integrarlo con pysentimiento es una habilidad muy demandada en el mercado."],
    ["✅ Dashboard interactivo",
     "El resultado final es un dashboard HTML profesional con Plotly y Bootstrap, "
     "que cualquier persona puede abrir en el navegador sin instalar nada."],
    ["✅ Código reproducible",
     "El proyecto usa un entorno virtual, scripts modulares y base de datos SQLite. "
     "Cualquier persona puede clonar el repositorio y reproducir los resultados."],
    ["✅ Análisis completo",
     "El análisis incluye distribución de sentimientos, engagement, geografía, clusters "
     "temáticos, nubes de palabras y métricas de confianza del modelo."],
]
vt = Table(value_rows, colWidths=[4*cm, 13*cm])
vt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
    ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_CARD, C_STEP_BG]),
    ("TEXTCOLOR",     (0,0), (0,-1), C_GREEN),
    ("TEXTCOLOR",     (1,0), (1,-1), C_GRAY),
    ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
    ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("GRID",          (0,0), (-1,-1), 0.3, C_DARK),
    ("TOPPADDING",    (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ("VALIGN",        (0,0), (-1,-1), "TOP"),
]))
story += [vt, space(10),
          info_box("Este tipo de proyecto de análisis de sentimientos sobre datos reales de redes "
                   "sociales es exactamente lo que buscan los empleadores para roles de Analista de "
                   "Datos, Data Scientist Junior y NLP Engineer."),
          PageBreak()]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — ARQUITECTURA DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(2, "Arquitectura del Pipeline NLP",
                "Los 6 pasos del proceso completo"),
    space(10),
    p("El proyecto sigue una arquitectura de pipeline secuencial donde cada etapa lee "
      "de la base de datos SQLite y escribe sus resultados de vuelta, garantizando "
      "la reproducibilidad y la persistencia incremental.", sBody),
    space(8),
    h2("Diagrama del Pipeline"),
]

# Pipeline diagram as table
flow_rows = [
    ["① EXTRACCIÓN",    "descarga_masiva.py",
     "YouTube Data API v3\nlotes de 100 comentarios\ncheckpointing cada 500",
     "4,314 comentarios\nen original.db (SQLite)"],
    ["② LIMPIEZA",      "limpieza_texto.py",
     "Minúsculas · eliminar URLs\nstopwords NLTK · lematización spaCy\nes_core_news_sm",
     "Columna texto_limpio\nen base_limpia.db"],
    ["③ GEOGRAFÍA",     "extract_locations.py",
     "GeoText detecta ciudades/países\nNormalización de alias\nFiltro de falsos positivos",
     "Columna ubicacion\n28.4% geolocalizados"],
    ["④ TOKENIZACIÓN",  "tokenizacion.py",
     "nltk.word_tokenize\nconfigurado para español",
     "Columna tokens\n(separados por coma)"],
    ["⑤ SENTIMIENTO",   "clasificador_sentimientos.py",
     "RoBERTuito (pysentimiento)\nlotes de 32 · GPU/CPU\ncheckpointing",
     "Columnas sentimiento\ny probabilidad"],
    ["⑥ DASHBOARD",     "exportar_dashboard_html.py",
     "Plotly 6.x · Bootstrap 5\n10 gráficas · 5 pestañas\nNubes de palabras",
     "dashboard_sentimientos.html\n(2.3 MB autocontenido)"],
]

pt = Table(flow_rows, colWidths=[3.2*cm, 4.2*cm, 5.2*cm, 4.2*cm])
pt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (0,-1), C_STEP_BG),
    ("BACKGROUND",    (1,0), (1,-1), C_DARK),
    ("BACKGROUND",    (2,0), (2,-1), C_CARD),
    ("BACKGROUND",    (3,0), (3,-1), C_STEP_BG),
    ("TEXTCOLOR",     (0,0), (0,-1), C_CYAN),
    ("TEXTCOLOR",     (1,0), (1,-1), C_WHITE),
    ("TEXTCOLOR",     (2,0), (2,-1), C_GRAY),
    ("TEXTCOLOR",     (3,0), (3,-1), C_GREEN),
    ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME",      (1,0), (-1,-1), "Helvetica"),
    ("FONTSIZE",      (0,0), (-1,-1), 8.5),
    ("GRID",          (0,0), (-1,-1), 0.3, C_GRAY),
    ("TOPPADDING",    (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ("VALIGN",        (0,0), (-1,-1), "TOP"),
]))
story += [pt, space(10),
    h2("Tabla resumen del pipeline"),
    pipeline_table(),
    space(10),
    h2("Resultados clave del análisis"),
]

results_rows = [
    ["Comentarios totales",    "4,314",          "100% clasificados"],
    ["Sentimiento positivo",   "1,793  (41.6%)", "Categoría dominante"],
    ["Sentimiento negativo",   "1,531  (35.5%)", "Oposición significativa"],
    ["Sentimiento neutro",     "990    (22.9%)", "Contenido descriptivo"],
    ["Confianza media",        "0.782",          "78.2% certeza promedio"],
    ["Alta confianza (≥0.8)",  "54.2%",          "Predicciones decisivas"],
    ["Pico de actividad",      "1,915 comentarios", "Día 1 tras publicación"],
    ["Likes en comentarios +", "71.7% del total","Resonancia social asimétrica"],
    ["Países detectados",      "20+ ubicaciones","Alcance latinoamericano"],
    ["Ratio positivo/negativo","1.17x",          "Leve preponderancia positiva"],
]
rt = Table(results_rows, colWidths=[5.5*cm, 4.5*cm, 6.8*cm])
rt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
    ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_CARD, C_STEP_BG]),
    ("TEXTCOLOR",     (0,0), (0,-1), C_GRAY),
    ("TEXTCOLOR",     (1,0), (1,-1), C_CYAN),
    ("TEXTCOLOR",     (2,0), (2,-1), C_WHITE),
    ("FONTNAME",      (1,0), (1,-1), "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("GRID",          (0,0), (-1,-1), 0.3, C_DARK),
    ("TOPPADDING",    (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("LEFTPADDING",   (0,0), (-1,-1), 8),
]))
story += [rt, PageBreak()]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — ESTRUCTURA DEL REPOSITORIO
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(3, "Preparar el Repositorio Local",
                "Estructura de carpetas recomendada y archivos a incluir/excluir"),
    space(10),
    h2("Estructura recomendada del repositorio"),
    p("Antes de subir el proyecto a GitHub, es fundamental organizar los archivos "
      "de forma clara. Aquí está la estructura recomendada para que el repositorio "
      "sea profesional y comprensible para cualquier persona que lo visite:", sBody),
    space(6),
    code_block([
        "sentiment-analysis-youtube/",
        "│",
        "├── README.md                    # Descripción del proyecto (OBLIGATORIO)",
        "├── LICENSE                      # Licencia MIT (recomendada)",
        "├── .gitignore                   # Archivos a excluir de Git",
        "│",
        "├── Analisis_exploratorio/",
        "│   ├── scripts/                 # 📁 Todo el código Python",
        "│   │   ├── main.py",
        "│   │   ├── descarga_masiva.py",
        "│   │   ├── limpieza_texto.py",
        "│   │   ├── extract_locations.py",
        "│   │   ├── tokenizacion.py",
        "│   │   ├── clasificador_sentimientos.py",
        "│   │   ├── exportar_dashboard_html.py",
        "│   │   └── exportar_dashboard_html_en.py",
        "│   ├── notebooks/               # 📓 Jupyter Notebooks",
        "│   │   ├── EDA_Comentarios.ipynb",
        "│   │   └── metricas_rendimiento.ipynb",
        "│   └── requirements.txt         # Dependencias Python",
        "│",
        "└── output/                      # 📊 Resultados del análisis",
        "    ├── dashboard_sentimientos.html",
        "    └── dashboard_sentiment_analysis_en.html",
    ], "Árbol de archivos del repositorio"),
    space(10),
    h2("Archivos que NO debes subir a GitHub"),
    warning_box("Los archivos de base de datos (.db), entornos virtuales (env/) y archivos "
                "de caché hacen el repositorio muy pesado y no son necesarios para reproducir "
                "el proyecto."),
    space(6),
]

exclude_rows = [
    ["❌ env_eda/",           "Entorno virtual (>500 MB)", "Se recrea con pip install -r requirements.txt"],
    ["❌ data/*.db",          "Bases de datos SQLite (privacidad + tamaño)", "Subir solo la estructura vacía o datos de muestra"],
    ["❌ __pycache__/",       "Archivos de caché de Python", "Se regeneran automáticamente al ejecutar"],
    ["❌ .DS_Store",          "Archivo de macOS (metadatos de carpetas)", "No tiene utilidad en el repositorio"],
    ["❌ *.pyc",              "Python bytecode compilado", "Se genera automáticamente"],
    ["❌ output/*.png",       "Imágenes temporales de matplotlib", "Los dashboards HTML ya las incluyen embebidas"],
]
et = Table(exclude_rows, colWidths=[3.5*cm, 5.5*cm, 7.8*cm])
et.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
    ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_CARD, C_STEP_BG]),
    ("TEXTCOLOR",     (0,0), (0,-1), C_PINK),
    ("TEXTCOLOR",     (1,0), (1,-1), C_GRAY),
    ("TEXTCOLOR",     (2,0), (2,-1), C_WHITE),
    ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 8.5),
    ("GRID",          (0,0), (-1,-1), 0.3, C_DARK),
    ("TOPPADDING",    (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ("VALIGN",        (0,0), (-1,-1), "TOP"),
]))
story += [et, space(10),
    h2("Contenido del archivo .gitignore"),
    code_block([
        "# Entornos virtuales",
        "env_eda/",
        "env/",
        "venv/",
        ".venv/",
        "",
        "# Bases de datos (datos privados)",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "",
        "# Python cache",
        "__pycache__/",
        "*.py[cod]",
        "*.pyo",
        ".pytest_cache/",
        "",
        "# Jupyter",
        ".ipynb_checkpoints/",
        "",
        "# Sistema operativo",
        ".DS_Store",
        "Thumbs.db",
        "",
        "# Variables de entorno (claves API)",
        ".env",
        "*.env",
        "secrets.py",
        "",
        "# IDEs",
        ".vscode/",
        ".idea/",
        "*.spyderproject",
    ], ".gitignore"),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — CREAR CUENTA GITHUB
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(4, "Crear tu Cuenta de GitHub",
                "Registro gratuito en 5 minutos"),
    space(10),
    h2("¿Qué es GitHub?"),
    p("GitHub es la plataforma más grande del mundo para alojar código fuente. "
      "Con más de 100 millones de desarrolladores, es el estándar de la industria "
      "para mostrar proyectos técnicos. Un perfil activo en GitHub es prácticamente "
      "obligatorio para cualquier rol técnico en datos o desarrollo.", sBody),
    space(8),
    h2("Paso a paso para registrarse"),
]
reg_steps = [
    ("1. Ir al sitio web",
     "Abre tu navegador y visita: https://github.com/signup\n"
     "Verás el formulario de registro de GitHub."),
    ("2. Datos de registro",
     "• Correo electrónico: usa tu correo profesional (jfbernalp@gmail.com)\n"
     "• Contraseña: mínimo 15 caracteres o 8 con números y letras\n"
     "• Nombre de usuario: será parte de tu URL de portafolio\n"
     "  ✅ Bueno: jfbernal-data · jfbernalp · jfbernal-analisis\n"
     "  ❌ Evitar: usuario123 · pepito1990 · xXdataXx"),
    ("3. Verificación",
     "GitHub te enviará un correo con un código de 6 dígitos.\n"
     "Ingrésalo en el formulario para verificar tu email."),
    ("4. Configurar perfil",
     "Después del registro, ve a Settings → Profile y agrega:\n"
     "• Foto profesional (o avatar)\n"
     "• Nombre completo\n"
     "• Descripción breve: 'Data Analyst | NLP | Python'\n"
     "• Universidad: Unicafam"),
    ("5. Habilitar autenticación",
     "Ve a Settings → Password and authentication\n"
     "Activa Two-factor authentication (2FA) para mayor seguridad."),
]
for title, desc in reg_steps:
    story += [
        p(f"<b>{title}</b>", sH3),
        p(desc.replace("\n", "<br/>"), sBody),
        space(4),
    ]
story += [
    tip_box("El nombre de usuario de GitHub será visible en tu URL de portafolio: "
            "https://github.com/TU_USUARIO — elige algo profesional y fácil de recordar."),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — CONFIGURAR GIT
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(5, "Configurar Git en tu Computador",
                "Identidad, autenticación y configuración inicial"),
    space(10),
    h2("¿Qué es Git?"),
    p("Git es el sistema de control de versiones más usado del mundo. Registra cada "
      "cambio que haces en tu código, permitiéndote volver a versiones anteriores, "
      "trabajar en equipo y mantener un historial completo del proyecto. "
      "Git ya está instalado en tu Mac (versión 2.50.1 detectada).", sBody),
    space(8),
    h2("1. Configurar tu identidad en Git"),
    p("Abre la Terminal (Command + Space → 'Terminal') y ejecuta:", sBody),
    code_block([
        "# Configura tu nombre (aparecerá en cada commit)",
        'git config --global user.name "Juan Felipe Bernal"',
        "",
        "# Configura tu email (debe ser el mismo de GitHub)",
        'git config --global user.email "jfbernalp@gmail.com"',
        "",
        "# Verificar que quedó guardado",
        "git config --global --list",
    ], "Terminal — Configuración de identidad"),
    space(8),
    h2("2. Configurar la rama principal como 'main'"),
    code_block([
        "# GitHub usa 'main' como nombre por defecto (antes era 'master')",
        'git config --global init.defaultBranch main',
    ], "Terminal"),
    space(8),
    h2("3. Autenticarse con GitHub (Token Personal)"),
    p("GitHub ya no acepta contraseñas por línea de comandos. Debes usar un "
      "<b>Personal Access Token (PAT)</b>. Sigue estos pasos:", sBody),
    space(6),
]
token_steps = [
    ("En GitHub →", "Settings → Developer settings → Personal access tokens → Tokens (classic)"),
    ("→ Generate new token", "Haz clic en 'Generate new token (classic)'"),
    ("Nombre del token:", "Portfolio deployment (o cualquier nombre descriptivo)"),
    ("Expiración:", "Selecciona '90 days' o 'No expiration' para tu portafolio"),
    ("Permisos a marcar:", "✅ repo  ✅ workflow  ✅ read:org"),
    ("Copiar el token:", "⚠️ IMPORTANTE: Cópialo ahora, no lo podrás ver de nuevo."),
]
tt = Table(token_steps, colWidths=[4*cm, 12.8*cm])
tt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
    ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_CARD, C_STEP_BG]),
    ("TEXTCOLOR",     (0,0), (0,-1), C_CYAN),
    ("TEXTCOLOR",     (1,0), (1,-1), C_GRAY),
    ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("GRID",          (0,0), (-1,-1), 0.3, C_DARK),
    ("TOPPADDING",    (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ("VALIGN",        (0,0), (-1,-1), "TOP"),
]))
story += [tt, space(8),
    code_block([
        "# Cuando Git te pida la contraseña, pega el Token (no tu contraseña de GitHub)",
        "# En Mac, el sistema operativo puede guardar el token en el Keychain:",
        "git config --global credential.helper osxkeychain",
        "",
        "# Alternativa: guardar el token en el archivo de credenciales",
        "git config --global credential.helper store",
        "# (La próxima vez que hagas push, escribe tu usuario y pega el token)",
    ], "Guardar credenciales para no escribirlas cada vez"),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — CREAR REPOSITORIO EN GITHUB
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(6, "Crear el Repositorio en GitHub",
                "Nombre profesional, visibilidad pública y configuración inicial"),
    space(10),
    h2("Paso a paso en la interfaz web de GitHub"),
]
create_steps = [
    ("1. Nuevo repositorio",
     "En GitHub, haz clic en el botón verde '+ New' (arriba a la derecha)\n"
     "o ve directamente a: https://github.com/new"),
    ("2. Nombre del repositorio",
     "Escribe un nombre profesional y descriptivo. Recomendaciones:\n"
     "  ✅ youtube-sentiment-analysis\n"
     "  ✅ nlp-sentiment-youtube-elsalvador\n"
     "  ✅ analisis-sentimientos-youtube-nlp\n"
     "  ❌ Evitar: mi-proyecto, trabajo, prueba123"),
    ("3. Descripción",
     "Agrega una descripción corta (aparece en los resultados de búsqueda):\n"
     "  'NLP pipeline to classify 4,314 YouTube comments using RoBERTuito. "
     "spaCy · NLTK · Plotly · Python'"),
    ("4. Visibilidad",
     "Selecciona 'Public' — los repositorios públicos son visibles para "
     "reclutadores y es requisito para GitHub Pages gratuito."),
    ("5. Inicializar con README",
     "✅ Marca 'Add a README file' — esto crea automáticamente la rama 'main' "
     "y evita errores al hacer el primer push desde tu computador."),
    ("6. .gitignore",
     "En el selector de template para .gitignore, busca y selecciona 'Python'.\n"
     "Esto agrega un .gitignore básico que ya excluye __pycache__, *.pyc, etc."),
    ("7. Licencia",
     "Selecciona 'MIT License' — es la más común para proyectos académicos "
     "y de portafolio. Permite que otros usen y aprendan de tu código."),
    ("8. Crear repositorio",
     "Haz clic en 'Create repository' (botón verde).\n"
     "GitHub te llevará a la página de tu nuevo repositorio."),
]
for title, desc in create_steps:
    story += [
        p(f"<b>{title}</b>", sH3),
        p(desc.replace("\n", "<br/>"), sBody),
        space(4),
    ]
story += [
    info_box("Una vez creado, verás la URL de tu repositorio en la forma:\n"
             "https://github.com/TU_USUARIO/youtube-sentiment-analysis\n"
             "Cópiala, la necesitarás en el siguiente paso."),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — INICIALIZAR GIT LOCAL
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(7, "Inicializar Git e Ignorar Archivos",
                "git init, .gitignore y primer commit local"),
    space(10),
    h2("1. Abrir Terminal en la carpeta del proyecto"),
    code_block([
        "# Navegar a la raíz del proyecto",
        "cd /Users/jfbernalp/Documents/Procesamiento_texto",
        "",
        "# Verificar que estás en el lugar correcto",
        "pwd",
        "# Debe mostrar: /Users/jfbernalp/Documents/Procesamiento_texto",
        "",
        "# Ver los archivos disponibles",
        "ls -la",
    ], "Terminal — Navegar al proyecto"),
    space(8),
    h2("2. Crear el archivo .gitignore"),
    p("Crea el archivo .gitignore en la raíz del proyecto con el contenido "
      "descrito en la Sección 3:", sBody),
    code_block([
        "# Crear el .gitignore directamente desde la terminal",
        "cat > .gitignore << 'EOF'",
        "env_eda/",
        "*.db",
        "*.sqlite",
        "__pycache__/",
        "*.py[cod]",
        ".ipynb_checkpoints/",
        ".DS_Store",
        ".env",
        "secrets.py",
        "year-mm-dd-backup.dump",
        "EOF",
        "",
        "# Verificar que se creó correctamente",
        "cat .gitignore",
    ], "Terminal — Crear .gitignore"),
    space(8),
    h2("3. Inicializar el repositorio Git"),
    code_block([
        "# Inicializar Git en la carpeta del proyecto",
        "git init",
        "# Output esperado: 'Initialized empty Git repository in .../.git/'",
        "",
        "# Conectar con el repositorio de GitHub (reemplaza TU_USUARIO)",
        "git remote add origin https://github.com/TU_USUARIO/youtube-sentiment-analysis.git",
        "",
        "# Traer el README y .gitignore que GitHub creó",
        "git pull origin main --allow-unrelated-histories",
    ], "Terminal — Inicializar Git"),
    space(8),
    h2("4. Verificar qué archivos serán incluidos"),
    code_block([
        "# Ver el estado actual (archivos nuevos en rojo = no trackeados aún)",
        "git status",
        "",
        "# Verificar que las bases de datos están excluidas",
        "git check-ignore -v Analisis_exploratorio/data/base_limpia.db",
        "# Debe mostrar: .gitignore:2:*.db  data/base_limpia.db",
        "",
        "# Ver todos los archivos que SÍ serán incluidos",
        "git ls-files --others --exclude-standard",
    ], "Terminal — Verificar estado"),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — SUBIR EL CÓDIGO
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(8, "Subir el Código a GitHub",
                "git add → commit → push — el flujo completo"),
    space(10),
    h2("El flujo básico de Git"),
    p("Git tiene tres estados principales por los que pasa un archivo antes de "
      "llegar a GitHub:", sBody),
    space(6),
]

flow_git = [
    ["Working Directory\n(archivos en tu PC)",
     "→\ngit add",
     "Staging Area\n(preparados para commit)",
     "→\ngit commit",
     "Repositorio Local\n(.git/)",
     "→\ngit push",
     "GitHub\n(repositorio remoto)"]
]
fg = Table(flow_git, colWidths=[2.4*cm, 1.2*cm, 2.8*cm, 1.8*cm, 2.8*cm, 1.5*cm, 2.8*cm])
fg.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (0,0), C_DARK),
    ("BACKGROUND",    (2,0), (2,0), C_STEP_BG),
    ("BACKGROUND",    (4,0), (4,0), C_DARK),
    ("BACKGROUND",    (6,0), (6,0), C_CARD),
    ("TEXTCOLOR",     (0,0), (0,0), C_GRAY),
    ("TEXTCOLOR",     (1,0), (1,0), C_CYAN),
    ("TEXTCOLOR",     (2,0), (2,0), C_WHITE),
    ("TEXTCOLOR",     (3,0), (3,0), C_CYAN),
    ("TEXTCOLOR",     (4,0), (4,0), C_WHITE),
    ("TEXTCOLOR",     (5,0), (5,0), C_CYAN),
    ("TEXTCOLOR",     (6,0), (6,0), C_GREEN),
    ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
    ("FONTSIZE",      (0,0), (-1,-1), 8),
    ("FONTNAME",      (1,0), (1,0), "Helvetica-Bold"),
    ("FONTNAME",      (3,0), (3,0), "Helvetica-Bold"),
    ("FONTNAME",      (5,0), (5,0), "Helvetica-Bold"),
    ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",    (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ("BOX",           (0,0), (-1,-1), 0.3, C_GRAY),
]))
story += [fg, space(10),
    h2("1. Agregar los archivos del proyecto"),
    code_block([
        "# Agregar todos los scripts Python",
        "git add Analisis_exploratorio/scripts/",
        "",
        "# Agregar los notebooks",
        "git add Analisis_exploratorio/notebooks/",
        "",
        "# Agregar el dashboard (la evidencia visual del análisis)",
        "git add output/dashboard_sentimientos.html",
        "git add output/dashboard_sentiment_analysis_en.html",
        "",
        "# Agregar archivos de configuración del proyecto",
        "git add .gitignore",
        "",
        "# Verificar lo que está en staging (en verde = listo para commit)",
        "git status",
    ], "Terminal — git add"),
    space(8),
    h2("2. Crear el primer commit"),
    code_block([
        "# Crear el commit con un mensaje descriptivo y profesional",
        'git commit -m "feat: pipeline NLP completo para análisis de sentimientos YouTube',
        "",
        "- Extracción: 4314 comentarios vía YouTube Data API v3",
        "- Preprocesamiento: limpieza, lematización spaCy, tokenización NLTK",
        "- Clasificación: RoBERTuito (pysentimiento) · 41.6% pos · 35.5% neg",
        "- Dashboard interactivo: Plotly + Bootstrap · 5 pestañas · nubes de palabras",
        '- Versiones en español e inglés"',
    ], "Terminal — git commit"),
    space(8),
    h2("3. Subir al repositorio de GitHub"),
    code_block([
        "# Subir la rama main a GitHub",
        "git push -u origin main",
        "",
        "# Si pide usuario: escribe tu nombre de usuario de GitHub",
        "# Si pide contraseña: pega el Personal Access Token (NO tu contraseña)",
        "",
        "# Output esperado:",
        "# Enumerating objects: X, done.",
        "# Counting objects: 100% (X/X), done.",
        "# Writing objects: 100% (X/X), 2.30 MiB | ...",
        "# Branch 'main' set up to track remote branch 'main' from 'origin'.",
    ], "Terminal — git push"),
    space(8),
    h2("4. Verificar que el código llegó a GitHub"),
    p("Abre tu navegador y ve a: <b>https://github.com/TU_USUARIO/youtube-sentiment-analysis</b>", sBody),
    p("Deberías ver todos tus scripts y el dashboard en la interfaz de GitHub.", sBody),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 9 — GITHUB PAGES
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(9, "Publicar el Dashboard con GitHub Pages",
                "URL pública y gratuita para tu dashboard interactivo"),
    space(10),
    p("GitHub Pages es un servicio gratuito de GitHub que convierte los archivos "
      "HTML de tu repositorio en un sitio web público accesible desde cualquier "
      "navegador del mundo. No necesitas servidor, no necesitas pagar hosting.", sBody),
    space(8),
    h2("Paso 1 — Subir el dashboard como index.html"),
    p("Para que GitHub Pages sirva el dashboard automáticamente, el archivo principal "
      "debe llamarse <b>index.html</b>. Tienes dos opciones:", sBody),
    code_block([
        "# Opción A: Renombrar el archivo antes de subirlo",
        "cp output/dashboard_sentimientos.html output/index.html",
        "git add output/index.html",
        'git commit -m "docs: agregar index.html para GitHub Pages"',
        "git push origin main",
        "",
        "# Opción B: Crear una carpeta docs/ (GitHub Pages la detecta automáticamente)",
        "mkdir -p docs",
        "cp output/dashboard_sentimientos.html docs/index.html",
        "git add docs/",
        'git commit -m "docs: dashboard en carpeta docs/ para GitHub Pages"',
        "git push origin main",
    ], "Terminal — Preparar archivo para Pages"),
    space(8),
    h2("Paso 2 — Activar GitHub Pages en la configuración"),
]
pages_steps = [
    ("1.", "Ve a tu repositorio en GitHub"),
    ("2.", "Haz clic en la pestaña 'Settings' (ícono de engranaje)"),
    ("3.", "En el menú lateral izquierdo, busca y haz clic en 'Pages'"),
    ("4.", "En 'Source' selecciona: Deploy from a branch"),
    ("5.", "En 'Branch' selecciona: main"),
    ("6.", "En la carpeta selecciona: /docs (o / root si pusiste index.html en la raíz)"),
    ("7.", "Haz clic en 'Save'"),
    ("8.", "Espera 1-2 minutos — GitHub compilará el sitio"),
]
for n, desc in pages_steps:
    story.append(p(f"<b>{n}</b>  {desc}", sBulletCyan))
story.append(space(8))
story.append(h2("Paso 3 — Obtener y compartir la URL"))
story += [
    code_block([
        "# Tu URL de GitHub Pages tendrá este formato:",
        "https://TU_USUARIO.github.io/youtube-sentiment-analysis/",
        "",
        "# Si pusiste el archivo en /docs:",
        "https://TU_USUARIO.github.io/youtube-sentiment-analysis/",
        "",
        "# Si pusiste el archivo en /output:",
        "https://TU_USUARIO.github.io/youtube-sentiment-analysis/output/index.html",
    ], "URL de tu dashboard público"),
    space(8),
    tip_box("Una vez activo, cualquier persona en el mundo puede acceder a tu dashboard "
            "con solo abrir el link. Es perfecto para compartir en clase, en LinkedIn, "
            "en tu CV o con empleadores."),
    space(8),
    h2("Actualizar el dashboard en el futuro"),
    code_block([
        "# Cuando regeneres el dashboard con Python, solo haz:",
        "cp output/dashboard_sentimientos.html docs/index.html",
        "git add docs/index.html",
        'git commit -m "update: regenerar dashboard con nuevos datos"',
        "git push origin main",
        "",
        "# GitHub Pages se actualizará automáticamente en ~1 minuto",
    ], "Terminal — Actualizar el dashboard"),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 10 — README PROFESIONAL
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(10, "Escribir un README Profesional",
                "La tarjeta de presentación del proyecto en GitHub"),
    space(10),
    p("El README.md es lo primero que ve cualquier persona que visita tu repositorio. "
      "Un buen README puede hacer la diferencia entre que alguien explore tu proyecto "
      "o lo ignore. Usa el formato Markdown (.md) que GitHub renderiza automáticamente.", sBody),
    space(8),
    h2("Plantilla del README.md recomendada"),
    code_block([
        "# 🧠 Análisis de Sentimientos en Comentarios de YouTube",
        "",
        "> Pipeline NLP para clasificar 4,314 comentarios del video",
        '> *"Esto es lo que nadie te cuenta de El Salvador"*',
        "> usando el modelo RoBERTuito (Transformer bidireccional para español).",
        "",
        "## 🔗 Dashboard Interactivo",
        "",
        "[![Dashboard](https://img.shields.io/badge/Ver%20Dashboard-Live-00f2fe)]",
        "(https://TU_USUARIO.github.io/youtube-sentiment-analysis/)",
        "",
        "## 📊 Resultados Clave",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        "| Comentarios analizados | 4,314 |",
        "| Sentimiento positivo | 41.6% |",
        "| Sentimiento negativo | 35.5% |",
        "| Confianza media del modelo | 0.782 |",
        "| Likes en comentarios positivos | 71.7% del total |",
        "",
        "## 🏗️ Pipeline",
        "",
        "```",
        "YouTube API → SQLite → spaCy → NLTK → RoBERTuito → Plotly Dashboard",
        "```",
        "",
        "## 🚀 Cómo ejecutar",
        "",
        "```bash",
        "# 1. Clonar el repositorio",
        "git clone https://github.com/TU_USUARIO/youtube-sentiment-analysis.git",
        "",
        "# 2. Crear entorno virtual",
        "python3 -m venv env && source env/bin/activate",
        "",
        "# 3. Instalar dependencias",
        "pip install -r requirements.txt",
        "",
        "# 4. Generar el dashboard",
        "python Analisis_exploratorio/scripts/exportar_dashboard_html.py",
        "```",
        "",
        "## 🛠️ Tecnologías",
        "",
        "Python · spaCy · NLTK · pysentimiento · scikit-learn · Plotly · SQLite",
    ], "README.md — Plantilla"),
    space(8),
    h2("Crear el requirements.txt"),
    p("El requirements.txt lista todas las librerías necesarias para reproducir "
      "el proyecto. Créalo así:", sBody),
    code_block([
        "# Dentro del entorno virtual activo, ejecutar:",
        "source Analisis_exploratorio/env_eda/bin/activate",
        "",
        "# Generar el archivo con las versiones exactas instaladas",
        "pip freeze > requirements.txt",
        "",
        "# Verificar el contenido",
        "cat requirements.txt",
        "",
        "# Agregar al repositorio",
        "git add requirements.txt",
        'git commit -m "chore: agregar requirements.txt"',
        "git push origin main",
    ], "Terminal — Crear requirements.txt"),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 11 — VERIFICAR Y MANTENER
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(11, "Verificar y Mantener el Repositorio",
                "Buenas prácticas y comandos útiles"),
    space(10),
    h2("Comandos Git esenciales para el día a día"),
]
cmd_rows = [
    ["git status",                 "Ver qué archivos cambiaron desde el último commit"],
    ["git log --oneline",          "Ver el historial de commits de forma compacta"],
    ["git diff",                   "Ver exactamente qué líneas cambiaron en los archivos"],
    ["git add .",                  "Agregar TODOS los archivos modificados al staging"],
    ['git commit -m "mensaje"',    "Crear un nuevo punto de control en el historial"],
    ["git push origin main",       "Enviar los commits locales a GitHub"],
    ["git pull origin main",       "Traer cambios de GitHub a tu computador"],
    ["git restore archivo.py",     "Descartar cambios en un archivo (¡irreversible!)"],
    ["git log --oneline -5",       "Ver los últimos 5 commits"],
    ["git remote -v",              "Ver a qué URL remota está conectado el repositorio"],
]
ct = Table(cmd_rows, colWidths=[6*cm, 11*cm])
ct.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (0,-1), C_CODE_BG),
    ("BACKGROUND",    (1,0), (1,-1), C_CARD),
    ("ROWBACKGROUNDS",(1,0), (1,-1), [C_CARD, C_STEP_BG]),
    ("TEXTCOLOR",     (0,0), (0,-1), C_GREEN),
    ("TEXTCOLOR",     (1,0), (1,-1), C_GRAY),
    ("FONTNAME",      (0,0), (0,-1), "Courier"),
    ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("GRID",          (0,0), (-1,-1), 0.3, C_DARK),
    ("TOPPADDING",    (0,0), (-1,-1), 7),
    ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
]))
story += [ct, space(10),
    h2("Flujo de trabajo para actualizar el proyecto"),
    code_block([
        "# 1. Haz tus cambios en los scripts Python",
        "",
        "# 2. Regenera el dashboard",
        "source Analisis_exploratorio/env_eda/bin/activate",
        "python Analisis_exploratorio/scripts/exportar_dashboard_html.py",
        "",
        "# 3. Copia el dashboard actualizado a docs/",
        "cp output/dashboard_sentimientos.html docs/index.html",
        "",
        "# 4. Sube los cambios a GitHub",
        "git add .",
        'git commit -m "update: mejorar visualizaciones del dashboard"',
        "git push origin main",
        "",
        "# 5. En 1-2 minutos, GitHub Pages actualiza el sitio web público",
    ], "Flujo completo de actualización"),
    space(8),
    h2("Checklist de verificación del repositorio"),
]
checklist = [
    ("✅", "README.md presente y con descripción clara del proyecto"),
    ("✅", "requirements.txt con todas las dependencias"),
    ("✅", ".gitignore excluyendo env/, *.db, __pycache__"),
    ("✅", "Código organizado en carpetas lógicas (scripts/, notebooks/, output/)"),
    ("✅", "Dashboard HTML subido y accesible vía GitHub Pages"),
    ("✅", "Commits con mensajes descriptivos (no 'cambios', 'fix', 'wip')"),
    ("✅", "Repositorio en modo Public"),
    ("✅", "Licencia MIT agregada"),
    ("⭐", "Agregar topics al repo: python, nlp, sentiment-analysis, youtube, plotly"),
    ("⭐", "Agregar website URL (el link de GitHub Pages) en el campo 'About'"),
    ("⭐", "Fijar el repositorio en tu perfil de GitHub (máximo 6 repositorios fijados)"),
]
for icon, text in checklist:
    story.append(p(f"<b>{icon}</b>  {text}", sBullet))
story += [PageBreak()]

# ─────────────────────────────────────────────────────────────────────────
# SECCIÓN 12 — PRESENTACIÓN EN CLASE
# ─────────────────────────────────────────────────────────────────────────
story += [
    step_header(12, "Presentación en Clase y Portafolio",
                "Cómo mostrar y explicar el proyecto efectivamente"),
    space(10),
    h2("Guión de presentación recomendado (5-10 minutos)"),
]
guion = [
    ("0:00 – 1:00", "INTRODUCCIÓN",
     "Presenta el tema: 'Este proyecto analiza automáticamente qué piensan "
     "las personas sobre las políticas de seguridad en El Salvador, usando comentarios "
     "reales de YouTube y un modelo de Inteligencia Artificial.'"),
    ("1:00 – 2:30", "EL PIPELINE (Tab 1 del dashboard)",
     "Explica los 6 pasos del proceso. Muestra el diagrama del pipeline. "
     "Destaca que usaste la YouTube Data API v3 para extraer datos REALES, "
     "no un dataset predefinido."),
    ("2:30 – 4:00", "RESULTADOS DE SENTIMIENTO (Tabs 1 y 2)",
     "Muestra las gráficas de distribución. Explica el ratio 1.17x positivo/negativo. "
     "Muestra la asimetría de likes: los positivos se llevan el 71.7% aunque son el 41.6%."),
    ("4:00 – 5:30", "NUBES DE PALABRAS (Tab 4)",
     "Aquí está el momento más visual. Muestra el antes y después del preprocesamiento. "
     "El contraste es muy llamativo y explica de manera intuitiva qué hace el NLP."),
    ("5:30 – 7:00", "CONFIANZA DEL MODELO (Tab 2)",
     "Explica que RoBERTuito es un Transformer bidireccional entrenado sobre Twitter "
     "latinoamericano. Muestra el score de confianza y por qué 'neutro' tiene menor certeza."),
    ("7:00 – 8:30", "GEOGRAFÍA (Tab 3)",
     "Muestra el mapa de origen. Destaca que el debate va mucho más allá de El Salvador: "
     "Colombia, Venezuela, Argentina. Esto conecta el análisis con fenómenos sociales más amplios."),
    ("8:30 – 10:00", "CONCLUSIONES Y REPO (Tab 5 + GitHub)",
     "Lee 2-3 hallazgos clave. Muestra el repositorio de GitHub y el link público del dashboard. "
     "Termina con: 'Este es mi portafolio en GitHub. Pueden ver el código y replicar el análisis.'"),
]
for tiempo, titulo, desc in guion:
    gt = Table([[
        p(tiempo, S("t", fontName="Courier", fontSize=8.5, textColor=C_CYAN,
                    leading=12, alignment=TA_CENTER)),
        p(f"<b>{titulo}</b>", S("tt", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=C_YELLOW, leading=12)),
        p(desc, sBody)
    ]], colWidths=[2.8*cm, 3.5*cm, 10.5*cm])
    gt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), C_STEP_BG),
        ("BACKGROUND",    (1,0), (1,-1), C_DARK),
        ("BACKGROUND",    (2,0), (2,-1), C_CARD),
        ("GRID",          (0,0), (-1,-1), 0.3, C_GRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story += [gt, space(3)]

story += [
    space(8),
    h2("Cómo incluirlo en tu CV y LinkedIn"),
    p("Una vez que el repositorio esté en GitHub y el dashboard en GitHub Pages:", sBody),
]
cv_tips = [
    "En tu CV, bajo 'Proyectos': <b>Análisis de Sentimientos YouTube (NLP)</b> — "
    "Python · RoBERTuito · Plotly | github.com/TU_USUARIO/youtube-sentiment-analysis",
    "En LinkedIn → Sección 'Proyectos': agrega el link del dashboard de GitHub Pages "
    "como URL del proyecto. Sube una captura del dashboard como imagen del proyecto.",
    "En la sección 'Habilidades' de LinkedIn agrega: Natural Language Processing, "
    "Sentiment Analysis, Python, Plotly, scikit-learn, spaCy.",
    "Ancla el repositorio en tu perfil de GitHub: Settings → Customize your pins → "
    "selecciona este repositorio.",
    "Comparte el link del dashboard en WhatsApp o email: "
    "'Mira mi análisis de sentimientos sobre El Salvador — link aquí'",
]
for tip in cv_tips:
    story.append(p(f"▸  {tip}", sBulletCyan))

story += [
    space(10),
    info_box("Un dashboard interactivo publicado en GitHub Pages que cualquiera pueda "
             "abrir con un clic es la demostración más poderosa de tus habilidades técnicas. "
             "Muestra que no solo escribes código, sino que produces resultados reales y comunicables."),
    PageBreak()
]

# ─────────────────────────────────────────────────────────────────────────
# PÁGINA FINAL — RESUMEN RÁPIDO
# ─────────────────────────────────────────────────────────────────────────
story += [
    p("RESUMEN DE COMANDOS", S("", fontName="Helvetica-Bold", fontSize=13,
                                textColor=C_CYAN, alignment=TA_CENTER, spaceAfter=8)),
    hr(C_CYAN),
    space(4),
    h2("Todos los comandos en orden — de cero a GitHub en 15 minutos"),
    code_block([
        "# ─────────────────────────────────────────────────────────────",
        "# PASO 1: Configurar Git (solo la primera vez)",
        "# ─────────────────────────────────────────────────────────────",
        'git config --global user.name "Tu Nombre"',
        'git config --global user.email "tu@email.com"',
        'git config --global init.defaultBranch main',
        "git config --global credential.helper osxkeychain",
        "",
        "# ─────────────────────────────────────────────────────────────",
        "# PASO 2: Ir a la carpeta del proyecto",
        "# ─────────────────────────────────────────────────────────────",
        "cd /Users/jfbernalp/Documents/Procesamiento_texto",
        "",
        "# ─────────────────────────────────────────────────────────────",
        "# PASO 3: Inicializar y conectar con GitHub",
        "# ─────────────────────────────────────────────────────────────",
        "git init",
        "git remote add origin https://github.com/TU_USUARIO/youtube-sentiment-analysis.git",
        "git pull origin main --allow-unrelated-histories",
    ], "COMANDOS — Configuración e Inicialización (Pasos 1-3)"),
    space(6),
    code_block([
        "# ─────────────────────────────────────────────────────────────",
        "# PASO 4: Preparar y subir los archivos",
        "# ─────────────────────────────────────────────────────────────",
        "git add Analisis_exploratorio/scripts/",
        "git add Analisis_exploratorio/notebooks/",
        "git add output/dashboard_sentimientos.html",
        "git add output/dashboard_sentiment_analysis_en.html",
        "git add .gitignore requirements.txt",
        'git commit -m "feat: pipeline NLP completo + dashboard interactivo"',
        "git push -u origin main",
        "",
        "# ─────────────────────────────────────────────────────────────",
        "# PASO 5: Activar GitHub Pages (se hace en la web de GitHub)",
        "# Settings → Pages → main branch → /docs o /root → Save",
        "# ─────────────────────────────────────────────────────────────",
        "mkdir -p docs",
        "cp output/dashboard_sentimientos.html docs/index.html",
        "git add docs/",
        'git commit -m "docs: publicar dashboard en GitHub Pages"',
        "git push origin main",
        "",
        "# Tu dashboard estará en:",
        "# https://TU_USUARIO.github.io/youtube-sentiment-analysis/",
    ], "COMANDOS — Subida y GitHub Pages (Pasos 4-5)"),
    space(10),
    hr(C_PINK),
    space(8),
    p("Pipeline NLP · spaCy · NLTK · RoBERTuito · scikit-learn · Plotly 6.x · Python 3.13",
      sCentered),
    p("Fundación Universitaria Cafam — Unicafam · Tecnología en Análisis de Datos · 2025",
      sCentered),
]

# ═══════════════════════════════════════════════════════════════════════════
# GENERAR EL PDF
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="Guía GitHub Portafolio — Análisis de Sentimientos YouTube",
    author="Pipeline NLP · Unicafam 2025",
    subject="Cómo subir un proyecto NLP a GitHub como portafolio profesional",
)

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

size_mb = os.path.getsize(PDF_PATH) / 1024 / 1024
print(f"✅ PDF generado: {PDF_PATH}")
print(f"   Tamaño: {size_mb:.1f} MB")
print(f"   Páginas estimadas: ~{len(story)//20 + 1}")
