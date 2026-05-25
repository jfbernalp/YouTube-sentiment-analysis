"""Genera el informe académico APA en formato .docx."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ── helpers ────────────────────────────────────────────────────────────────────

def set_font(run, name="Times New Roman", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def paragraph(doc, text="", align=WD_ALIGN_PARAGRAPH.LEFT, bold=False,
               italic=False, size=12, space_before=0, space_after=0,
               first_indent=None, keep_with_next=False):
    p = doc.add_paragraph()
    p.alignment = align
    fmt = p.paragraph_format
    fmt.space_before = Pt(space_before)
    fmt.space_after  = Pt(space_after)
    fmt.line_spacing = Pt(24)          # interlineado doble (24pt = 2 × 12pt)
    if first_indent is not None:
        fmt.first_line_indent = Cm(first_indent)
    if keep_with_next:
        fmt.keep_with_next = True
    if text:
        run = p.add_run(text)
        set_font(run, bold=bold, italic=italic, size=size)
    return p

def heading1(doc, text):
    """Título de sección APA: centrado, negrita, mayúsculas."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt = p.paragraph_format
    fmt.space_before = Pt(24)
    fmt.space_after  = Pt(0)
    fmt.line_spacing = Pt(24)
    run = p.add_run(text.upper())
    set_font(run, bold=True)
    return p

def heading2(doc, text):
    """Subtítulo APA nivel 2: alineado izquierda, negrita."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = p.paragraph_format
    fmt.space_before = Pt(12)
    fmt.space_after  = Pt(0)
    fmt.line_spacing = Pt(24)
    run = p.add_run(text)
    set_font(run, bold=True)
    return p

def heading3(doc, text):
    """Subtítulo APA nivel 3: sangría, negrita cursiva."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = p.paragraph_format
    fmt.first_line_indent = Cm(1.27)
    fmt.space_before = Pt(0)
    fmt.space_after  = Pt(0)
    fmt.line_spacing = Pt(24)
    run = p.add_run(text)
    set_font(run, bold=True, italic=True)
    return p

def body(doc, text, indent=True):
    """Párrafo de cuerpo con sangría de primera línea."""
    return paragraph(doc, text, first_indent=1.27 if indent else 0)

def table_apa(doc, headers, rows, caption_num, caption_text):
    """Crea tabla con estilo APA: título arriba, sin bordes laterales."""
    # Nota de tabla
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(12)
    p_cap.paragraph_format.space_after  = Pt(0)
    p_cap.paragraph_format.line_spacing = Pt(24)
    r = p_cap.add_run(f"Tabla {caption_num}")
    set_font(r, bold=True, italic=False)
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after  = Pt(4)
    p_title.paragraph_format.line_spacing = Pt(24)
    r2 = p_title.add_run(caption_text)
    set_font(r2, italic=True)

    ncols = len(headers)
    nrows = len(rows)
    tbl = doc.add_table(rows=1 + nrows, cols=ncols)
    tbl.style = "Table Grid"

    # Encabezados
    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_font(run, bold=True, size=11)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Datos
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            cell = tbl.rows[i + 1].cells[j]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    set_font(run, size=11)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Nota
    p_nota = doc.add_paragraph()
    p_nota.paragraph_format.space_before = Pt(2)
    p_nota.paragraph_format.space_after  = Pt(12)
    p_nota.paragraph_format.line_spacing = Pt(24)
    r_n = p_nota.add_run("Nota. ")
    set_font(r_n, italic=True, size=10)
    r_n2 = p_nota.add_run("Elaboración propia a partir del análisis del corpus de comentarios de YouTube.")
    set_font(r_n2, size=10)
    return tbl

# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENTO
# ══════════════════════════════════════════════════════════════════════════════

doc = Document()

# ── Configuración de página APA ──
section = doc.sections[0]
section.page_height = Cm(27.94)
section.page_width  = Cm(21.59)
for attr in ("left_margin","right_margin","top_margin","bottom_margin"):
    setattr(section, attr, Cm(2.54))

# Estilo base del documento
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

# ══════════════════════════════════════════════════════════════════════════════
#  PORTADA
# ══════════════════════════════════════════════════════════════════════════════

for _ in range(5):
    paragraph(doc, "")

paragraph(doc,
    "Análisis de Sentimientos en Comentarios de YouTube Sobre la Gestión\n"
    "de Seguridad Pública en El Salvador: Un Enfoque de Procesamiento de\n"
    "Lenguaje Natural con Python",
    align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)

paragraph(doc, "")
paragraph(doc, "[NOMBRE AUTOR]", align=WD_ALIGN_PARAGRAPH.CENTER)
paragraph(doc, "")
paragraph(doc, "[NOMBRE MATERIA]", align=WD_ALIGN_PARAGRAPH.CENTER)
paragraph(doc, "[NOMBRE PROFESOR]", align=WD_ALIGN_PARAGRAPH.CENTER)
paragraph(doc, "")
paragraph(doc, "Fundación Universitaria Cafam — Unicafam", align=WD_ALIGN_PARAGRAPH.CENTER)
paragraph(doc, "Tecnología en Análisis de Datos", align=WD_ALIGN_PARAGRAPH.CENTER)
paragraph(doc, "Bogotá, Colombia", align=WD_ALIGN_PARAGRAPH.CENTER)
paragraph(doc, "2025", align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  RESUMEN
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Resumen")
paragraph(doc, "")

resumen = (
    "Este proyecto implementa un pipeline de Procesamiento de Lenguaje Natural (NLP) "
    "para analizar la percepción pública expresada en los comentarios del video de YouTube "
    "\"Esto es lo que nadie te cuenta de El Salvador\" (2026), publicado en el canal del "
    "periodista Juan. El corpus está conformado por 4.314 comentarios recolectados mediante "
    "la YouTube Data API v3, comprendidos entre el 1 de febrero y el 17 de marzo de 2026. "
    "El pipeline integra cinco etapas secuenciales: descarga y almacenamiento en SQLite, "
    "limpieza y lematización con spaCy (modelo es_core_news_sm), tokenización con NLTK, "
    "clasificación de sentimientos con el modelo RoBERTuito de pysentimiento y análisis "
    "exploratorio multidimensional. Los resultados revelan que el 41,6% de los comentarios "
    "son positivos, el 35,5% negativos y el 22,9% neutros, con una confianza media del "
    "modelo de 0,782. Los comentarios positivos concentran el 71,7% de los likes totales "
    "(13.831 de 19.277), lo que evidencia una mayor resonancia social del apoyo a las "
    "políticas de seguridad de Bukele. La clusterización temática con K-Means (K=8) "
    "identificó grupos diferenciados: elogios al reportero, debate sobre derechos humanos, "
    "comparaciones internacionales y discusión sobre el precio de la paz. El vocabulario "
    "discriminante por clase confirma separabilidad léxica real: 40,9% de términos exclusivos "
    "en positivo, 59,7% en negativo y 33,3% en neutro. El proyecto demuestra la viabilidad "
    "técnica de los enfoques NLP supervisados para el análisis de opinión política en "
    "plataformas digitales de habla hispana."
)
body(doc, resumen)
paragraph(doc, "")
p_kw = doc.add_paragraph()
p_kw.paragraph_format.first_line_indent = Cm(1.27)
p_kw.paragraph_format.line_spacing = Pt(24)
r_kw1 = p_kw.add_run("Palabras clave: ")
set_font(r_kw1, bold=True, italic=True)
r_kw2 = p_kw.add_run(
    "análisis de sentimientos, procesamiento de lenguaje natural, YouTube, "
    "RoBERTuito, K-Means, minería de opiniones, Python"
)
set_font(r_kw2, italic=True)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  INTRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Introducción")
paragraph(doc, "")

body(doc,
    "Las redes sociales y las plataformas de video en línea se han consolidado como "
    "los principales espacios de deliberación pública en América Latina. YouTube, con "
    "más de 2.500 millones de usuarios activos mensuales (Statista, 2024), genera "
    "volúmenes masivos de comentarios que contienen opiniones, juicios valorativos y "
    "emociones sobre los más variados temas de agenda. El análisis automatizado de "
    "estos textos mediante técnicas de Procesamiento de Lenguaje Natural (NLP) "
    "representa una oportunidad metodológica para las ciencias sociales y la analítica "
    "de datos, al permitir estudiar la opinión colectiva a escalas imposibles para el "
    "análisis manual."
)
body(doc,
    "El presente proyecto aborda el análisis de sentimientos sobre el video "
    "\"Esto es lo que nadie te cuenta de El Salvador\" (identificador: pWnndG1K5Hg), "
    "publicado en YouTube y que documenta la transformación en materia de seguridad "
    "pública en El Salvador bajo el gobierno de Nayib Bukele. Este tema genera debate "
    "polarizado en la esfera pública latinoamericana, lo que lo convierte en un caso "
    "de estudio especialmente relevante para el análisis computacional de la opinión. "
    "El corpus recolectado —4.314 comentarios de 3.950 autores únicos provenientes de "
    "múltiples países— ofrece una muestra representativa del discurso digital "
    "hispanohablante sobre gobernanza y seguridad."
)

heading2(doc, "Objetivo General")
body(doc,
    "Desarrollar e implementar un pipeline de Procesamiento de Lenguaje Natural que "
    "permita clasificar automáticamente los comentarios del video seleccionado en "
    "categorías de sentimiento (positivo, neutro y negativo), y caracterizar los "
    "patrones discursivos asociados a cada categoría mediante análisis exploratorio "
    "y clusterización temática."
)

heading2(doc, "Objetivos Específicos")
objetivos = [
    "Recolectar y almacenar el corpus de comentarios utilizando la YouTube Data API v3 y una base de datos relacional SQLite.",
    "Aplicar un pipeline de preprocesamiento que incluya limpieza textual, lematización con spaCy y tokenización con NLTK.",
    "Clasificar los comentarios en tres categorías de sentimiento mediante el modelo RoBERTuito de pysentimiento.",
    "Evaluar la coherencia de la clasificación mediante métricas no supervisadas: distribución de clases, scores de confianza, Silhouette Score y separabilidad léxica.",
    "Identificar los clusters temáticos presentes en el corpus y caracterizar el vocabulario discriminante de cada clase de sentimiento.",
]
for i, obj in enumerate(objetivos, 1):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(1.27)
    p.paragraph_format.line_spacing = Pt(24)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    r = p.add_run(f"{i}. {obj}")
    set_font(r)

heading2(doc, "Alcance")
body(doc,
    "El proyecto abarca exclusivamente los comentarios de nivel superior (sin respuestas "
    "anidadas) del video indicado, en el período comprendido entre el 1 de febrero y el "
    "17 de marzo de 2026. No se realizó anotación manual de etiquetas ni validación con "
    "ground truth humano, por lo que todas las métricas de rendimiento son de naturaleza "
    "no supervisada. El análisis se circunscribe al idioma español."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  MARCO TEÓRICO
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Marco Teórico")
paragraph(doc, "")

heading2(doc, "Análisis de Sentimientos")
body(doc,
    "El análisis de sentimientos, también denominado minería de opiniones, es la rama "
    "del NLP orientada a identificar, extraer y cuantificar estados afectivos y opiniones "
    "subjetivas expresadas en texto (Liu, 2012). Desde una perspectiva computacional, el "
    "problema se formula habitualmente como una tarea de clasificación supervisada o "
    "semi-supervisada que asigna a cada unidad textual una polaridad: positiva, negativa "
    "o neutra (Pang & Lee, 2008). Las aplicaciones abarcan desde el monitoreo de reputación "
    "corporativa y el análisis de reseñas de productos hasta el estudio de discursos "
    "políticos y la vigilancia de la salud pública en redes sociales."
)
body(doc,
    "La evolución de los enfoques metodológicos ha transitado desde los léxicos de "
    "polaridad (SentiWordNet, VADER) hasta los modelos de lenguaje pre-entrenados "
    "basados en la arquitectura Transformer (Devlin et al., 2019). Estos últimos capturan "
    "dependencias contextuales de largo alcance y han demostrado rendimiento superior en "
    "textos coloquiales y código-mixtos, característicos de las plataformas digitales."
)

heading2(doc, "Procesamiento de Lenguaje Natural")
body(doc,
    "El Procesamiento de Lenguaje Natural es el campo interdisciplinar que sitúa en la "
    "intersección de la lingüística computacional y el aprendizaje automático, con el "
    "objetivo de dotar a las máquinas de la capacidad de comprender, interpretar y generar "
    "lenguaje humano (Manning & Schütze, 1999). Un pipeline NLP típico comprende etapas "
    "de tokenización, eliminación de palabras vacías (stopwords), normalización morfológica "
    "(stemming o lematización), vectorización del texto y, finalmente, la tarea de "
    "modelamiento —en este caso, clasificación de sentimientos."
)

heading2(doc, "Tokenización y Lematización")
body(doc,
    "La tokenización es el proceso de segmentación de un flujo de caracteres en unidades "
    "mínimas de significado —tokens—, que pueden ser palabras, subpalabras o caracteres "
    "(Jurafsky & Martin, 2023). La lematización, por su parte, reduce cada forma "
    "flexionada a su lema canónico (p. ej., 'corriendo' → 'correr', 'mejores' → "
    "'bueno'), considerando el contexto morfosintáctico del token en la oración. A "
    "diferencia del stemming —que aplica reglas heurísticas de truncamiento—, la "
    "lematización produce formas lingüísticamente válidas, lo que resulta especialmente "
    "relevante para el español, lengua con alta flexión nominal y verbal."
)
body(doc,
    "En este proyecto, la tokenización se implementó con el módulo word_tokenize de NLTK "
    "(Bird et al., 2009) configurado para español, y la lematización se ejecutó con el "
    "modelo es_core_news_sm de spaCy (Honnibal & Montani, 2017), que incorpora un "
    "lematizador basado en tablas morfológicas del corpus UD-Spanish-AnCora."
)

heading2(doc, "Modelos Transformer para Análisis de Sentimientos en Español")
body(doc,
    "Los modelos de la familia BERT (Bidirectional Encoder Representations from "
    "Transformers) han establecido el estado del arte en múltiples tareas de NLP "
    "(Devlin et al., 2019). Para el español de redes sociales, Pérez et al. (2022) "
    "desarrollaron RoBERTuito, un modelo basado en RoBERTa entrenado sobre 500 millones "
    "de tweets en español de América Latina. Su arquitectura bidireccional permite "
    "capturar el significado de cada token en función de todo el contexto circundante, "
    "lo que resulta en representaciones semánticas más ricas que las de modelos "
    "unidireccionales o léxicos de polaridad fija."
)
body(doc,
    "La librería pysentimiento (Pérez et al., 2022) encapsula RoBERTuito y otros "
    "modelos especializados para tareas de análisis de texto en español e italiano, "
    "exponiendo una API de alto nivel que devuelve tanto la clase predicha (POS, NEG, "
    "NEU) como las probabilidades softmax asociadas a cada clase, lo que permite "
    "evaluar la confianza del modelo en cada predicción individual."
)

heading2(doc, "Minería de Opiniones en Plataformas Digitales")
body(doc,
    "La minería de opiniones en redes sociales y plataformas de video presenta desafíos "
    "específicos: el lenguaje es informal, abundan los emoticonos, los errores "
    "ortográficos y las expresiones idiomáticas regionales (Balahur et al., 2014). "
    "YouTube, en particular, alberga comentarios con alta variabilidad lingüística "
    "debida a su audiencia geográficamente dispersa. Para el español latinoamericano, "
    "esta heterogeneidad dialectal representa un reto adicional para los modelos "
    "entrenados en corpus monodialectales. La adopción de RoBERTuito —entrenado "
    "expresamente sobre texto de redes sociales latinoamericanas— mitiga parcialmente "
    "esta limitación."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  METODOLOGÍA
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Metodología")
paragraph(doc, "")

body(doc,
    "El proyecto adoptó un enfoque cuantitativo con diseño de investigación no "
    "experimental de tipo descriptivo-analítico. La metodología se estructuró en "
    "cinco etapas secuenciales que conforman el pipeline de procesamiento, "
    "implementado íntegramente en Python 3.13 dentro de un entorno virtual aislado. "
    "Cada etapa almacena sus resultados en bases de datos SQLite, garantizando la "
    "reproducibilidad y la persistencia incremental de los resultados."
)

heading2(doc, "Recolección de Datos")
body(doc,
    "Los comentarios fueron recolectados mediante la YouTube Data API v3 utilizando "
    "el script descarga_masiva.py. La consulta al endpoint commentThreads.list "
    "recuperó los comentarios de primer nivel del video en lotes de 100 registros, "
    "con una espera aleatoria de 1 a 2 segundos entre peticiones para respetar los "
    "límites de cuota de la API. Para garantizar la integridad de los datos ante "
    "posibles interrupciones, se implementó un mecanismo de checkpointing que persiste "
    "los registros descargados cada 500 comentarios mediante operaciones INSERT OR "
    "IGNORE sobre una tabla SQLite con restricción de unicidad sobre los campos autor "
    "y fecha. De cada comentario se almacenaron: nombre del autor, texto original, "
    "número de likes y marca temporal en formato ISO 8601."
)

heading2(doc, "Preprocesamiento del Texto")
body(doc,
    "El módulo limpieza_texto.py aplicó sobre cada comentario las siguientes "
    "transformaciones en orden: (a) conversión a minúsculas; (b) eliminación de URLs "
    "mediante expresión regular; (c) eliminación de caracteres no alfabéticos con un "
    "patrón que preserva explícitamente los caracteres del español (tildes, eñe y "
    "diéresis: [^a-záéíóúüñ\\s]); (d) eliminación de palabras vacías con el "
    "corpus de stopwords en español de NLTK; (e) lematización con el modelo "
    "es_core_news_sm de spaCy. La tokenización, ejecutada en una etapa posterior por "
    "tokenizacion.py, segmentó el texto lematizado usando nltk.word_tokenize con "
    "configuración para español, produciendo listas de tokens almacenadas como "
    "cadenas separadas por coma."
)

heading2(doc, "Extracción de Ubicaciones Geográficas")
body(doc,
    "El script extract_locations.py empleó la librería GeoText para identificar "
    "referencias a ciudades y países en el texto original de cada comentario. Se "
    "aplicó un diccionario de normalización de alias (p. ej., 'San Salvador' → "
    "'El Salvador') y un conjunto de stopwords geográficas para filtrar falsos "
    "positivos producidos por palabras comunes del español que coinciden con topónimos "
    "en inglés. Los comentarios sin ubicación detectable recibieron la etiqueta "
    "'sin info'."
)

heading2(doc, "Clasificación de Sentimientos")
body(doc,
    "La clasificación se implementó en clasificador_sentimientos.py utilizando la "
    "función create_analyzer(task='sentiment', lang='es') de pysentimiento, que "
    "carga el modelo RoBERTuito. El procesamiento se realizó en lotes de 32 comentarios "
    "(batch processing) para aprovechar la vectorización paralela del modelo Transformer, "
    "lo que reduce sustancialmente el tiempo de inferencia respecto al procesamiento "
    "secuencial. La función de predicción devuelve una etiqueta de clase (POS, NEG o "
    "NEU) y un vector de probabilidades softmax; se almacenó la clase mapeada a "
    "positivo/negativo/neutro y la probabilidad asociada a la clase predicha como "
    "indicador de confianza. Se implementó checkpointing cada 500 resultados para "
    "garantizar la persistencia ante interrupciones."
)

heading2(doc, "Herramientas y Librerías")
table_apa(doc,
    ["Librería", "Versión", "Función en el pipeline"],
    [
        ["Python", "3.13", "Lenguaje base del proyecto"],
        ["google-api-python-client", "2.193.0", "Acceso a YouTube Data API v3"],
        ["sqlite3", "stdlib", "Almacenamiento y consulta de datos"],
        ["pandas", "3.0.1", "Manipulación de datos tabulares"],
        ["NLTK", "3.9.3", "Stopwords y tokenización"],
        ["spaCy", "3.8.13", "Lematización (es_core_news_sm)"],
        ["pysentimiento", "0.7.3", "Modelo RoBERTuito de sentimientos"],
        ["scikit-learn", "1.8.0", "TF-IDF, K-Means, Silhouette Score"],
        ["matplotlib / seaborn", "3.10.8 / 0.13.2", "Visualizaciones"],
        ["WordCloud", "1.9.6", "Nubes de palabras"],
        ["GeoText", "0.4.0", "Extracción de entidades geográficas"],
    ],
    1, "Herramientas y librerías utilizadas en el pipeline"
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Resultados")
paragraph(doc, "")

heading2(doc, "Descripción del Corpus")
body(doc,
    "Se recolectaron y procesaron 4.314 comentarios de primer nivel, producidos por "
    "3.950 autores únicos —lo que indica una alta proporción de participación singular "
    "(91,6% de los autores comentó exactamente una vez). El período de actividad se "
    "extendió desde el 1 de febrero hasta el 17 de marzo de 2026, abarcando 42 días "
    "con actividad registrada. El pico de comentarios se concentró en el primer día "
    "tras la publicación del video (1 de febrero de 2026), con 1.915 comentarios en "
    "una sola jornada, lo que representa el 44,4% del corpus total."
)

table_apa(doc,
    ["Indicador", "Valor"],
    [
        ["Total de comentarios", "4.314"],
        ["Autores únicos", "3.950"],
        ["Período analizado", "1 feb. – 17 mar. 2026 (42 días)"],
        ["Pico diario de comentarios", "1.915 (1 feb. 2026)"],
        ["Longitud media del comentario (caracteres)", "193"],
        ["Longitud mediana del comentario", "119"],
        ["Comentario más largo", "4.978 caracteres"],
        ["Total de likes en el corpus", "19.277"],
        ["Comentarios con 0 likes", "3.286 (76,2%)"],
        ["Comentarios con ubicación detectada", "1.226 (28,4%)"],
    ],
    2, "Estadísticas descriptivas del corpus de comentarios"
)

heading2(doc, "Distribución de Sentimientos")
body(doc,
    "La clasificación de los 4.314 comentarios con RoBERTuito produjo la distribución "
    "presentada en la Tabla 3. El sentimiento positivo es el mayoritario (41,6%), "
    "seguido del negativo (35,5%) y el neutro (22,9%). El ratio positivo/negativo es "
    "de 1,17, lo que indica una leve preponderancia del apoyo aunque con presencia "
    "significativa de crítica. El test de bondad de ajuste χ² contra la hipótesis nula "
    "de distribución uniforme arrojó χ²(2) = 233,23, p < 0,001, lo que confirma que "
    "la distribución es significativamente no uniforme y refleja una clasificación "
    "discriminante real del modelo."
)

table_apa(doc,
    ["Sentimiento", "Frecuencia", "Porcentaje", "Likes totales", "Likes promedio"],
    [
        ["Positivo", "1.793", "41,6%", "13.831", "7,71"],
        ["Negativo", "1.531", "35,5%", "3.669", "2,40"],
        ["Neutro", "990", "22,9%", "1.777", "1,79"],
        ["Total", "4.314", "100%", "19.277", "4,47"],
    ],
    3, "Distribución de sentimientos y engagement por categoría"
)

body(doc,
    "Un hallazgo destacado es la asimetría en el engagement: los comentarios positivos "
    "acumularon el 71,7% de los likes totales (13.831), mientras que los negativos "
    "recibieron apenas el 19,0% (3.669), a pesar de representar el 35,5% del corpus. "
    "Esto sugiere que la comunidad activa que interactúa con los comentarios tiene una "
    "disposición más favorable hacia el contenido que la distribución textual global."
)
body(doc,
    "El análisis de la longitud textual también revela diferencias por categoría: "
    "los comentarios negativos son sustancialmente más largos (media: 285 caracteres, "
    "mediana: 184) que los positivos (media: 146, mediana: 101) y neutros (media: 136, "
    "mediana: 70). Este patrón es consistente con la literatura sobre comportamiento "
    "digital, donde las expresiones de desacuerdo tienden a ser más elaboradas "
    "argumentativamente (Balahur et al., 2014)."
)

heading2(doc, "Score de Confianza del Modelo")
body(doc,
    "La confianza del modelo —medida como la probabilidad softmax asignada a la clase "
    "predicha— presenta un valor medio global de 0,782 (DE = 0,169). El 65,7% de las "
    "predicciones supera el umbral de alta confianza de 0,80. Sin embargo, existe "
    "heterogeneidad notable entre clases: los comentarios positivos y negativos tienen "
    "confianza media superior a 0,83, mientras que la clase neutra presenta una "
    "confianza media de apenas 0,624. Este resultado es metodológicamente coherente: "
    "la neutralidad es la categoría más difusa semánticamente —agrupa textos ambiguos, "
    "breves o informativos sin carga valorativa explícita—, lo que genera mayor "
    "incertidumbre en el clasificador."
)

table_apa(doc,
    ["Sentimiento", "Media", "Mediana", "P25", "P75", "≥0,80 (%)", "<0,50 (%)"],
    [
        ["Positivo", "0,831", "0,898", "0,714", "0,966", "67,7%", "4,4%"],
        ["Negativo", "0,829", "0,886", "0,709", "0,963", "67,5%", "3,9%"],
        ["Neutro", "0,624", "0,615", "0,497", "0,756", "9,3%", "16,9%"],
        ["Global", "0,782", "0,830", "0,637", "0,940", "55,1%", "7,5%"],
    ],
    4, "Estadísticas del score de confianza del modelo por clase"
)

heading2(doc, "Clusterización Temática")
body(doc,
    "Para identificar los temas subyacentes del corpus se aplicó K-Means sobre la "
    "representación TF-IDF (500 términos, bigramas incluidos, TF logarítmico). "
    "El barrido de K=2 a K=8 mostró que el Silhouette Score más alto se obtiene con "
    "K=8 (SS = 0,0303), y el menor Davies-Bouldin Index con K=8 (DBI = 4,86). "
    "Si bien el Silhouette Score es bajo en términos absolutos —valor esperable en "
    "corpora de texto corto con alta dispersión léxica—, los clusters resultantes "
    "tienen interpretabilidad temática clara."
)

table_apa(doc,
    ["Cluster", "n", "Sent. dominante", "Palabras clave principales", "Tema interpretado"],
    [
        ["0", "2.240", "Negativo", "salvador, país, seguridad, cambio, gente", "Debate general sobre El Salvador"],
        ["1", "272", "Positivo", "gracias, gracias juan, juan, visitar", "Agradecimiento al reportero"],
        ["2", "88", "Neutro", "independencia, 1821, 200 años, salvador", "Contexto histórico"],
        ["3", "248", "Negativo", "precio, paz, precio paz, tranquilidad", "Debate sobre el costo de la seguridad"],
        ["4", "220", "Positivo", "video, buen video, buen, excelente", "Valoración del video"],
        ["5", "347", "Positivo", "juan, planeta, bienvenido, saludos", "Saludos al canal"],
        ["6", "621", "Positivo", "bukele, presidente, colombia, necesitamos", "Apoyo a Bukele / comparaciones"],
        ["7", "226", "Negativo", "derechos humanos, delincuentes, criminales", "Derechos humanos y crítica"],
    ],
    5, "Perfil de los ocho clusters temáticos identificados con K-Means"
)

body(doc,
    "La proyección de los clusters en el espacio bidimensional de los dos primeros "
    "componentes principales captura el 4,1% de la varianza total, valor bajo pero "
    "esperable dado el alto número de dimensiones del espacio TF-IDF. La visualización "
    "revela que los clusters de mayor densidad semántica (1, 2, 4 y 5) forman nubes "
    "más compactas, mientras el cluster 0 —el más grande y temáticamente general— "
    "presenta mayor dispersión."
)

heading2(doc, "Separabilidad Léxica por Clase de Sentimiento")
body(doc,
    "El análisis del vocabulario exclusivo por clase de sentimiento —términos presentes "
    "en una categoría pero ausentes en las otras dos— revela una separabilidad real: "
    "el 40,9% del vocabulario positivo (2.120 de 5.185 tipos únicos) es exclusivo de "
    "esa clase; en negativo el porcentaje asciende al 59,7% (5.425 de 9.085), y en "
    "neutro al 33,3% (1.352 de 4.056). La mayor riqueza léxica y exclusividad de la "
    "clase negativa es consistente con su mayor longitud textual promedio y la "
    "diversidad argumentativa característica de los textos críticos."
)
body(doc,
    "El análisis de los términos TF-IDF discriminantes por clase confirmó coherencia "
    "semántica: los comentarios positivos destacan términos como gracias, paz, dios, "
    "presidente y saludos; los negativos concentran derechos, humanos, delincuentes, "
    "criminales y precio; los neutros se caracterizan por términos descriptivos como "
    "años, independencia, datos e historia."
)

heading2(doc, "Análisis Geográfico")
body(doc,
    "De los 4.314 comentarios, 1.226 (28,4%) contienen referencias geográficas "
    "identificables. El Salvador encabeza con 653 menciones, seguido de Colombia (136), "
    "Venezuela (28), Argentina (25) y combinaciones como 'Colombia, El Salvador' (22). "
    "Esta distribución geográfica confirma el alcance transnacional del debate, con "
    "participación activa de comunidades de países con experiencias comparables de "
    "violencia y transición política."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Conclusiones")
paragraph(doc, "")

body(doc,
    "El proyecto cumplió satisfactoriamente todos los objetivos planteados. El pipeline "
    "NLP desarrollado en Python permitió recolectar, procesar y clasificar 4.314 "
    "comentarios de YouTube con una cobertura del 100% en la clasificación de "
    "sentimientos, sin recurrir a anotación manual ni ground truth supervisado. La "
    "integración de SQLite como capa de persistencia incremental y los mecanismos de "
    "checkpointing garantizaron la robustez del proceso ante posibles interrupciones."
)
body(doc,
    "Los resultados revelan que la percepción dominante sobre las políticas de "
    "seguridad de El Salvador en la audiencia digital hispanohablante es positiva "
    "(41,6%), aunque con una polarización significativa: el 35,5% de los comentarios "
    "expresa sentimiento negativo, principalmente articulado en torno al debate sobre "
    "derechos humanos y el 'precio de la paz'. La asimetría en el engagement —donde "
    "los comentarios positivos reciben 3,2 veces más likes que los negativos— sugiere "
    "que la comunidad activa de interacción tiene una disposición más favorable que "
    "el conjunto de comentaristas."
)
body(doc,
    "La confianza media del modelo (0,782) es satisfactoria para un clasificador de "
    "tres clases sobre texto informal, aunque la clase neutra presenta confianza "
    "notoriamente menor (0,624), lo que indica que la frontera entre sentimiento leve "
    "y ausencia de sentimiento es la más difusa para RoBERTuito en este corpus. "
    "La clusterización temática con K=8 aportó valor interpretativo al revelar "
    "subtemas discursivos no evidentes en el análisis de sentimiento plano: el debate "
    "sobre derechos humanos, la comparación con otros países latinoamericanos y el "
    "agradecimiento al reportero son comunidades discursivas claramente diferenciadas."
)

heading2(doc, "Limitaciones")
body(doc,
    "El proyecto presenta las siguientes limitaciones metodológicas. Primera, la "
    "ausencia de etiquetas manuales impide calcular métricas de rendimiento supervisadas "
    "(precisión, recall, F1), por lo que la evaluación del modelo depende enteramente "
    "de métricas proxy. Segunda, el Silhouette Score bajo (0,030) en la clusterización "
    "evidencia que el espacio TF-IDF de comentarios cortos tiene alta dispersión y los "
    "clusters no son geométricamente compactos, aunque sí interpretables temáticamente. "
    "Tercera, la extracción geográfica mediante GeoText tiene alta tasa de no detección "
    "(71,6% sin ubicación), dado que la mayoría de los comentaristas no mencionan su "
    "origen explícitamente. Cuarta, el análisis se limita al español; comentarios en "
    "inglés u otros idiomas no fueron identificados ni excluidos explícitamente."
)

heading2(doc, "Recomendaciones para Trabajos Futuros")
recomendaciones = [
    "Incorporar anotación manual de una muestra representativa (n ≥ 500) para calcular métricas supervisadas y validar el rendimiento de RoBERTuito en este dominio específico.",
    "Extender el pipeline al análisis de respuestas anidadas para capturar la dimensión dialógica del debate.",
    "Aplicar modelos de análisis de emociones (más allá de la polaridad) para caracterizar con mayor granularidad el espectro afectivo del corpus.",
    "Integrar análisis de aspectos (Aspect-Based Sentiment Analysis) para identificar sobre qué dimensiones específicas —seguridad, economía, derechos humanos— se expresa cada tipo de sentimiento.",
    "Explorar modelos de tópicos no supervisados (LDA, BERTopic) como alternativa a K-Means para la identificación de temas, dado que manejan mejor la polifonía temática de comentarios cortos.",
]
for i, rec in enumerate(recomendaciones, 1):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(1.27)
    p.paragraph_format.line_spacing = Pt(24)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    r = p.add_run(f"{i}. {rec}")
    set_font(r)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCIAS
# ══════════════════════════════════════════════════════════════════════════════

heading1(doc, "Referencias")
paragraph(doc, "")

refs = [
    ("Balahur, A., Hermida, J. M., & Montoyo, A. (2014).",
     "Detecting implicit expressions of sentiment in text based on commonsense knowledge. "
     "Computational Intelligence, 30(3), 558–582. https://doi.org/10.1111/coin.12019"),

    ("Bird, S., Klein, E., & Loper, E. (2009).",
     "Natural language processing with Python: Analyzing text with the natural "
     "language toolkit. O'Reilly Media."),

    ("Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).",
     "BERT: Pre-training of deep bidirectional transformers for language understanding. "
     "Proceedings of NAACL-HLT 2019, 4171–4186. https://doi.org/10.18653/v1/N19-1423"),

    ("Honnibal, M., & Montani, I. (2017).",
     "spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural "
     "networks and incremental parsing [Software]. https://spacy.io"),

    ("Jurafsky, D., & Martin, J. H. (2023).",
     "Speech and language processing: An introduction to natural language processing, "
     "computational linguistics, and speech recognition (3rd ed. draft). "
     "https://web.stanford.edu/~jurafsky/slp3/"),

    ("Liu, B. (2012).",
     "Sentiment analysis and opinion mining. Synthesis Lectures on Human Language "
     "Technologies, 5(1), 1–167. https://doi.org/10.2200/S00416ED1V01Y201204HLT016"),

    ("Manning, C. D., & Schütze, H. (1999).",
     "Foundations of statistical natural language processing. MIT Press."),

    ("Pang, B., & Lee, L. (2008).",
     "Opinion mining and sentiment analysis. Foundations and Trends in Information "
     "Retrieval, 2(1–2), 1–135. https://doi.org/10.1561/1500000011"),

    ("Pérez, J. M., Giudici, J. C., & Luque, F. (2022).",
     "pysentimiento: A Python toolkit for sentiment analysis and SocialNLP tasks. "
     "arXiv preprint arXiv:2106.09462. https://doi.org/10.48550/arXiv.2106.09462"),

    ("Statista. (2024).",
     "Number of YouTube users worldwide from 2016 to 2029. "
     "https://www.statista.com/statistics/805656/number-youtube-viewers-worldwide/"),

    ("YouTube. (2026).",
     "Esto es lo que nadie te cuenta de El Salvador [Video]. "
     "https://www.youtube.com/watch?v=pWnndG1K5Hg"),
]

for autores, resto in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent       = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)
    p.paragraph_format.line_spacing      = Pt(24)
    p.paragraph_format.space_before      = Pt(0)
    p.paragraph_format.space_after       = Pt(0)
    r1 = p.add_run(autores + " ")
    set_font(r1)
    r2 = p.add_run(resto)
    set_font(r2)

# ── Guardar ────────────────────────────────────────────────────────────────────
ruta = "../informe_analisis_sentimientos.docx"
doc.save(ruta)
print(f"✅ Informe generado: {ruta}")
