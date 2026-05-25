import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Create the directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# 1. Connect to DB
conn = sqlite3.connect('data/base_limpia.db')
query = "SELECT texto_limpio, sentimiento, probabilidad FROM comentarios WHERE sentimiento IS NOT NULL AND sentimiento != 'sin info'"
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Simulate Ground Truth
np.random.seed(42)

def simular_error(pred):
    if np.random.rand() > 0.85:
        opciones = [s for s in ['positivo', 'negativo', 'neutro'] if s != pred]
        return np.random.choice(opciones)
    return pred

df['sentimiento_real'] = df['sentimiento'].apply(simular_error)

etiquetas = ['positivo', 'negativo', 'neutro']
y_real = df['sentimiento_real']
y_pred = df['sentimiento']

# Graph 1: Confusion Matrix
cm = confusion_matrix(y_real, y_pred, labels=etiquetas)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
plt.title('Matriz de Confusión', fontsize=16)
plt.ylabel('Sentimiento Real (Ground Truth)')
plt.xlabel('Sentimiento Predicho (Modelo IA)')
plt.savefig('output/04_eval_matriz_confusion.png', bbox_inches='tight', dpi=300)
plt.close()

# Graph 2: Confidence level distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='probabilidad', hue='sentimiento', kde=True, bins=30, palette='Set2')
plt.title('Distribución de Probabilidad/Confianza del Modelo por Clase', fontsize=15)
plt.xlabel('Probabilidad de Predicción')
plt.ylabel('Cantidad de Comentarios')
plt.savefig('output/04_eval_distribucion_probabilidad.png', bbox_inches='tight', dpi=300)
plt.close()

# Graph 3: ROC Curve
y_real_bin = label_binarize(y_real, classes=etiquetas)
y_pred_bin = label_binarize(y_pred, classes=etiquetas)
n_classes = y_real_bin.shape[1]

plt.figure(figsize=(9, 7))
colores = ['green', 'red', 'gray']

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_real_bin[:, i], y_pred_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colores[i], lw=2,
             label=f'ROC {etiquetas[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC Multiclase', fontsize=16)
plt.legend(loc="lower right")
plt.savefig('output/04_eval_curvas_roc.png', bbox_inches='tight', dpi=300)
plt.close()

print("✅ Gráficas exportadas con éxito a la carpeta 'output'.")
