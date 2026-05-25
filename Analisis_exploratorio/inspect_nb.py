import json
with open('notebooks/04_evaluacion_modelo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        outputs = cell.get('outputs', [])
        print(f"Cell length: {len(source)}, Outputs: {len(outputs)}")
