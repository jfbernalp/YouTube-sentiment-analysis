import json
import base64
import os

notebook_path = 'notebooks/04_evaluacion_modelo.ipynb'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

image_count = 1
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                # Sometimes it's a list of strings, sometimes a single string
                if isinstance(img_data, list):
                    img_data = ''.join(img_data)
                
                # decode and save
                img_bytes = base64.b64decode(img_data)
                filepath = os.path.join(output_dir, f'evaluacion_modelo_grafica_{image_count}.png')
                with open(filepath, 'wb') as img_f:
                    img_f.write(img_bytes)
                print(f'Saved {filepath}')
                image_count += 1
