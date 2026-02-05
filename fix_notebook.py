import json

# Correcting the JSON string format by escaping backslashes and quotes
notebook_path = 'notebooks/dodo.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell.get('id') == '6843b82b':
        cell['source'] = [
            "import asyncio\n",
            "from langchain_community.document_loaders import PlaywrightURLLoader\n",
            "\n",
            "urls = ['https://dodopizza.ru/moscow']\n",
            "loader = PlaywrightURLLoader(urls=urls, remove_selectors=[\"header\", \"footer\"])\n",
            "data = await loader.aload()"
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
    elif cell.get('id') == 'd6d5f45b':
        # Remove WebBaseLoader import if it exists
        cell['source'] = [line for line in cell['source'] if 'WebBaseLoader' not in line]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook corrected successfully.")
