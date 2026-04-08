import json

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Notebook Format: {nb.get('nbformat')}.{nb.get('nbformat_minor')}")
    
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    
    print(f"Total cells: {len(cells)}")
    print(f"Code cells: {len(code_cells)}")
    print(f"Markdown cells: {len(markdown_cells)}")
    
    print("\n--- Summary of Code Cells ---")
    for i, cell in enumerate(code_cells):
        source = "".join(cell.get('source', []))
        # Print first line of each code cell to see what it's doing
        first_line = source.split('\n')[0] if source else ""
        print(f"Cell {i}: {first_line[:100]}...")
        
        # Look for model definition
        if "Sequential" in source or "Model(" in source or "layers." in source:
            print(f"\n[FOUND MODEL DEFINITION IN CELL {i}]")
            print(source)
            print("-" * 40)

if __name__ == "__main__":
    analyze_notebook('e:/gajalakshmi/project/Suhwa/mutemotion-wlasl-translation-model.ipynb')
