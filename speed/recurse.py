import os

def print_tree(start_path, max_depth=3):
    start_path = os.path.abspath(start_path)
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        if level > max_depth:
            continue
        
        indent = '│   ' * level
        print(f"{indent}├── {os.path.basename(root)}/")
        
        subindent = '│   ' * (level + 1)
        for d in dirs:
            if d not in ['.venv', 'node_modules', '__pycache__']:
                print(f"{subindent}├── {d}/")

# Change '.' if needed
print_tree('clean_clips', max_depth=1)