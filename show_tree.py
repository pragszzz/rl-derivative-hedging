import os

EXCLUDE_DIRS = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints'}

def list_dir_tree(start_path='.', max_depth=4):
    start_path = os.path.abspath(start_path)
    for root, dirs, files in os.walk(start_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        level = root.replace(start_path, "").count(os.sep)
        if level > max_depth:
            continue

        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")

        sub_indent = " " * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

if __name__ == "__main__":
    list_dir_tree(".")