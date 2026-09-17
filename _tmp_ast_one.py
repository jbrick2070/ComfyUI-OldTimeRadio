import ast
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    ast.parse(fh.read())
print("AST_OK", path)
