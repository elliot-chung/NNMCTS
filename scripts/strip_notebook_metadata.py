"""Remove execution metadata and Colab identity fields from a Jupyter notebook."""
import argparse
import json
from pathlib import Path


def strip_notebook(path: Path) -> None:
  notebook = json.loads(path.read_text(encoding="utf-8"))
  notebook.get("metadata", {}).pop("colab", None)

  for cell in notebook.get("cells", []):
    cell.pop("execution_count", None)
    cell["outputs"] = []
    metadata = cell.get("metadata", {})
    metadata.pop("executionInfo", None)
    metadata.pop("id", None)
    metadata.pop("colab", None)

  path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("notebook", type=Path, nargs="?", default=Path("MCTS.ipynb"))
  args = parser.parse_args()
  strip_notebook(args.notebook)
  print(f"Stripped metadata from {args.notebook}")


if __name__ == "__main__":
  main()
