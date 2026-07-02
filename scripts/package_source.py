import argparse
import zipfile
from pathlib import Path

EXCLUDE_DIR_NAMES = {
  ".git",
  ".venv",
  "venv",
  "__pycache__",
  "node_modules",
  "cdk.out",
  "dist",
  "smoke_local",
}

EXCLUDE_EXTENSIONS = {".pkl", ".pt"}


def should_skip(path: Path) -> bool:
  for part in path.parts:
    if part in EXCLUDE_DIR_NAMES:
      return True
  return path.suffix in EXCLUDE_EXTENSIONS


def package_repo(repo_root: Path, output_path: Path) -> None:
  if output_path.exists():
    output_path.unlink()

  with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file_path in sorted(repo_root.rglob("*")):
      if not file_path.is_file():
        continue
      relative = file_path.relative_to(repo_root)
      if should_skip(relative):
        continue
      archive.write(file_path, relative.as_posix())


def main() -> None:
  parser = argparse.ArgumentParser(description="Package NNMCTS source for cloud GPU training.")
  parser.add_argument(
    "--repo-root",
    type=Path,
    default=Path(__file__).resolve().parents[1],
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "dist" / "nnmcts-source.zip",
  )
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  package_repo(args.repo_root, args.output)
  print(args.output)


if __name__ == "__main__":
  main()
