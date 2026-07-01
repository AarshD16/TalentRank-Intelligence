"""Build a RoleSpec YAML from a job_description.docx file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.role_parser import parse_rolespec_from_docx
from src.rolespec import save_rolespec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RoleSpec YAML from a JD docx.")
    parser.add_argument("--jd", default=Path("job_description.docx"), type=Path)
    parser.add_argument("--out", default=Path("configs/rolespec_redrob_ai_engineer.yaml"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rolespec = parse_rolespec_from_docx(args.jd)
    save_rolespec(args.out, rolespec)
    print(f"Wrote RoleSpec to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
