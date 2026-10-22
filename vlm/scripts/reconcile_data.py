from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import unicodedata


def filename_stem(media_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", media_name).strip("_")
    return unicodedata.normalize("NFKD", safe_name).encode("ascii", "ignore").decode()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mediascope.dataset import prepare_dataset

    parser = argparse.ArgumentParser(description='Prepare the supplied train/test dataset manifests')
    parser.add_argument('--data-dir', type=Path, default=Path('../data/mediascope-clustered'))
    parser.add_argument('--output-dir', type=Path, default=Path('work/dataset'))
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.data_dir, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
