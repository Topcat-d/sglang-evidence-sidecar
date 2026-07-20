"""Command-line verifier for Smoke Attestation Transcript v0 JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglang.srt.evidence_sidecar.abi_v0 import verify_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    try:
        result = verify_jsonl(args.transcript)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
