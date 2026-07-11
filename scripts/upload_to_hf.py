"""Release publisher: push Atlas weights + fact index to the Hugging Face Hub.

Run on the machine that holds the trained artifacts (the GPU box):

    # full release (all four artifacts)
    python scripts/upload_to_hf.py \
        --retriever-ckpt path/to/retriever.pt \
        --composer-ckpt  path/to/composer.pt \
        --encoded        path/to/msmarco_encoded.pt \
        --para-targets   path/to/para_targets.pt \
        --manifest       path/to/manifest.json \
        --tag v0.1.1

    # partial release: only upload what changed, tag everything
    python scripts/upload_to_hf.py \
        --composer-ckpt path/to/composer.pt \
        --manifest      path/to/manifest.json \
        --para-targets  path/to/para_targets.pt \
        --tag v0.1.1

    # tag-only (no uploads): pin the current repo state under a new tag
    python scripts/upload_to_hf.py --tag v0.1.1

Every file argument is optional; files not given are skipped (the tag then
pins whatever is already at HEAD in that repo). BOTH repos are always tagged
so `Atlas.from_pretrained()`'s version pinning resolves.

Requires a logged-in HF account with write access (`hf auth login`).
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

DEFAULT_MODEL_REPO = "kirmada-jsr/atlas"
DEFAULT_INDEX_REPO = "kirmada-jsr/atlas-index"

# Canonical filenames the pipeline downloads (see atlas/pipeline.py).
MODEL_FILES = {
    "retriever_ckpt": "retriever.pt",
    "composer_ckpt": "composer.pt",
    "manifest": "manifest.json",
}
INDEX_FILES = {
    "encoded": "msmarco_encoded.pt",
    "para_targets": "para_targets.pt",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Upload Atlas artifacts to the HF Hub.")
    p.add_argument("--retriever-ckpt", help="Path to atlas_sonar_retriever.pt")
    p.add_argument("--composer-ckpt", help="Path to atlas_sonar_composer.pt")
    p.add_argument("--manifest", help="Path to manifest.json (component versions)")
    p.add_argument("--encoded", help="Path to msmarco_encoded.pt (fact pool)")
    p.add_argument("--para-targets", help="Path to para_targets.pt")
    p.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--index-repo", default=DEFAULT_INDEX_REPO)
    p.add_argument("--tag", required=True, help="Release tag, e.g. v0.1.1 (must match the package version)")
    p.add_argument("--private", action="store_true", help="Create the HF repos as private")
    args = p.parse_args()

    given = {
        "retriever_ckpt": args.retriever_ckpt,
        "composer_ckpt": args.composer_ckpt,
        "manifest": args.manifest,
        "encoded": args.encoded,
        "para_targets": args.para_targets,
    }
    paths = {}
    for name, path in given.items():
        if path is None:
            print(f"  {name:16s} (not given: skipped; tag pins existing HEAD)")
            continue
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{name.replace('_', '-')}: not found: {path}")
        paths[name] = path
        print(f"  {name:16s} {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    api = HfApi()
    user = api.whoami()["name"]
    print(f"\n  Logged in as: {user}")
    print(f"  Model repo:   {args.model_repo}")
    print(f"  Index repo:   {args.index_repo}")
    print(f"  Tag:          {args.tag}\n")

    # --- create repos (idempotent) ---
    api.create_repo(args.model_repo, repo_type="model", private=args.private, exist_ok=True)
    api.create_repo(args.index_repo, repo_type="dataset", private=args.private, exist_ok=True)

    # --- upload model files (only those given) ---
    for arg_name, repo_filename in MODEL_FILES.items():
        if arg_name not in paths:
            continue
        print(f"  Uploading {paths[arg_name]} -> {args.model_repo}/{repo_filename}")
        api.upload_file(
            path_or_fileobj=paths[arg_name],
            path_in_repo=repo_filename,
            repo_id=args.model_repo,
            repo_type="model",
        )

    # --- upload fact index files (only those given) ---
    for arg_name, repo_filename in INDEX_FILES.items():
        if arg_name not in paths:
            continue
        print(f"  Uploading {paths[arg_name]} -> {args.index_repo}/{repo_filename}  "
              f"(large file — this can take a while)")
        api.upload_file(
            path_or_fileobj=paths[arg_name],
            path_in_repo=repo_filename,
            repo_id=args.index_repo,
            repo_type="dataset",
        )

    # --- tag both repos so version pinning resolves ---
    for repo_id, repo_type in ((args.model_repo, "model"), (args.index_repo, "dataset")):
        print(f"  Tagging {repo_id} @ {args.tag}")
        api.create_tag(repo_id, tag=args.tag, repo_type=repo_type, exist_ok=True)

    print("\n  Done. Verify with:")
    print(f"    python -c \"from atlas import Atlas; Atlas.from_pretrained('{args.model_repo}', '{args.index_repo}')\"")


if __name__ == "__main__":
    main()
