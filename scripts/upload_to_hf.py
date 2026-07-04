"""One-time publisher: push Atlas weights + fact index to the Hugging Face Hub.

Run on the machine that holds the trained artifacts (the GPU box):

    python scripts/upload_to_hf.py \
        --retriever-ckpt ~/Atlas/atlas/experiments/atlas_v5.1/retriever-checkpoints/atlas_sonar_retriever.pt \
        --composer-ckpt  ~/Atlas/atlas/experiments/atlas_v5.1/atlas_sonar_composer.pt \
        --encoded        ~/Atlas/atlas/experiments/atlas_v5.1/atlas_sonar_data_400k/msmarco_encoded.pt \
        --para-targets   ~/Atlas/atlas/experiments/atlas_v5.1/atlas_sonar_data_400k/para_targets.pt \
        --tag v0.1.0

Requires a logged-in HF account with write access (`hf auth login`).
Creates the repos if needed, uploads under the canonical in-repo filenames,
then tags BOTH repos so `Atlas.from_pretrained()`'s version pinning resolves.
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
}
INDEX_FILES = {
    "encoded": "msmarco_encoded.pt",
    "para_targets": "para_targets.pt",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Upload Atlas artifacts to the HF Hub.")
    p.add_argument("--retriever-ckpt", required=True, help="Path to atlas_sonar_retriever.pt")
    p.add_argument("--composer-ckpt", required=True, help="Path to atlas_sonar_composer.pt")
    p.add_argument("--encoded", required=True, help="Path to msmarco_encoded.pt (fact pool)")
    p.add_argument("--para-targets", required=True, help="Path to para_targets.pt")
    p.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--index-repo", default=DEFAULT_INDEX_REPO)
    p.add_argument("--tag", required=True, help="Release tag, e.g. v0.1.0 (must match the package version)")
    p.add_argument("--private", action="store_true", help="Create the HF repos as private")
    args = p.parse_args()

    paths = {
        "retriever_ckpt": os.path.expanduser(args.retriever_ckpt),
        "composer_ckpt": os.path.expanduser(args.composer_ckpt),
        "encoded": os.path.expanduser(args.encoded),
        "para_targets": os.path.expanduser(args.para_targets),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{name.replace('_', '-')}: not found: {path}")
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

    # --- upload model weights ---
    for arg_name, repo_filename in MODEL_FILES.items():
        print(f"  Uploading {paths[arg_name]} -> {args.model_repo}/{repo_filename}")
        api.upload_file(
            path_or_fileobj=paths[arg_name],
            path_in_repo=repo_filename,
            repo_id=args.model_repo,
            repo_type="model",
        )

    # --- upload fact index ---
    for arg_name, repo_filename in INDEX_FILES.items():
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
