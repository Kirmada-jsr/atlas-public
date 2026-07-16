"""Atlas command-line interface.

    atlas ask "what is dobutamine used for?"
    atlas ask "how do vaccines work?" --mode verbose
    atlas repl
    atlas serve            # Gradio demo (pip install atlas-sonar[demo])

Loading flags (shared by all subcommands) mirror ``Atlas.from_pretrained``;
per-question flags mirror ``Atlas.ask``. Defaults are the validated ones.

Two output modes: ``qa`` (default) prints only the answer paragraph;
``verbose`` shows every pipeline stage (retrieved top-k, distinct facts
after dedup, groups, and the per-group fused sentences). The repl toggles
between them live with ``:v``.
"""

from __future__ import annotations

import argparse
import textwrap
import time

from .pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_FUSE_THRESHOLD,
    DEFAULT_INDEX_REPO,
    DEFAULT_K,
    DEFAULT_MAX_GROUP,
    DEFAULT_MODEL_REPO,
    Atlas,
    AtlasResult,
)

WRAP = 100


def _w(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=WRAP,
                         initial_indent=indent, subsequent_indent=indent)


def _div(c: str = "─") -> str:
    return c * WRAP


def _print_qa(result: AtlasResult) -> None:
    print()
    print(_w(result.answer or "", indent="  "))


def _print_verbose(result: AtlasResult, elapsed: float) -> None:
    """Render every pipeline stage, same layout as the validated dev harness."""
    print(_div())
    print(f"  TOP-{len(result.retrieved)} RETRIEVED SENTENCES")
    print(_div())
    for rank, (sentence, score) in enumerate(result.retrieved, start=1):
        print(f"  Rank {rank}  score={score:.4f}")
        print(_w(sentence))

    print()
    print(_div())
    print(f"  {len(result.deduped)} DISTINCT FACTS (cosine dedup)")
    print(_div())
    for s in result.deduped:
        print(_w(s))

    print()
    print(_div())
    print(f"  GROUPING -> {len(result.groups)} GROUPS")
    print(_div())
    for grp, sent in zip(result.groups, result.sentences):
        tag = f"fuse{grp}" if len(grp) > 1 else f"solo[{grp[0]}]"
        print(f"  {tag}:")
        print(_w(sent))

    if result.answer is not None:
        print()
        print(_div())
        print("  ANSWER")
        print(_div())
        print(_w(result.answer, indent="  "))

    print(f"\n  ({elapsed:.1f}s)")


def _ask_and_print(atlas: Atlas, question: str, args: argparse.Namespace, mode: str) -> None:
    t0 = time.time()
    result = atlas.ask(
        question,
        k=args.k,
        dedup_threshold=args.dedup_threshold,
        fuse_threshold=args.fuse_threshold,
        max_group=args.max_group,
    )
    if mode == "verbose":
        _print_verbose(result, time.time() - t0)
    else:
        _print_qa(result)


def _load(args: argparse.Namespace) -> Atlas:
    return Atlas.from_pretrained(
        model_repo=args.model_repo,
        index_repo=args.index_repo,
        revision=args.revision,
        device=args.device,
        pool_device=args.pool_device,
        chunk_size=args.chunk_size,
    )


def _cmd_ask(args: argparse.Namespace) -> None:
    atlas = _load(args)
    _ask_and_print(atlas, args.question, args, args.mode)


def _cmd_version(args: argparse.Namespace) -> None:
    from . import __version__

    print(f"atlas-sonar {__version__}")
    try:
        import json

        from huggingface_hub import hf_hub_download

        from .pipeline import MANIFEST_FILE, _default_revision

        path = hf_hub_download(
            args.model_repo, MANIFEST_FILE, revision=_default_revision()
        )
        with open(path) as f:
            manifest = json.load(f)
        for name, c in manifest.get("components", {}).items():
            print(f"  {name}: {c.get('version', '?')}  ({c.get('source', '')})")
    except Exception:
        print("  (component manifest unavailable for this revision)")


def _cmd_repl(args: argparse.Namespace) -> None:
    atlas = _load(args)
    mode = args.mode
    print("\n" + "=" * WRAP)
    print("  Atlas Interactive Mode")
    print("  Type a question and press Enter. ':v' toggles verbose, 'quit' exits.")
    print("=" * WRAP)

    while True:
        try:
            question = input("\natlas> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not question:
            continue
        if question.lower() in {":q", "quit", "exit", "q"}:
            print("  Exiting.")
            break
        if question.lower() == ":v":
            mode = "verbose" if mode == "qa" else "qa"
            print(f"  [mode={mode}]")
            continue

        _ask_and_print(atlas, question, args, mode)


def build_demo(atlas: Atlas | None = None, **load_kwargs):
    """Build the Gradio demo (used by `atlas serve` and app.py)."""
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit(
            "Gradio is not installed. Run: pip install 'atlas-sonar[demo]'"
        ) from e

    if atlas is None:
        atlas = Atlas.from_pretrained(**load_kwargs)

    def _answer(question: str, k: int):
        if not question.strip():
            return "", "", ""
        result = atlas.ask(question.strip(), k=int(k))
        retrieved = "\n\n".join(
            f"[{i+1}] (score={s:.4f})  {sent}"
            for i, (sent, s) in enumerate(result.retrieved)
        )
        groups = "\n\n".join(
            f"[{'fuse' if len(grp) > 1 else 'solo'} {grp}]  {sent}\n"
            + "\n".join(f"    <- {result.deduped[i]}" for i in grp)
            for grp, sent in zip(result.groups, result.sentences)
        )
        return result.answer or "", groups, retrieved

    with gr.Blocks(title="Atlas") as demo:
        gr.Markdown("# Atlas\nRetrieval, grouping and fusion in SONAR embedding space.")
        question = gr.Textbox(label="Question", placeholder="what is dobutamine used for?")
        k = gr.Slider(1, 16, value=DEFAULT_K, step=1, label="k (retrieved sentences)")
        answer = gr.Textbox(label="Answer")
        groups = gr.Textbox(label="Groups (fused sentence <- source facts)", lines=8)
        retrieved = gr.Textbox(label="Retrieved sentences", lines=8)
        question.submit(_answer, [question, k], [answer, groups, retrieved])

    return demo


def _cmd_serve(args: argparse.Namespace) -> None:
    demo = build_demo(
        model_repo=args.model_repo,
        index_repo=args.index_repo,
        revision=args.revision,
        device=args.device,
        pool_device=args.pool_device,
        chunk_size=args.chunk_size,
    )
    demo.launch()


def main() -> None:
    p = argparse.ArgumentParser(prog="atlas", description="Atlas: ask questions against a pretrained SONAR fact memory.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, with_ask_opts: bool = True) -> None:
        sp.add_argument("--model-repo", default=DEFAULT_MODEL_REPO,
                        help=f"HF model repo (default {DEFAULT_MODEL_REPO})")
        sp.add_argument("--index-repo", default=DEFAULT_INDEX_REPO,
                        help=f"HF dataset repo with the fact pool (default {DEFAULT_INDEX_REPO})")
        sp.add_argument("--revision", default=None,
                        help="HF revision (default: tag matching this package version)")
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (default auto)")
        sp.add_argument("--pool-device", default=None,
                        help="Keep the fact pool resident on this device (e.g. cuda). Default: CPU, streamed.")
        sp.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Sentences per scoring chunk (default {DEFAULT_CHUNK_SIZE})")
        if with_ask_opts:
            sp.add_argument("--mode", choices=["qa", "verbose"], default="qa",
                            help="qa prints only the answer; verbose shows every pipeline stage")
            sp.add_argument("--k", type=int, default=DEFAULT_K,
                            help=f"Retrieved sentences (default {DEFAULT_K})")
            sp.add_argument("--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD,
                            help="Cosine similarity at or above which retrieved facts "
                                 f"collapse as near-duplicates (default {DEFAULT_DEDUP_THRESHOLD})")
            sp.add_argument("--fuse-threshold", type=float, default=DEFAULT_FUSE_THRESHOLD,
                            help="Fusability probability at or above which two facts may "
                                 f"share one output sentence (default {DEFAULT_FUSE_THRESHOLD})")
            sp.add_argument("--max-group", type=int, default=DEFAULT_MAX_GROUP,
                            help=f"Maximum facts fused into one sentence (default {DEFAULT_MAX_GROUP})")

    sp_ask = sub.add_parser("ask", help="Ask a single question")
    sp_ask.add_argument("question", help="The question to ask")
    add_common(sp_ask)
    sp_ask.set_defaults(func=_cmd_ask)

    sp_repl = sub.add_parser("repl", help="Interactive question loop")
    add_common(sp_repl)
    sp_repl.set_defaults(func=_cmd_repl)

    sp_serve = sub.add_parser("serve", help="Launch the Gradio demo")
    add_common(sp_serve, with_ask_opts=False)
    sp_serve.set_defaults(func=_cmd_serve)

    sp_version = sub.add_parser("version", help="Show package + component versions")
    sp_version.add_argument("--model-repo", default=DEFAULT_MODEL_REPO,
                            help=f"HF model repo to read the manifest from (default {DEFAULT_MODEL_REPO})")
    sp_version.set_defaults(func=_cmd_version)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
