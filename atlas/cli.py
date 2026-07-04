"""Atlas command-line interface.

    atlas ask "what is dobutamine used for?"
    atlas repl
    atlas serve            # Gradio demo (pip install atlas-sonar[demo])

Loading flags (shared by all subcommands) mirror ``Atlas.from_pretrained``;
per-question flags mirror ``Atlas.ask``. Defaults are the validated ones.
"""

from __future__ import annotations

import argparse
import textwrap

import torch
import torch.nn.functional as F

from .pipeline import DEFAULT_INDEX_REPO, DEFAULT_MODEL_REPO, Atlas, AtlasResult

WRAP = 100


def _w(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=WRAP,
                         initial_indent=indent, subsequent_indent=indent)


def _div(c: str = "─") -> str:
    return c * WRAP


def _short(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n] + "…"


def _print_result(atlas: Atlas, result: AtlasResult, k: int, n_neighbors: int) -> None:
    """Render one ask() result — same layout as the validated dev interactive mode."""
    print(_div("─"))
    print(f"  TOP-{k} RETRIEVED SENTENCES")
    print(_div("─"))
    for rank, (sentence, score) in enumerate(result.retrieved, start=1):
        print(f"  Rank {rank}  score={score:.4f}")
        print(_w(sentence))

    if result.answer is not None:
        print()
        print(_div("─"))
        print("  COMPOSER DECODE")
        print(_div("─"))
        print(_w(result.answer))

    print()
    print(_div("─"))
    print("  NEAREST PASSAGES TO COMPOSED OUTPUT")
    print(_div("─"))
    for rank, p in enumerate(result.passages, start=1):
        print(f"\n  {rank}. Similarity={p['similarity']:.4f}")
        print(f"     Q: {_short(p['question'], 120)}")
        for s in p["sentences"]:
            print(_w(s, indent="     "))

    print()
    print(_div("─"))
    comp_to_pool_sims = F.cosine_similarity(
        F.normalize(result.embedding.unsqueeze(0), dim=-1),
        F.normalize(atlas.sent_embs[torch.tensor(result.retrieved_indices)].cpu(), dim=-1),
    )
    print("  Composed vector similarity to retrieved sentences: "
          + "  ".join(f"s{i+1}={v:.3f}" for i, v in enumerate(comp_to_pool_sims.tolist())))


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
    result = atlas.ask(
        args.question,
        k=args.k,
        score_mode=args.score_mode,
        n_neighbors=args.n_neighbors,
        decode=not args.no_decode,
    )
    print()
    _print_result(atlas, result, args.k, args.n_neighbors)


def _cmd_repl(args: argparse.Namespace) -> None:
    atlas = _load(args)
    print("\n" + "=" * WRAP)
    print("  Atlas Interactive Mode")
    print("  Type a question and press Enter. Type 'quit' to exit.")
    print("=" * WRAP)

    while True:
        try:
            question = input("\n  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("  Exiting.")
            break

        print()
        result = atlas.ask(
            question,
            k=args.k,
            score_mode=args.score_mode,
            n_neighbors=args.n_neighbors,
            decode=not args.no_decode,
        )
        _print_result(atlas, result, args.k, args.n_neighbors)


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

    def _answer(question: str, k: int, score_mode: str):
        if not question.strip():
            return "", "", ""
        result = atlas.ask(question.strip(), k=int(k), score_mode=score_mode)
        retrieved = "\n\n".join(
            f"[{i+1}] (score={s:.4f})  {sent}"
            for i, (sent, s) in enumerate(result.retrieved)
        )
        passages = "\n\n".join(
            f"[{i+1}] (sim={p['similarity']:.4f})  Q: {p['question']}\n"
            + "\n".join(f"    {s}" for s in p["sentences"])
            for i, p in enumerate(result.passages)
        )
        return result.answer or "", retrieved, passages

    with gr.Blocks(title="Atlas") as demo:
        gr.Markdown("# Atlas\nRetrieval + composition in SONAR embedding space.")
        question = gr.Textbox(label="Question", placeholder="what is dobutamine used for?")
        with gr.Row():
            k = gr.Slider(1, 10, value=3, step=1, label="k (retrieved sentences)")
            score_mode = gr.Radio(["uniform", "retriever"], value="uniform", label="score mode")
        answer = gr.Textbox(label="Composed answer (SONAR decode)")
        retrieved = gr.Textbox(label="Retrieved sentences", lines=6)
        passages = gr.Textbox(label="Nearest stored passages", lines=10)
        question.submit(_answer, [question, k, score_mode], [answer, retrieved, passages])

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
    p = argparse.ArgumentParser(prog="atlas", description="Atlas — ask questions against a pretrained SONAR fact memory.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, with_ask_opts: bool = True) -> None:
        sp.add_argument("--model-repo", default=DEFAULT_MODEL_REPO,
                        help=f"HF model repo (default {DEFAULT_MODEL_REPO})")
        sp.add_argument("--index-repo", default=DEFAULT_INDEX_REPO,
                        help=f"HF dataset repo with the fact index (default {DEFAULT_INDEX_REPO})")
        sp.add_argument("--revision", default=None,
                        help="HF revision (default: tag matching this package version)")
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (default auto)")
        sp.add_argument("--pool-device", default=None,
                        help="Keep the fact pool resident on this device (e.g. cuda). Default: CPU, streamed.")
        sp.add_argument("--chunk-size", type=int, default=4096,
                        help="Sentences per scoring chunk (default 4096)")
        if with_ask_opts:
            sp.add_argument("--k", type=int, default=3, help="Retrieved sentences (default 3)")
            sp.add_argument("--score-mode", choices=["uniform", "retriever"], default="uniform",
                            help="Composer score conditioning (default uniform — validated)")
            sp.add_argument("--n-neighbors", type=int, default=5,
                            help="Nearest stored passages to show (default 5)")
            sp.add_argument("--no-decode", action="store_true",
                            help="Skip SONAR decoding (retrieval + embedding only)")

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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
