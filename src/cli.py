"""Command-line interface for the manga colorizer.

Examples
--------
# Check model availability
python src/cli.py check

# Colorize a single image (default engine = comicnet)
python src/cli.py colorize input/page_01.png output/page_01.png

# Colorize a whole chapter, then build a CBZ and a PDF
python src/cli.py chapter input/ output/ --cbz out/chapter1.cbz --pdf out/chapter1.pdf

# Just batch-colorize a folder
python src/cli.py batch input/ output/

# Pick the engine explicitly
python src/cli.py batch input/ output/ --engine comicnet
python src/cli.py batch input/ output/ --engine zhang
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorize import load_settings  # noqa: E402
from batch import build_engine, colorize_folder, colorize_input, colorize_archive  # noqa: E402


def _apply_overrides(settings: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "engine", None):
        settings["engine"] = args.engine
    if getattr(args, "long_side", None):
        settings["long_side"] = args.long_side
    return settings


# Recognised output kinds.
OUTPUT_KINDS = ("image", "cbz", "pdf")


def _parse_outputs(value) -> set[str]:
    """Normalise a --out value into a set of output kinds.

    Accepts a comma-separated string ("image,cbz"), a list, or "all".
    Unknown tokens raise a ValueError. An empty selection defaults to image.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        tokens = [str(v) for v in value]
    else:
        tokens = str(value).split(",")

    out: set[str] = set()
    for tok in tokens:
        t = tok.strip().lower()
        if not t:
            continue
        if t == "all":
            return set(OUTPUT_KINDS)
        if t in ("img", "images", "png"):
            t = "image"
        if t not in OUTPUT_KINDS:
            raise ValueError(
                f"Unknown output '{tok}'. Choose from: image, cbz, pdf (or 'all')."
            )
        out.add(t)
    return out if out else {"image"}


def _resolve_outputs(settings: dict, args: argparse.Namespace) -> set[str]:
    """Determine which outputs to produce (CLI --out overrides settings)."""
    raw = getattr(args, "out", None)
    if raw is None:
        raw = settings.get("outputs", ["image", "cbz", "pdf"])
    return _parse_outputs(raw)


def _cmd_check(args: argparse.Namespace) -> int:
    settings = load_settings()
    root = Path(__file__).resolve().parent.parent
    engine = (settings.get("engine", "mangacolv2") or "mangacolv2").lower()
    print(f"engine    : {engine}")

    req = []
    opt = []
    # Required: weights for the active engine.
    if engine == "mangacolv2":
        base = root / settings.get("mangacolv2_model_dir", "models/mangacolv2")
        req.append(base / settings.get("mangacolv2_generator", "generator.zip"))
        req.append(base / settings.get("mangacolv2_denoiser", "net_rgb.pth"))
    elif engine == "comicnet":
        base = root / settings.get("comicnet_model_dir", "models/comicnet")
        req.append(base / settings.get("comicnet_model_file", "colorizer.pth"))
    elif engine == "zhang":
        req.append(root / settings.get("model_dir", "models")
                   / settings.get("model_file", "colorization_release_v2.pth"))

    # Optional: upscale model when enabled.
    if settings.get("upscale", False):
        opt.append(root / settings.get("upscale_model",
                                       "models/upscale/RealESRGAN_x2plus.onnx"))

    ok = True
    for f in req:
        if f.exists() and f.stat().st_size > 100_000:
            print(f"  [OK     ] {f.name} ({f.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  [MISSING] {f.name}  ({f})")
            ok = False
    for f in opt:
        if f.exists() and f.stat().st_size > 100_000:
            print(f"  [OK     ] {f.name} ({f.stat().st_size/1e6:.1f} MB)  [upscale]")
        else:
            print(f"  [MISSING] {f.name}  [upscale]  ({f})")
            ok = False

    if not ok:
        print(f"\nRun: python scripts/download_model.py --engine {engine}")
        print("     python scripts/download_model.py --engine upscale")
        return 1
    print("\nModel files present. Ready.")
    return 0


def _cmd_colorize(args: argparse.Namespace) -> int:
    settings = _apply_overrides(load_settings(), args)
    colorizer = build_engine(settings)
    ok = colorizer.colorize_file(args.input, args.output)
    print(f"{'done' if ok else 'failed'} -> {args.output}")
    return 0 if ok else 1


def _emit_outputs(settings: dict, args: argparse.Namespace,
                  produced: list, want: set[str] | None = None) -> None:
    """Build CBZ/PDF and optionally prune images according to the selection."""
    from package import make_cbz, make_pdf

    if not produced:
        print("[output] nothing colorized; no outputs written.")
        return

    if want is None:
        try:
            want = _resolve_outputs(settings, args)
        except ValueError as exc:
            print(f"[output] {exc}")
            raise SystemExit(2)

    out_dir = Path(getattr(args, "output", Path(".")))
    print(f"[output] formats: {', '.join(sorted(want))}")

    if "cbz" in want:
        cbz_path = getattr(args, "cbz", None) or (out_dir / "chapter.cbz")
        make_cbz(out_dir, Path(cbz_path), files=produced,
                 title=getattr(args, "title", None) or Path(cbz_path).stem)
    if "pdf" in want:
        pdf_path = getattr(args, "pdf", None) or (out_dir / "chapter.pdf")
        make_pdf(out_dir, Path(pdf_path), files=produced)

    keep = bool(settings.get("keep_images", False))
    if "image" not in want and not keep:
        for p in produced:
            try:
                Path(p).unlink()
            except OSError:
                pass
        print("[output] intermediate images removed "
              "(use --out image or keep_images=true to keep them)")


def _cmd_batch(args: argparse.Namespace) -> int:
    settings = _apply_overrides(load_settings(), args)
    colorizer = build_engine(settings)
    try:
        want = _resolve_outputs(settings, args)
    except ValueError as exc:
        print(f"[output] {exc}")
        return 2
    # Handles both loose images and .cbz/.zip archives in the input folder.
    ok, failed, produced = colorize_input(
        args.input, args.output, colorizer=colorizer, upscale=args.upscale,
        repack_cbz=("cbz" in want),
    )
    _emit_outputs(settings, args, produced, want=want)
    return 0 if failed == 0 else 1


def _cmd_cbz(args: argparse.Namespace) -> int:
    settings = _apply_overrides(load_settings(), args)
    colorizer = build_engine(settings)
    ok, failed, produced = colorize_archive(
        args.input, args.output, colorizer=colorizer, upscale=args.upscale,
        repack_cbz=True,  # the whole point of this command
    )
    # Respect --out for extra formats (pdf = also emit a PDF).
    try:
        want = _resolve_outputs(settings, args)
    except ValueError as exc:
        print(f"[cbz] {exc}")
        return 2
    if "pdf" in want and produced:
        from package import make_pdf

        pdf_path = Path(args.output) / f"{Path(args.input).stem}_colorized.pdf"
        make_pdf(Path(args.output), pdf_path, files=produced)
    print(f"[cbz] done. colorized -> {args.output}")
    return 0 if failed == 0 else 1


def _cmd_chapter(args: argparse.Namespace) -> int:
    settings = _apply_overrides(load_settings(), args)
    colorizer = build_engine(settings)
    try:
        want = _resolve_outputs(settings, args)
    except ValueError as exc:
        print(f"[output] {exc}")
        return 2
    ok, failed, produced = colorize_input(
        args.input, args.output, colorizer=colorizer, upscale=args.upscale,
        repack_cbz=("cbz" in want),
    )
    _emit_outputs(settings, args, produced, want=want)
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manga-colorizer",
        description="Automatically colorize grayscale manga on CPU.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--engine", choices=["mangacolv2", "comicnet", "zhang"],
                        help="colorization engine (default from settings)")

    def _add_upscale_flag(sp):
        sp.add_argument("--upscale", dest="upscale", action="store_true", default=None,
                        help="force-enable Real-ESRGAN upscaling")
        sp.add_argument("--no-upscale", dest="upscale", action="store_false",
                        help="disable upscaling even if enabled in settings")

    def _add_out_flag(sp):
        sp.add_argument(
            "--out", default=None,
            help="comma list of outputs: image,cbz,pdf (or 'all'). "
                 "Overrides settings['outputs'].",
        )

    c1 = sub.add_parser("colorize", help="colorize a single image")
    c1.add_argument("input", type=Path)
    c1.add_argument("output", type=Path)
    add_common(c1)
    c1.set_defaults(func=_cmd_colorize)

    c2 = sub.add_parser("batch", help="colorize a folder / chapter")
    c2.add_argument("input", type=Path)
    c2.add_argument("output", type=Path)
    c2.add_argument("--long-side", type=int, default=None)
    _add_out_flag(c2)
    _add_upscale_flag(c2)
    add_common(c2)
    c2.set_defaults(func=_cmd_batch)

    c3 = sub.add_parser("chapter", help="colorize a chapter + build chosen outputs")
    c3.add_argument("input", type=Path)
    c3.add_argument("output", type=Path)
    c3.add_argument("--cbz", type=Path, default=None,
                    help="(legacy) write a .cbz archive with this name")
    c3.add_argument("--pdf", type=Path, default=None,
                    help="(legacy) write a .pdf with this name")
    c3.add_argument("--title", default=None, help="comic title for CBZ metadata")
    c3.add_argument("--long-side", type=int, default=None)
    _add_out_flag(c3)
    _add_upscale_flag(c3)
    add_common(c3)
    c3.set_defaults(func=_cmd_chapter)

    c4 = sub.add_parser("check", help="verify model files are present")
    c4.set_defaults(func=_cmd_check)

    c5 = sub.add_parser("cbz", help="colorize a .cbz/.zip comic into a new .cbz")
    c5.add_argument("input", type=Path, help="input .cbz or .zip file")
    c5.add_argument("output", type=Path, help="output folder for results")
    c5.add_argument("--long-side", type=int, default=None)
    _add_out_flag(c5)
    _add_upscale_flag(c5)
    add_common(c5)
    c5.set_defaults(func=_cmd_cbz)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
