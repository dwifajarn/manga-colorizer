"""Tkinter desktop GUI for the manga colorizer.

A single-window form (input/output folder, engine, quality sliders, output
formats, upscale toggle) with a progress bar and a live log. The heavy
colorization pipeline runs in a **worker thread** so the window never freezes,
and a **Stop** button cancels cooperatively via :mod:`cancel`.

The GUI is a thin front-end: it only chooses settings and then calls the
existing :func:`batch.colorize_input` / :func:`cli._emit_outputs` machinery. No
colorization logic lives here.

Run it with:

    python src/gui.py
    # or double-click JALANKAN-GUI.bat on Windows
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Make sibling modules importable regardless of the working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ROOT = _HERE.parent
VERSION = "1.1.0"

ENGINES = ("mangacolv2", "comicnet", "zhang")
# Human labels for the engine dropdown.
ENGINE_LABELS = {
    "mangacolv2": "mangacolv2  (manga, default)",
    "comicnet": "comicnet    (manga, alternate)",
    "zhang": "zhang       (photo)",
}
LABEL_TO_ENGINE = {v: k for k, v in ENGINE_LABELS.items()}


class _QueueWriter:
    """A file-like object that forwards writes to a queue.

    Used to redirect the pipeline's ``print()`` output into the GUI log while it
    runs in the worker thread. Tk widgets may only be touched from the main
    thread, so we push text onto a queue that the main loop polls.
    """

    def __init__(self, q: "queue.Queue[str]") -> None:
        self._q = q

    def write(self, text: str) -> int:
        if text:
            self._q.put(text)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - nothing buffered
        pass

    def isatty(self) -> bool:
        return False


class MangaColorizerGUI:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Manga Colorizer v{VERSION}")
        self.root.minsize(640, 560)

        self._worker: threading.Thread | None = None
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._progress_queue: "queue.Queue[tuple[int, int, str]]" = queue.Queue()
        self._running = False

        # Tk variables bound to the widgets.
        self.var_input = tk.StringVar(value=str(ROOT / "input"))
        self.var_output = tk.StringVar(value=str(ROOT / "output"))
        self.var_engine = tk.StringVar(value=ENGINE_LABELS["mangacolv2"])
        self.var_size = tk.IntVar(value=576)
        self.var_saturation = tk.DoubleVar(value=1.7)
        self.var_png = tk.BooleanVar(value=True)
        self.var_cbz = tk.BooleanVar(value=True)
        self.var_pdf = tk.BooleanVar(value=True)
        self.var_upscale = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Idle.")

        self._build_ui()
        self._poll_queues()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        style = ttk.Style()
        # A slightly roomier default on Windows keeps the sliders pleasant.
        try:
            style.configure("TButton", padding=6)
        except tk.TclError:  # pragma: no cover
            pass

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        # --- Paths -----------------------------------------------------
        paths = ttk.LabelFrame(outer, text="Folders", padding=10)
        paths.pack(fill="x")
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="Input:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(paths, textvariable=self.var_input).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(paths, text="Browse…", command=self._pick_input).grid(
            row=0, column=2)

        ttk.Label(paths, text="Output:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(paths, textvariable=self.var_output).grid(
            row=1, column=1, sticky="ew", padx=6)
        ttk.Button(paths, text="Browse…", command=self._pick_output).grid(
            row=1, column=2)

        # --- Options ---------------------------------------------------
        opts = ttk.LabelFrame(outer, text="Options", padding=10)
        opts.pack(fill="x", pady=(10, 0))
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="Engine:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(
            opts, textvariable=self.var_engine, state="readonly",
            values=list(ENGINE_LABELS.values()),
        ).grid(row=0, column=1, sticky="ew", padx=6, columnspan=2)

        # Quality slider (mangacolv2_size).
        ttk.Label(opts, text="Quality size:").grid(
            row=1, column=0, sticky="w", pady=3)
        size_scale = ttk.Scale(
            opts, from_=256, to=768, variable=self.var_size, command=self._on_size,
        )
        size_scale.grid(row=1, column=1, sticky="ew", padx=6)
        self.lbl_size = ttk.Label(opts, text="576", width=4)
        self.lbl_size.grid(row=1, column=2, sticky="e")

        ttk.Label(opts, text="Saturation:").grid(
            row=2, column=0, sticky="w", pady=3)
        sat_scale = ttk.Scale(
            opts, from_=0.5, to=2.5, variable=self.var_saturation,
            command=self._on_sat,
        )
        sat_scale.grid(row=2, column=1, sticky="ew", padx=6)
        self.lbl_sat = ttk.Label(opts, text="1.7", width=4)
        self.lbl_sat.grid(row=2, column=2, sticky="e")

        # Output format checkboxes.
        fmt = ttk.Frame(opts)
        fmt.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(fmt, text="Formats:").pack(side="left", padx=(0, 8))
        ttk.Checkbutton(fmt, text="PNG", variable=self.var_png).pack(
            side="left", padx=3)
        ttk.Checkbutton(fmt, text="CBZ", variable=self.var_cbz).pack(
            side="left", padx=3)
        ttk.Checkbutton(fmt, text="PDF", variable=self.var_pdf).pack(
            side="left", padx=3)
        ttk.Checkbutton(fmt, text="Upscale 2×", variable=self.var_upscale).pack(
            side="left", padx=(16, 3))

        # --- Progress --------------------------------------------------
        prog = ttk.LabelFrame(outer, text="Progress", padding=10)
        prog.pack(fill="x", pady=(10, 0))
        prog.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(prog, textvariable=self.var_status).grid(
            row=1, column=0, sticky="w", pady=(4, 0))

        # --- Log -------------------------------------------------------
        logf = ttk.LabelFrame(outer, text="Log", padding=6)
        logf.pack(fill="both", expand=True, pady=(10, 0))
        logf.rowconfigure(0, weight=1)
        logf.columnconfigure(0, weight=1)

        self.log = tk.Text(logf, height=10, wrap="none", state="disabled",
                           background="#111", foreground="#ddd",
                           font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)

        # --- Buttons ---------------------------------------------------
        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(10, 0))
        self.btn_start = ttk.Button(btns, text="▶  Start", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(btns, text="■  Stop", command=self._stop,
                                    state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(btns, text="📂  Open Output",
                   command=self._open_output).pack(side="left", padx=6)
        ttk.Button(btns, text="Exit", command=self._on_close).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Small widget callbacks
    # ------------------------------------------------------------------
    def _on_size(self, _value: str) -> None:
        # Snap to a multiple of 32 (the engine requires it).
        v = int(float(self.var_size.get()))
        v = max(256, min(768, (v // 32) * 32 or 288))
        self.var_size.set(v)
        self.lbl_size.configure(text=str(v))

    def _on_sat(self, _value: str) -> None:
        self.var_saturation.set(round(float(self.var_saturation.get()), 1))
        self.lbl_sat.configure(text=f"{self.var_saturation.get():.1f}")

    def _pick_input(self) -> None:
        chosen = filedialog.askdirectory(
            title="Choose input folder", initialdir=self.var_input.get() or str(ROOT))
        if chosen:
            self.var_input.set(chosen)

    def _pick_output(self) -> None:
        chosen = filedialog.askdirectory(
            title="Choose output folder", initialdir=self.var_output.get() or str(ROOT))
        if chosen:
            self.var_output.set(chosen)

    def _open_output(self) -> None:
        import os
        import subprocess

        out = Path(self.var_output.get())
        out.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            else:
                subprocess.Popen(["xdg-open", str(out)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open output", f"Could not open folder:\n{exc}")

    # ------------------------------------------------------------------
    # Logging / progress plumbing
    # ------------------------------------------------------------------
    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        # Keep only the last ~2000 lines to bound memory.
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > 2000:
            self.log.delete("1.0", f"{line_count - 2000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.var_status.set(text)

    def _poll_queues(self) -> None:
        """Drain worker queues on the Tk main loop (thread-safe UI updates)."""
        try:
            while True:
                self.append_log(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        try:
            while True:
                current, total, status = self._progress_queue.get_nowait()
                pct = 0 if total == 0 else int(current * 100 / total)
                self.progress.configure(value=pct)
                self._set_status(f"{status}  ({current}/{total})")
        except queue.Empty:
            pass

        self.root.after(120, self._poll_queues)

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------
    def _collect_settings(self) -> dict:
        from colorize import load_settings

        settings = load_settings()
        settings["engine"] = LABEL_TO_ENGINE.get(
            self.var_engine.get(), "mangacolv2")
        settings["mangacolv2_size"] = int(self.var_size.get())
        settings["post_saturation"] = float(self.var_saturation.get())
        settings["upscale"] = bool(self.var_upscale.get())

        outputs = []
        if self.var_png.get():
            outputs.append("image")
        if self.var_cbz.get():
            outputs.append("cbz")
        if self.var_pdf.get():
            outputs.append("pdf")
        settings["outputs"] = outputs or ["image"]
        return settings

    def _start(self) -> None:
        if self._running:
            return
        settings = self._collect_settings()
        in_dir = Path(self.var_input.get().strip())
        out_dir = Path(self.var_output.get().strip())

        if not in_dir.exists():
            messagebox.showerror("Input", f"Input folder not found:\n{in_dir}")
            return
        # Quick pre-flight: is there anything to do?
        try:
            import batch as _batch

            exts = settings.get("supported_ext", [".png", ".jpg", ".jpeg"])
            has_work = bool(_batch.collect_archives(in_dir)
                            or _batch.collect_images(in_dir, exts))
        except Exception:  # noqa: BLE001
            has_work = True
        if not has_work:
            messagebox.showwarning(
                "Nothing to do",
                f"No images or .cbz/.zip files found in:\n{in_dir}")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        self.append_log(f"\n=== Start · engine={settings['engine']} · "
                        f"formats={','.join(settings['outputs'])} · "
                        f"upscale={settings['upscale']} ===\n")
        self.progress.configure(value=0)
        self._set_status("Starting…")
        self._running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self._worker = threading.Thread(
            target=self._run_job, args=(settings, in_dir, out_dir), daemon=True)
        self._worker.start()

    def _run_job(self, settings: dict, in_dir: Path, out_dir: Path) -> None:
        """Runs in the worker thread. Never touches Tk widgets directly."""
        import cancel as _cancel

        _cancel.reset()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _QueueWriter(self._log_queue)
        try:
            import batch as _batch
            from cli import _emit_outputs, _parse_outputs

            colorizer = _batch.build_engine(settings)
            want = _parse_outputs(settings.get("outputs", ["image"]))

            def on_progress(current: int, total: int, src: Path) -> None:
                cancelled = _cancel.is_cancelled()
                note = "cancelling…" if cancelled else f"colored {src.name}"
                self._progress_queue.put((current, total, note))

            ok, failed, archives, loose = _batch.colorize_input(
                in_dir, out_dir, colorizer=colorizer,
                upscale=settings.get("upscale"), on_progress=on_progress,
            )

            if _cancel.is_cancelled():
                self._log_queue.put("[gui] stopped by user; packing what is done.\n")

            # Reuse the CLI emitter so naming/layout matches the terminal flow.
            class _Args:
                output = out_dir
                input = in_dir
                out = None
                title = None
                cbz = None
                pdf = None

            _emit_outputs(settings, _Args(), archives, loose, want=want)

            if _cancel.is_cancelled():
                self._finish(f"Stopped. success={ok} failed={failed}")
            else:
                self._finish(f"Finished. success={ok} failed={failed}")
        except Exception as exc:  # noqa: BLE001
            self._log_queue.put("[gui] error:\n" + traceback.format_exc())
            self._finish(f"Error: {exc}", error=True)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _finish(self, status: str, error: bool = False) -> None:
        """Called from the worker thread; hop back to the main thread."""
        self.root.after(0, lambda: self._finish_ui(status, error))

    def _finish_ui(self, status: str, error: bool) -> None:
        self._running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_status(status)
        if error:
            self.append_log(f"\n[gui] {status}\n")
        else:
            self.progress.configure(value=100)
            self.append_log(f"\n[gui] {status}\n")

    def _stop(self) -> None:
        if not self._running:
            return
        import cancel as _cancel

        _cancel.cancel()
        self._set_status("Cancelling… (finishing current page)")
        self.btn_stop.configure(state="disabled")
        self.append_log("\n[gui] Stop requested — will finish the current page.\n")

    def _on_close(self) -> None:
        if self._running and not messagebox.askyesno(
                "Quit", "A job is running. Stop it and quit?"):
            return
        import cancel as _cancel

        _cancel.cancel()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    MangaColorizerGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
