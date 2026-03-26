"""YT2 Studio — Video summarizer & short creator with a modern dark UI."""

import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import customtkinter as ctk

SCRIPT_DIR = Path(__file__).parent.resolve()
SETTINGS_FILE = SCRIPT_DIR / "settings.json"

# ── Quality Presets ───────────────────────────────────────────────────────────
# Each preset controls whisper model + scene detection together.
# Users pick one; internals are handled.

QUALITY_PRESETS = {
    "Fast":     {"whisper_model": "tiny.en",   "scene_threshold": 30.0},
    "Balanced": {"whisper_model": "small.en",  "scene_threshold": 27.0},
    "Deep":     {"whisper_model": "medium.en", "scene_threshold": 22.0},
}

# ── Known Models ──────────────────────────────────────────────────────────────
# Curated list of Ollama models confirmed to work with the pipeline.
# Vision models get image analysis; text-only models fall back gracefully.

KNOWN_MODELS = [
    "kimi-k2.5:cloud",       # free cloud, vision, best quality
    "qwen2.5-coder:cloud",   # free cloud, text-only
    "llava:7b",              # local, vision
    "moondream:latest",      # local, vision, lightweight
    "minicpm-v:latest",      # local, vision
    "mistral:latest",        # local, text-only
    "llama3:latest",         # local, text-only
]

VISION_MODELS = {"kimi-k2.5", "llava", "bakllava", "moondream",
                 "minicpm-v", "qwen2-vl", "llava-llama3", "llava-phi3"}

# ── Settings ─────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "quality": "Balanced",
    "ollama_model": "kimi-k2.5:cloud",
    "font_name": "Super Carnival.ttf",
    "font_size": 100,
    "font_border_weight": 10,
    "resolution_w": 1080,
    "resolution_h": 1920,
    "percent_main_clip": 40,
    "text_position_percent": 30,
    "num_threads": max(1, os.cpu_count() or 4),
    "language": "en",
}


def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def get_quality_params(settings):
    """Resolve quality preset into whisper_model + scene_threshold."""
    preset = QUALITY_PRESETS.get(settings.get("quality", "Balanced"),
                                QUALITY_PRESETS["Balanced"])
    return preset["whisper_model"], preset["scene_threshold"]


def open_path(path):
    path = Path(path)
    if path.suffix.lower() in (".html", ".htm"):
        webbrowser.open(path.as_uri())
    elif sys.platform == "win32":
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


def get_available_fonts():
    fonts_dir = SCRIPT_DIR / "assets" / "fonts"
    if not fonts_dir.exists():
        return []
    return [f.name for f in fonts_dir.iterdir() if f.suffix.lower() in (".ttf", ".otf")]


# ── Palette ──────────────────────────────────────────────────────────────────

P = {
    "bg":         "#09090b",
    "surface":    "#18181b",
    "surface2":   "#111113",
    "elevated":   "#27272a",
    "border":     "#27272a",
    "border_dim": "#1f1f23",
    "text":       "#fafafa",
    "text2":      "#a1a1aa",
    "text3":      "#71717a",
    "accent":       "#3b82f6",
    "accent_hover": "#2563eb",
    "green":       "#22c55e",
    "green_hover": "#16a34a",
    "red":         "#ef4444",
    "red_hover":   "#dc2626",
    "btn":         "#27272a",
    "btn_hover":   "#3f3f46",
    "input":       "#0f0f12",
}


# ── Reusable Widgets ─────────────────────────────────────────────────────────


class LogBox(ctk.CTkTextbox):
    def __init__(self, master, **kw):
        super().__init__(
            master, state="disabled",
            font=("Consolas", 11),
            fg_color=P["bg"],
            text_color=P["text3"],
            border_width=1,
            border_color=P["border_dim"],
            corner_radius=12,
            **kw,
        )

    def log(self, text):
        self.configure(state="normal")
        self.insert("end", text + "\n")
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


def _label(parent, text, size=11, color=None, bold=False, **kw):
    weight = "bold" if bold else "normal"
    return ctk.CTkLabel(parent, text=text, font=("", size, weight),
                        text_color=color or P["text2"], **kw)


def _card(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=P["surface"], corner_radius=14,
                        border_width=1, border_color=P["border_dim"], **kw)


def _entry(parent, placeholder="", width=None, height=36, **kw):
    opts = {"height": height, "font": ("", 12), "fg_color": P["input"],
            "border_color": P["border"], "corner_radius": 10,
            "placeholder_text": placeholder}
    if width:
        opts["width"] = width
    opts.update(kw)
    return ctk.CTkEntry(parent, **opts)


def _btn(parent, text, color=None, hover=None, width=None, height=36, bold=True, **kw):
    opts = {
        "text": text, "height": height, "corner_radius": 10,
        "font": ("", 12, "bold" if bold else "normal"),
        "fg_color": color or P["accent"],
        "hover_color": hover or P["accent_hover"],
    }
    if width:
        opts["width"] = width
    opts.update(kw)
    return ctk.CTkButton(parent, **opts)


def _progress(parent):
    bar = ctk.CTkProgressBar(parent, height=4, corner_radius=2,
                             fg_color=P["border_dim"], progress_color=P["accent"])
    bar.set(0)
    return bar


def _pill(parent, text, color, text_color="#fff"):
    return ctk.CTkLabel(parent, text=text, font=("", 9, "bold"),
                        text_color=text_color, fg_color=color,
                        corner_radius=8, height=20, width=54)


def _dropdown(parent, values, current, width=180):
    om = ctk.CTkOptionMenu(
        parent, values=values, width=width, height=32, corner_radius=8,
        fg_color=P["input"], button_color=P["btn"],
        button_hover_color=P["btn_hover"], font=("", 12),
    )
    om.set(current)
    return om


# ── Summarize Tab ────────────────────────────────────────────────────────────


class SummarizeTab(ctk.CTkFrame):
    """Clean, opinionated summarizer. Paste URL, press go, get a visual guide."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._working = False
        self._output_path = None

        # ── URL Card ──
        card = _card(self)
        card.pack(fill="x", padx=24, pady=(24, 12))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=20)

        # Top row: label + vision badge
        top = ctk.CTkFrame(inner, fg_color="transparent")
        top.pack(fill="x", pady=(0, 10))
        _label(top, "Paste a YouTube URL to generate a visual guide",
               size=12, color=P["text2"]).pack(side="left")

        model_name = app.settings["ollama_model"].split(":")[0].lower()
        is_vision = model_name in VISION_MODELS
        badge_color = P["green"] if is_vision else P["elevated"]
        badge_text_color = "#fff" if is_vision else P["text3"]
        badge_label = "VISION" if is_vision else "TEXT"
        self.vision_badge = _pill(top, badge_label, badge_color, badge_text_color)
        self.vision_badge.pack(side="right")

        # URL row
        url_row = ctk.CTkFrame(inner, fg_color="transparent")
        url_row.pack(fill="x")

        self.url_entry = _entry(url_row, "https://www.youtube.com/watch?v=...", height=42)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.start_btn = _btn(url_row, "  Analyze  ", height=42, width=130,
                              command=self._on_start)
        self.start_btn.pack(side="right")

        # ── Progress ──
        self.progress = _progress(self)
        self.progress.pack(fill="x", padx=24, pady=(0, 6))

        # ── Log ──
        self.log_box = LogBox(self)
        self.log_box.pack(fill="both", expand=True, padx=24, pady=(0, 10))

        # ── Bottom Bar ──
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=24, pady=(0, 20))

        self.open_btn = _btn(bar, "Open Output", P["green"], P["green_hover"],
                             width=130, state="disabled", command=self._open_output)
        self.open_btn.pack(side="left")

        _btn(bar, "Open Folder", P["btn"], P["btn_hover"], width=130,
             bold=False, command=self._open_folder).pack(side="left", padx=10)

    # ── Logic ──

    def _log(self, text):
        self.after(0, self.log_box.log, text)

    def _set_progress(self, step, total=6):
        self.after(0, self.progress.set, step / total)

    def update_vision_badge(self):
        model_name = self.app.settings["ollama_model"].split(":")[0].lower()
        is_vision = model_name in VISION_MODELS
        if is_vision:
            self.vision_badge.configure(fg_color=P["green"], text_color="#fff", text="VISION")
        else:
            self.vision_badge.configure(fg_color=P["elevated"], text_color=P["text3"], text="TEXT")

    def _on_start(self):
        if self._working:
            return
        url = self.url_entry.get().strip()
        if not url:
            self.log_box.log("Enter a YouTube URL first.")
            return

        self._working = True
        self._output_path = None
        self.start_btn.configure(state="disabled", text="  Working...  ")
        self.open_btn.configure(state="disabled")
        self.log_box.clear()
        self.progress.set(0)
        self.app.set_status("Analyzing video...")

        # All params come from settings — no per-run tweaking
        model = self.app.settings["ollama_model"]
        whisper_model, scene_threshold = get_quality_params(self.app.settings)

        step_counter = {"n": 0}
        original_log = self._log

        def progress_log(text):
            original_log(text)
            if text.startswith("["):
                step_counter["n"] += 1
                self._set_progress(step_counter["n"])

        def worker():
            try:
                from summarizer import run_pipeline
                json_path = run_pipeline(
                    url, model=model, whisper_model=whisper_model,
                    scene_threshold=scene_threshold, log=progress_log,
                )
                # run_pipeline returns single .json file
                self._output_path = json_path
                progress_log(f"\nDone!")
                progress_log(f"  Output: {json_path}")
                self._set_progress(5)
                self.after(0, self._on_done, True)
            except Exception as e:
                progress_log(f"\nERROR: {e}")
                self.after(0, self._on_done, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, success):
        self._working = False
        self.start_btn.configure(state="normal", text="  Analyze  ")
        if success and self._output_path:
            self.open_btn.configure(state="normal")
        self.app.set_status("Done" if success else "Failed")

    def _open_output(self):
        if self._output_path and Path(self._output_path).exists():
            open_path(self._output_path)

    def _open_folder(self):
        folder = SCRIPT_DIR / "output" / "summaries"
        folder.mkdir(parents=True, exist_ok=True)
        open_path(folder)


# ── Create Short Tab ─────────────────────────────────────────────────────────


class ShortsTab(ctk.CTkFrame):
    """Create vertical short-form video with auto-captions and background."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._working = False
        self._output_path = None

        # ── Input Card ──
        card = _card(self)
        card.pack(fill="x", padx=24, pady=(24, 12))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=20)

        _label(inner, "Select a video to turn into a vertical short with captions",
               size=12, color=P["text2"]).pack(anchor="w", pady=(0, 14))

        # Video input
        _label(inner, "INPUT VIDEO", size=10, color=P["text3"],
               bold=True).pack(anchor="w", pady=(0, 6))
        vid_row = ctk.CTkFrame(inner, fg_color="transparent")
        vid_row.pack(fill="x", pady=(0, 14))

        self.video_entry = _entry(vid_row, "Select an MP4 file...")
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        _btn(vid_row, "Browse", P["btn"], P["btn_hover"], 80,
             bold=False, command=self._browse_video).pack(side="right")

        # Background
        _label(inner, "BACKGROUND VIDEO", size=10, color=P["text3"],
               bold=True).pack(anchor="w", pady=(0, 6))
        bg_row = ctk.CTkFrame(inner, fg_color="transparent")
        bg_row.pack(fill="x", pady=(0, 10))

        self.bg_entry = _entry(bg_row, "Leave empty for random from assets/backgrounds")
        self.bg_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        _btn(bg_row, "Browse", P["btn"], P["btn_hover"], 80,
             bold=False, command=self._browse_bg).pack(side="right")

        # Create button
        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(6, 0))
        self.start_btn = _btn(btn_row, "  Create Short  ", height=40, width=150,
                              command=self._on_start)
        self.start_btn.pack(side="right")

        # ── Progress ──
        self.progress = _progress(self)
        self.progress.pack(fill="x", padx=24, pady=(0, 6))

        # ── Log ──
        self.log_box = LogBox(self)
        self.log_box.pack(fill="both", expand=True, padx=24, pady=(0, 10))

        # ── Bottom Bar ──
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=24, pady=(0, 20))

        self.open_btn = _btn(bar, "Open Output", P["green"], P["green_hover"],
                             width=130, state="disabled", command=self._open_output)
        self.open_btn.pack(side="left")

        _btn(bar, "Open Folder", P["btn"], P["btn_hover"], width=130,
             bold=False, command=self._open_folder).pack(side="left", padx=10)

    def _browse_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mkv *.webm *.avi *.mov"), ("All", "*.*")])
        if path:
            self.video_entry.delete(0, "end")
            self.video_entry.insert(0, path)

    def _browse_bg(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Background Video",
            filetypes=[("Video files", "*.mp4 *.mkv *.webm"), ("All", "*.*")])
        if path:
            self.bg_entry.delete(0, "end")
            self.bg_entry.insert(0, path)

    def _log(self, text):
        self.after(0, self.log_box.log, text)

    def _on_start(self):
        if self._working:
            return
        video_path = self.video_entry.get().strip()
        if not video_path:
            self.log_box.log("Select a video file first.")
            return

        self._working = True
        self._output_path = None
        self.start_btn.configure(state="disabled", text="  Working...  ")
        self.open_btn.configure(state="disabled")
        self.log_box.clear()
        self.progress.set(0)
        self.app.set_status("Creating short...")

        bg_path = self.bg_entry.get().strip() or None
        whisper_model, _ = get_quality_params(self.app.settings)

        # All config from hard-set defaults — no per-run sliders
        config = {
            "font_name": self.app.settings["font_name"],
            "font_size": self.app.settings["font_size"],
            "font_border_weight": self.app.settings["font_border_weight"],
            "resolution_w": self.app.settings["resolution_w"],
            "resolution_h": self.app.settings["resolution_h"],
            "percent_main_clip": self.app.settings["percent_main_clip"],
            "text_position_percent": self.app.settings["text_position_percent"],
            "whisper_model": whisper_model,
            "language": self.app.settings["language"],
            "num_threads": self.app.settings["num_threads"],
        }

        step_counter = {"n": 0}
        original_log = self._log

        def progress_log(text):
            original_log(text)
            if text.startswith("["):
                step_counter["n"] += 1
                self.after(0, self.progress.set, step_counter["n"] / 4)

        def worker():
            try:
                from shorts import create_short
                path = create_short(
                    video_path, background_path=bg_path,
                    config=config, log=progress_log,
                )
                self._output_path = path
                progress_log(f"\nDone! Output: {path}")
                self.after(0, self.progress.set, 1.0)
                self.after(0, self._on_done, True)
            except Exception as e:
                progress_log(f"\nERROR: {e}")
                self.after(0, self._on_done, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, success):
        self._working = False
        self.start_btn.configure(state="normal", text="  Create Short  ")
        if success and self._output_path:
            self.open_btn.configure(state="normal")
        self.app.set_status("Done" if success else "Failed")

    def _open_output(self):
        if self._output_path and Path(self._output_path).exists():
            open_path(self._output_path)

    def _open_folder(self):
        folder = SCRIPT_DIR / "output" / "shorts"
        folder.mkdir(parents=True, exist_ok=True)
        open_path(folder)


# ── Settings Tab ─────────────────────────────────────────────────────────────


class SettingsTab(ctk.CTkScrollableFrame):
    """Minimal settings — global presets, not individual knobs."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app

        card = _card(self)
        card.pack(fill="x", padx=24, pady=24)

        c = ctk.CTkFrame(card, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=28, pady=24)

        row = 0

        def heading(text, r):
            _label(c, text, size=10, color=P["text3"],
                   bold=True).grid(row=r, column=0, columnspan=2,
                                   sticky="w", pady=(16 if r > 0 else 0, 10))
            return r + 1

        def setting_row(label, widget, r):
            _label(c, label, size=12, color=P["text2"]).grid(
                row=r, column=0, sticky="w", pady=8, padx=(0, 24))
            widget.grid(row=r, column=1, sticky="w", pady=8)
            return r + 1

        # ── ANALYSIS ──
        row = heading("ANALYSIS", row)

        # Quality preset
        self.quality_var = _dropdown(c, list(QUALITY_PRESETS.keys()),
                                     app.settings.get("quality", "Balanced"), width=160)
        row = setting_row("Quality", self.quality_var, row)

        # Quality description
        desc = _label(c, "Controls transcription accuracy and scene detection sensitivity",
                      size=10, color=P["text3"])
        desc.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        row += 1

        # LLM Model — dropdown of known models
        self.model_var = _dropdown(c, KNOWN_MODELS,
                                   app.settings["ollama_model"], width=220)
        row = setting_row("LLM Model", self.model_var, row)

        # Model description
        model_desc = _label(c, "Vision models analyze screenshots + audio. Text models use transcript only.",
                            size=10, color=P["text3"])
        model_desc.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        row += 1

        # ── SHORTS ──
        row = heading("SHORTS", row)

        fonts = get_available_fonts() or ["Super Carnival.ttf"]
        self.font_var = _dropdown(c, fonts, app.settings["font_name"], width=220)
        row = setting_row("Caption Font", self.font_var, row)

        self.lang_var = _dropdown(c, ["en", "auto"], app.settings["language"], width=100)
        row = setting_row("Language", self.lang_var, row)

        # ── Actions ──
        row = heading("", row)

        btn_frame = ctk.CTkFrame(c, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="w")

        _btn(btn_frame, "Save", P["green"], P["green_hover"],
             width=120, command=self._save).pack(side="left", padx=(0, 10))

        _btn(btn_frame, "Clear Cache", P["btn"], P["btn_hover"],
             width=120, bold=False, command=self._clear_cache).pack(side="left")

        row += 1

        self.info_label = _label(c, "", size=11, color=P["accent"])
        self.info_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _save(self):
        s = self.app.settings
        s["quality"] = self.quality_var.get()
        s["ollama_model"] = self.model_var.get()
        s["font_name"] = self.font_var.get()
        s["language"] = self.lang_var.get()
        save_settings(s)

        # Update vision badge on summarize tab
        for tab_widget in self.app.tabview.tab("Summarize").winfo_children():
            if isinstance(tab_widget, SummarizeTab):
                tab_widget.update_vision_badge()
                break

        self.info_label.configure(text="Settings saved.")
        self.after(3000, lambda: self.info_label.configure(text=""))

    def _clear_cache(self):
        cache_dir = SCRIPT_DIR / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
        self.info_label.configure(text="Cache cleared.")
        self.after(3000, lambda: self.info_label.configure(text=""))


# ── Main App ─────────────────────────────────────────────────────────────────


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.settings = load_settings()

        self.title("YT2 Studio")
        self.geometry("900x680")
        self.minsize(780, 560)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color=P["surface2"])

        # ── Header ──
        header = ctk.CTkFrame(self, fg_color=P["bg"], height=52, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.pack(side="left", padx=20)

        ctk.CTkLabel(logo_frame, text="YT2", font=("", 18, "bold"),
                     text_color=P["accent"]).pack(side="left")
        ctk.CTkLabel(logo_frame, text="Studio", font=("", 18),
                     text_color=P["text"]).pack(side="left", padx=(4, 0))

        _label(header, "Video Analysis & Short Creator", size=11,
               color=P["text3"]).pack(side="left", padx=(8, 0))

        _pill(header, "v2.0", P["elevated"], P["text3"]).pack(side="right", padx=20)

        # ── Tabs ──
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=P["surface2"],
            segmented_button_fg_color=P["bg"],
            segmented_button_selected_color=P["accent"],
            segmented_button_selected_hover_color=P["accent_hover"],
            segmented_button_unselected_color=P["elevated"],
            segmented_button_unselected_hover_color=P["btn_hover"],
            corner_radius=10,
            border_width=0,
        )
        self.tabview.pack(fill="both", expand=True, padx=12, pady=(6, 0))

        self.tabview.add("Analyze")
        self.tabview.add("Create Short")
        self.tabview.add("Settings")

        SummarizeTab(self.tabview.tab("Analyze"), self).pack(fill="both", expand=True)
        ShortsTab(self.tabview.tab("Create Short"), self).pack(fill="both", expand=True)
        SettingsTab(self.tabview.tab("Settings"), self).pack(fill="both", expand=True)

        # ── Status Bar ──
        status = ctk.CTkFrame(self, fg_color=P["bg"], height=28, corner_radius=0)
        status.pack(fill="x")
        status.pack_propagate(False)

        self.status_label = _label(status, "Ready", size=10, color=P["text3"])
        self.status_label.pack(side="left", padx=16, pady=4)

    def set_status(self, text):
        self.status_label.configure(text=text)


if __name__ == "__main__":
    app = App()
    app.mainloop()
