"""Shorts creator — creates vertical short-form videos with captions and background."""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, clips_array,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = SCRIPT_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
OUTPUT_DIR = SCRIPT_DIR / "output" / "shorts"

# Defaults (overridable via config dict)
DEFAULTS = {
    "font_name": "Super Carnival.ttf",
    "font_size": 100,
    "font_border_weight": 10,
    "resolution_w": 1080,
    "resolution_h": 1920,
    "percent_main_clip": 40,
    "text_position_percent": 30,
    "whisper_model": "small.en",
    "language": "en",
    "num_threads": max(1, __import__('os').cpu_count() or 4),
}


def _round_down(num, decimals=0):
    return math.floor(num * 10 ** decimals) / 10 ** decimals


# ── Video Utilities ──────────────────────────────────────────────────────────


def _crop_clip(clip, width, height):
    """Crop a video clip to exact dimensions, maintaining aspect ratio."""
    ow, oh = clip.size
    wr = width / ow
    hr = height / oh
    max_r = max(wr, hr)

    clip = clip.resized((ow * max_r, oh * max_r))
    nw, nh = clip.size

    if wr > hr:
        hc = nh - height
        y1 = round(hc / 2)
        y2 = min(y1 + height, nh)
        clip = clip.cropped(y1=y1, y2=y2)
    elif hr > wr:
        wc = nw - width
        x1 = round(wc / 2)
        x2 = min(x1 + width, nw)
        clip = clip.cropped(x1=x1, x2=x2)
        clip = clip.resized((width, height))

    return clip


def _get_background_clip(duration, background_path, config):
    """Get a trimmed, cropped background clip."""
    full_clip = VideoFileClip(str(background_path))

    if full_clip.duration < duration:
        raise ValueError(f"Background video ({full_clip.duration:.1f}s) is shorter than input ({duration:.1f}s)")

    start = _round_down(random.uniform(0, full_clip.duration - duration))
    trimmed = full_clip.subclipped(start, start + duration)

    w, h = trimmed.size
    trimmed = _crop_clip(trimmed, round(w * 0.9), h)

    res_w = config["resolution_w"]
    res_h = config["resolution_h"]
    pct = config["percent_main_clip"]
    target_h = round(res_h * (1 - pct / 100))

    return _crop_clip(trimmed, res_w, target_h).with_audio(None)


# ── Transcription ────────────────────────────────────────────────────────────


def _transcribe_with_words(audio_clip, config, log=print):
    """Transcribe audio and return word-level timestamps."""
    log("  Transcribing for captions...")
    import whisper

    # Save audio to temp file
    temp_dir = SCRIPT_DIR / "cache" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio = temp_dir / f"shorts_{int(time.time())}.mp3"
    audio_clip.write_audiofile(str(temp_audio), codec="mp3", verbose=False, logger=None)

    model = whisper.load_model(config["whisper_model"])
    result = model.transcribe(str(temp_audio), language=config["language"],
                              word_timestamps=True, verbose=False)

    # Clean up
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        temp_audio.unlink()
    except Exception:
        pass

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "timestamp": (word["start"], word["end"]),
                "text": word["word"].strip(),
            })

    log(f"  Got {len(words)} words")
    return words


# ── Caption Rendering ────────────────────────────────────────────────────────


def _create_text_image(text, font_path, font_size, max_width, border_weight):
    """Render text as a transparent PIL image."""
    image = Image.new("RGBA", (max_width, font_size * 10), (0, 0, 0, 0))
    font = ImageFont.truetype(str(font_path), font_size)
    draw = ImageDraw.Draw(image)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((max_width - w) / 2, round(h * 0.2)),
        text, font=font, fill="white",
        stroke_width=border_weight, stroke_fill="black",
    )
    image = image.crop((0, 0, max_width, round(h * 1.6)))
    return image


def _add_captions(clip, words, config, log=print):
    """Overlay word-level captions onto the video clip."""
    if not words:
        return clip

    log("  Adding captions...")
    font_path = FONTS_DIR / config["font_name"]
    font_size = config["font_size"]
    border_weight = config["font_border_weight"]
    res_h = config["resolution_h"]
    text_pos_pct = config["text_position_percent"]
    clip_w = clip.size[0]

    clips = []
    previous_time = 0
    queued_texts = []
    full_start = None
    end = 0

    for pos, word in enumerate(words):
        start, end = word["timestamp"]
        text = word["text"]

        if start > previous_time and not queued_texts:
            clips.append(clip.subclipped(previous_time, start))

        # Extend end to next word or add small padding
        if pos + 1 < len(words):
            next_start = words[pos + 1]["timestamp"][0]
            if next_start > end:
                end = end + min(0.5, next_start - end)

        # Queue short gaps together
        if end - previous_time < 0.3 and pos + 1 < len(words):
            if full_start is None:
                full_start = start
            queued_texts.append(text)
            continue

        queued_texts.append(text)
        combined_text = " ".join(queued_texts)
        queued_texts = []

        if full_start is None:
            full_start = start

        if full_start > clip.duration or end > clip.duration:
            full_start = None
            continue

        # Create caption overlay
        text_img = _create_text_image(combined_text, font_path, font_size, clip_w, border_weight)
        subclip = clip.subclipped(full_start, end)
        img_clip = ImageClip(np.array(text_img), duration=subclip.duration)
        y_offset = round(res_h * (text_pos_pct / 100))
        captioned = CompositeVideoClip([subclip, img_clip.with_position((0, y_offset))])
        clips.append(captioned)

        previous_time = end
        full_start = None

    # Remaining clip after last caption
    if words and clip.duration - end > 0.01:
        clips.append(clip.subclipped(end, clip.duration))

    if not clips:
        return clip

    return concatenate_videoclips(clips)


# ── Main Pipeline ────────────────────────────────────────────────────────────


def create_short(video_path, background_path=None, config=None, log=print):
    """Create a short-form vertical video with captions and background.

    Args:
        video_path: Path to input video
        background_path: Path to background video (random from assets/backgrounds if None)
        config: Dict of settings (uses DEFAULTS for missing keys)
        log: Callback for progress messages

    Returns:
        Path to output video
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Merge config with defaults
    cfg = {**DEFAULTS}
    if config:
        cfg.update(config)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Pick background
    if background_path is None:
        bg_files = [f for f in BACKGROUNDS_DIR.iterdir() if f.suffix.lower() in (".mp4", ".mkv", ".webm")]
        if not bg_files:
            raise FileNotFoundError(f"No background videos found in {BACKGROUNDS_DIR}")
        background_path = random.choice(bg_files)
    background_path = Path(background_path)

    log(f"[1/4] Loading video: {video_path.name}")
    input_clip = VideoFileClip(str(video_path))
    audio = input_clip.audio

    log("[2/4] Creating split-screen layout")
    bg_clip = _get_background_clip(input_clip.duration, background_path, cfg)
    _, bg_h = bg_clip.size
    target_h = cfg["resolution_h"] - bg_h
    main_clip = _crop_clip(input_clip, cfg["resolution_w"], target_h)
    combined = clips_array([[main_clip], [bg_clip]])

    log("[3/4] Transcribing & adding captions")
    words = _transcribe_with_words(audio, cfg, log)
    final = _add_captions(combined, words, cfg, log)

    log("[4/4] Saving video...")
    safe_name = video_path.stem[:60]
    output_path = OUTPUT_DIR / f"{safe_name}_short.mp4"

    end_time = round(((final.duration * 100 // final.fps) * final.fps / 100), 2)
    final = final.subclipped(t_end=end_time)

    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=final.fps,
        threads=cfg["num_threads"],
        verbose=False,
        logger=None,
    )

    input_clip.close()
    log(f"  Saved to: {output_path}")
    return output_path
