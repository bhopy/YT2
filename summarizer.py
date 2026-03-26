"""Summarizer pipeline — deep video analysis with hierarchical chapters."""

import base64
import json
import re
import subprocess
from pathlib import Path

import requests
from jinja2 import Environment, FileSystemLoader

SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_DIR = SCRIPT_DIR / "output" / "summaries"
TEMPLATE_DIR = SCRIPT_DIR / "templates"

OLLAMA_URL = "http://localhost:11434"


def fmt_time(seconds):
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_time(t):
    """Parse MM:SS or H:MM:SS back to seconds."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0


def extract_video_id(url):
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


# ── Download ─────────────────────────────────────────────────────────────────


def download_video(url, video_id, log=print):
    vid_dir = CACHE_DIR / video_id
    vid_dir.mkdir(parents=True, exist_ok=True)

    video_path = vid_dir / "video.mp4"
    audio_path = vid_dir / "audio.wav"
    info_path = vid_dir / "info.json"

    if not video_path.exists():
        log("  Downloading video...")
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(video_path),
                "--write-info-json",
                "--no-playlist",
                url,
            ],
            check=True,
            capture_output=True,
        )
        ytdlp_info = vid_dir / "video.info.json"
        if ytdlp_info.exists():
            ytdlp_info.rename(info_path)

    if not audio_path.exists():
        log("  Extracting audio...")
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path), "-y",
            ],
            check=True,
            capture_output=True,
        )

    metadata = {"title": video_id, "duration": 0}
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
            metadata["title"] = info.get("title", video_id)
            metadata["duration"] = info.get("duration", 0)

    return video_path, audio_path, metadata


# ── Transcribe ───────────────────────────────────────────────────────────────


def transcribe_audio(audio_path, whisper_model="small.en", log=print):
    log(f"  Transcribing with whisper ({whisper_model})...")
    import whisper

    model = whisper.load_model(whisper_model)
    result = model.transcribe(str(audio_path), verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return segments


# ── Scene Detection ──────────────────────────────────────────────────────────


def detect_scenes(video_path, threshold=27.0, log=print):
    log("  Detecting scene changes...")
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)

    scene_list = scene_manager.get_scene_list()
    timestamps = [scene[0].get_seconds() for scene in scene_list]
    if not timestamps or timestamps[0] > 1.0:
        timestamps.insert(0, 0.0)

    log(f"  Found {len(timestamps)} scene changes")
    return timestamps


# ── Keyframes (scene + interval) ─────────────────────────────────────────────


def extract_keyframes(video_path, scene_timestamps, duration, video_id, log=print):
    """Extract frames at scene changes AND at regular intervals for full coverage."""
    log("  Extracting keyframes...")
    frames_dir = CACHE_DIR / video_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Combine scene timestamps with regular 20-second intervals
    interval = 20
    interval_timestamps = [i * interval for i in range(int(duration // interval) + 1)]
    all_timestamps = sorted(set(scene_timestamps + interval_timestamps))

    frame_paths = []
    for i, ts in enumerate(all_timestamps):
        if ts > duration:
            break
        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        if not frame_path.exists():
            subprocess.run(
                [
                    "ffmpeg", "-ss", str(ts),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", "scale='min(1280,iw)':-2",
                    "-q:v", "3",
                    str(frame_path), "-y",
                ],
                capture_output=True,
            )
        if frame_path.exists():
            frame_paths.append((ts, frame_path))

    log(f"  Extracted {len(frame_paths)} keyframes ({len(scene_timestamps)} scenes + intervals)")
    return frame_paths


# ── Ollama ───────────────────────────────────────────────────────────────────


def check_ollama(model):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False, "Ollama not responding"
        models = [m["name"] for m in r.json().get("models", [])]
        for m in models:
            if m == model or m.startswith(model + ":") or model.startswith(m.split(":")[0]):
                return True, None
        return False, f"Model '{model}' not found. Available: {', '.join(models)}\n  Pull it with: ollama pull {model}"
    except Exception:
        return False, "Ollama is not running. Start it with: ollama serve"


def _strip_thinking(text):
    """Strip <think>...</think> reasoning blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if text.strip().startswith("<think>"):
        parts = text.split("</think>", 1)
        if len(parts) > 1:
            text = parts[1]
    return text.strip()


def ollama_chat(messages, model, temperature=0.3, num_ctx=16384, num_predict=4096, timeout=300):
    """Send a chat request to Ollama. Supports vision via 'images' key in messages."""
    # Thinking models (kimi, qwen-think, etc.) count thinking tokens against num_predict,
    # so we need much higher limits to avoid empty responses.
    is_cloud = ":cloud" in model.lower()
    is_thinking = model.split(":")[0].lower() in {"kimi-k2.5", "deepseek-r1", "qwq"}
    if is_cloud or is_thinking:
        num_predict = max(8192, num_predict * 4)  # 4x minimum 8K for thinking overhead

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }

    import time as _time
    max_retries = 6
    for attempt in range(max_retries):
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        if r.status_code == 429 and attempt < max_retries - 1:
            wait = [15, 30, 60, 90, 120][min(attempt, 4)]
            _time.sleep(wait)
            continue
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        return _strip_thinking(content)


def _check_vision_support(model):
    """Check if an Ollama model supports vision/image input."""
    VISION_MODELS = {"kimi-k2.5", "llava", "llava-llama3", "bakllava",
                     "moondream", "llava-phi3", "minicpm-v", "qwen2-vl"}
    model_base = model.split(":")[0].lower()
    if model_base in VISION_MODELS:
        return True
    try:
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=10)
        if r.status_code == 200:
            info = r.json()
            template = info.get("template", "")
            if "{{ .Images }}" in template or "image" in template.lower():
                return True
            families = info.get("details", {}).get("families", [])
            if any("clip" in f.lower() or "vision" in f.lower() for f in families):
                return True
    except Exception:
        pass
    return False


def parse_json_response(text):
    text = _strip_thinking(text)
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def parse_json_array_response(text):
    """Parse a JSON array from model response (for visual log batches)."""
    text = _strip_thinking(text)
    text = text.strip()
    # Try ```json [...] ``` first
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw [...]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return []


# ── Vision Summarization Pipeline ────────────────────────────────────────────


def _build_timestamped_transcript(segments):
    """Build full transcript text with timestamps every ~30 seconds."""
    lines = []
    last_ts = -30
    for seg in segments:
        if seg["start"] - last_ts >= 30:
            lines.append(f"\n[{fmt_time(seg['start'])}]")
            last_ts = seg["start"]
        lines.append(seg["text"])
    return " ".join(lines)


def _target_chapter_count(duration):
    """Calculate ideal number of chapters based on video length."""
    minutes = duration / 60
    if minutes < 5:
        return 2, 3
    elif minutes < 15:
        return 3, 6
    elif minutes < 30:
        return 5, 10
    elif minutes < 60:
        return 8, 15
    elif minutes < 90:
        return 10, 20
    elif minutes < 120:
        return 12, 25
    else:
        return 15, 30


def _merge_small_chapters(chapters, min_count, max_count):
    """Post-process: merge if too many chapters, enforce limits."""
    if len(chapters) <= max_count:
        return chapters
    chapters = sorted(chapters, key=lambda c: parse_time(c.get("start", "0:00")))
    while len(chapters) > max_count:
        best_idx = 0
        best_dur = float("inf")
        for i in range(len(chapters) - 1):
            s = parse_time(chapters[i].get("start", "0:00"))
            e = parse_time(chapters[i + 1].get("end", "0:00"))
            dur = e - s
            if dur < best_dur:
                best_dur = dur
                best_idx = i
        a = chapters[best_idx]
        b = chapters[best_idx + 1]
        merged = {
            "start": a["start"],
            "end": b["end"],
            "title": a.get("title", ""),
            "summary": f"{a.get('summary', '')} {b.get('summary', '')}".strip(),
        }
        chapters = chapters[:best_idx] + [merged] + chapters[best_idx + 2:]
    return chapters


def _get_transcript_for_range(segments, start_sec, end_sec):
    """Extract timestamped transcript text for a specific time range."""
    range_segs = [s for s in segments if s["end"] > start_sec and s["start"] < end_sec]
    if not range_segs:
        return "(no transcript for this section)"
    lines = []
    last_ts = -15
    for seg in range_segs:
        if seg["start"] - last_ts >= 15:
            lines.append(f"\n[{fmt_time(seg['start'])}]")
            last_ts = seg["start"]
        lines.append(seg["text"])
    return " ".join(lines)


def _get_frames_for_range(keyframes, start_sec, end_sec, max_frames=3):
    """Pick the best keyframes for a time range, spaced evenly."""
    in_range = [(ts, p) for ts, p in keyframes if start_sec - 2 <= ts <= end_sec + 2]
    if not in_range:
        mid = (start_sec + end_sec) / 2
        nearest = min(keyframes, key=lambda x: abs(x[0] - mid), default=None)
        return [nearest] if nearest else []
    if len(in_range) <= max_frames:
        return in_range
    indices = [0]
    step = (len(in_range) - 1) / (max_frames - 1)
    for i in range(1, max_frames - 1):
        indices.append(round(i * step))
    indices.append(len(in_range) - 1)
    return [in_range[i] for i in sorted(set(indices))]


def _fallback_uniform_chapters(duration):
    """Create uniform time-based chapters when topic detection fails."""
    min_ch, max_ch = _target_chapter_count(duration)
    target = (min_ch + max_ch) // 2
    chunk_len = duration / target
    return [
        {"start": i * chunk_len, "end": min((i + 1) * chunk_len, duration),
         "title": f"Section {i + 1}"}
        for i in range(target)
    ]


# ── 2-Pass Knowledge Extraction Pipeline ───────────────────────────────────
#
# Pass 1: Chunked analysis — split video into ~5min windows, send transcript
#          + keyframe images to the model, extract structured observations.
# Pass 2: Synthesis — all chunk observations → brief + key_extractions + chapters
#
# Output: single JSON file with everything Claude Code needs.


def _make_time_chunks(duration, max_chunks=12):
    """Split video duration into analysis chunks."""
    chunk_minutes = max(3, min(10, duration / 60 / max_chunks))
    chunk_sec = chunk_minutes * 60
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_sec, duration)
        chunks.append((start, end))
        start = end
    return chunks


def _remap_extraction_type(ext):
    """Post-process: normalize extraction types to valid set, fix mistyped content."""
    etype = ext.get("type", "").strip().lower()
    content = ext.get("content", "")
    valid = {"command", "setting", "error", "tool", "resource", "takeaway", "fact"}
    # Map old/variant types to current set
    if etype in ("terminal_command", "shortcut", "ui_action"):
        ext["type"] = "command"
    elif etype == "opinion":
        ext["type"] = "takeaway"
    elif etype not in valid:
        ext["type"] = "fact"

    # Content-based correction: keyboard shortcuts mistyped as fact/setting → command
    if ext["type"] in ("fact", "setting"):
        if re.search(r'(?i)\b(meta|ctrl|control|alt|shift|super)\s*[+]', content):
            ext["type"] = "command"
    # Shell commands mistyped as fact → command
    if ext["type"] == "fact":
        if re.search(r'(?i)^(sudo|pacman|yay|apt|pip|npm|git|flatpak|snap)\s', content):
            ext["type"] = "command"

    # Desktop effects mistyped as tool → setting
    if ext["type"] == "tool" and re.search(r'(?i)\b(effect|wobbly|magic lamp|translucen|magnif)', content):
        ext["type"] = "setting"

    # Default outcome if missing
    if not ext.get("outcome"):
        ext["outcome"] = "info"
    # Normalize outcome to valid values
    valid_outcomes = {"success", "failed", "partial", "info"}
    if ext["outcome"].strip().lower() not in valid_outcomes:
        ext["outcome"] = "info"
    else:
        ext["outcome"] = ext["outcome"].strip().lower()

    # Auto-fix outcome from detail text when model defaulted to "info"
    detail = ext.get("detail", "").lower()
    if ext["outcome"] == "info" and ext["type"] in ("command", "tool", "setting"):
        if re.search(r'(success|worked|installed|enabled|configured|activated|applied|set to)', detail):
            ext["outcome"] = "success"
        elif re.search(r'(fail|error|crash|broke|not found|not work|cannot|refused|missing)', detail):
            ext["outcome"] = "failed"

    # Normalize shortcut content format: "Meta + Ctrl + I" → "Meta+Ctrl+I"
    if ext["type"] == "command" and "+" in content:
        ext["content"] = re.sub(r'\s*\+\s*', '+', content)

    return ext


def _analyze_chunk(idx, total, start, end, segments, keyframes, model, use_vision, log):
    """Analyze a single time chunk — extract observations and structured facts."""
    log(f"    Chunk {idx+1}/{total} ({fmt_time(start)} - {fmt_time(end)})")

    transcript_text = _get_transcript_for_range(segments, start, end)
    frames = _get_frames_for_range(keyframes, start, end, max_frames=3)

    # Collect transcript timestamp anchors (~every 15s) for synthesis to use
    chunk_segments = [s for s in segments if s["end"] > start and s["start"] < end]
    transcript_timestamps = []
    last_anchor = -15
    for seg in chunk_segments:
        if seg["start"] - last_anchor >= 15:
            transcript_timestamps.append(fmt_time(seg["start"]))
            last_anchor = seg["start"]

    images_b64 = []
    if use_vision:
        for ts, path in frames:
            try:
                images_b64.append(_compress_image_b64(path, max_width=800, quality=60))
            except Exception:
                pass

    image_note = ""
    if images_b64:
        image_note = f"\n\n{len(images_b64)} screenshots from this segment are attached. Read ALL text visible on screen."
    elif use_vision:
        image_note = "\n\nNo screenshots available for this segment."

    prompt = f"""Analyze this video segment ({fmt_time(start)} - {fmt_time(end)}).

TRANSCRIPT:
{transcript_text}{image_note}

Return ONLY this JSON:
{{
  "observations": "What is happening and WHY. What is the speaker trying to accomplish? What do they learn or discover? 3-5 sentences that tell the STORY of this segment.",
  "screen_content": "Describe what is visually shown — terminal output, UI panels, code, error dialogs, settings values. Only notable things. Write 'Nothing notable' if no screenshots.",
  "extractions": [
    {{"type": "command", "content": "exact command or shortcut", "detail": "what happened", "outcome": "success|failed|partial|info"}},
    {{"type": "setting", "content": "setting = value", "detail": "effect or why", "outcome": "success|failed"}},
    {{"type": "error", "content": "exact error message", "detail": "cause and resolution", "outcome": "failed|partial"}},
    {{"type": "tool", "content": "tool/app name", "detail": "what for and verdict", "outcome": "success|failed|partial"}},
    {{"type": "resource", "content": "URL, package, or path", "detail": "context", "outcome": "info"}},
    {{"type": "takeaway", "content": "lesson learned", "detail": "context", "outcome": "info"}}
  ]
}}

QUALITY FILTER — only extract things that pass this test:
"Would someone reference this fact when solving a problem, learning this topic, or replicating what was shown?"

YES — extract these:
- Commands that were run and their results (sudo pacman -S steam → target not found)
- Keyboard shortcuts with what they do (Meta+V → clipboard history)
- Settings that were changed and WHY (display scale = 135% because 5% increments available)
- Errors encountered and how they were resolved
- Tools/apps used and whether they worked well
- Specific technical values (resolution, FPS, versions, memory)
- Conclusions or lessons the speaker reached

NO — skip these:
- Trivial UI navigation (clicked a menu, opened settings, scrolled down)
- Cosmetic choices (picked this wallpaper, tried that theme, changed a color)
- Obvious facts (app has a menu bar, window has a close button)
- Anything a reader couldn't act on or learn from

Aim for 3-10 HIGH-VALUE extractions. Zero is fine if the segment is just chitchat."""

    msg = {"role": "user", "content": prompt}
    if images_b64:
        msg["images"] = images_b64

    try:
        response = ollama_chat([msg], model, num_ctx=16384, num_predict=2048, timeout=240)
        data = parse_json_response(response)
    except Exception as e:
        log(f"    Warning: Chunk analysis failed ({e})")
        data = {
            "observations": transcript_text[:300],
            "screen_content": "Analysis failed",
            "extractions": [],
        }

    # Post-process: reclassify any "command" the model still outputs
    extractions = [_remap_extraction_type(e) for e in data.get("extractions", [])]

    return {
        "start": start, "end": end,
        "start_fmt": fmt_time(start), "end_fmt": fmt_time(end),
        "observations": data.get("observations", ""),
        "screen_content": data.get("screen_content", ""),
        "extractions": extractions,
        "had_images": len(images_b64) > 0,
        "transcript_timestamps": transcript_timestamps,
    }


def _synthesize_knowledge(chunk_analyses, model, duration, title, log):
    """Pass 2: Synthesize all chunk observations into final knowledge document."""
    min_ch, max_ch = _target_chapter_count(duration)
    minutes = duration / 60

    # Build continuous timestamped stream using transcript anchors (every ~15s)
    # NOT chunk boundaries — so the model has fine-grained timestamps to choose from
    timeline = ""
    all_extractions = []
    all_anchors = []
    for ca in chunk_analyses:
        anchors = ca.get('transcript_timestamps', [ca['start_fmt']])
        all_anchors.extend(anchors)
        all_extractions.extend(ca['extractions'])

    # List all available timestamps upfront
    timeline += f"TIMELINE MARKERS: {', '.join(all_anchors)}\n"
    timeline += f"(Use any of these as chapter start/end times)\n\n"

    for ca in chunk_analyses:
        anchors = ca.get('transcript_timestamps', [])
        anchor_str = f"[{', '.join(anchors)}] " if anchors else f"[{ca['start_fmt']}] "
        timeline += f"{anchor_str}{ca['observations']}\n"
        if ca['screen_content'] and ca['screen_content'] != "Nothing notable":
            timeline += f"  (On screen: {ca['screen_content']})\n"
        for e in ca['extractions']:
            outcome = e.get('outcome', 'info')
            timeline += f"  → [{e.get('type', 'fact')}] {e.get('content', '')} — {e.get('detail', '')} | {outcome}\n"
        timeline += "\n"

    num_ctx = 32768 if minutes > 60 else 16384
    num_predict = 8192 if minutes > 60 else 4096

    prompt = f"""You are creating a reference document from a {fmt_time(duration)} video titled "{title}".

Someone will read this document INSTEAD of watching the video. They need to understand everything that happened — what was tried, what worked, what failed, what was learned.

Below is a timestamped timeline of everything observed:

{timeline}

Return ONLY this JSON:
{{
  "brief": "3-5 sentences. First sentence: who is doing what. Then: list the main topics covered, specific tools/software tested, and key problems encountered. End with the overall outcome/verdict. Be specific — name tools, name errors, name commands. An AI will read this FIRST to decide if this document is relevant.",
  "chapters": [
    {{
      "start_fmt": "M:SS",
      "end_fmt": "M:SS",
      "title": "Descriptive title",
      "summary": "4-8 sentences. Tell the complete story of this section: what was attempted, what happened on screen, what commands were run and their output, what errors occurred, what the outcome was. Weave in visual details naturally (e.g. 'the terminal shows error X' not just 'error X occurred'). A reader should feel like they watched this part of the video."
    }}
  ]
}}

RULES:
1. Create {min_ch} to {max_ch} chapters based on where TOPICS actually change.
2. Summaries should be DENSE with information — specific commands, exact error messages, settings values, tool names. No vague descriptions.
3. Chapters must cover the ENTIRE video start to end.
4. Pick chapter start/end times from the TIMELINE MARKERS list. Choose timestamps where topics ACTUALLY shift — NOT at regular intervals. Chapters should have VARIED lengths (some 2min, some 5min, some 8min). If two adjacent sections cover the same topic, MERGE them into one longer chapter."""

    response = ollama_chat(
        [{"role": "user", "content": prompt}], model,
        num_ctx=num_ctx, num_predict=num_predict, timeout=420)
    data = parse_json_response(response)

    # Fallbacks
    if not data.get("brief"):
        data["brief"] = f"A {fmt_time(duration)} video titled '{title}'."
    if not data.get("chapters"):
        data["chapters"] = [
            {"start_fmt": ca["start_fmt"], "end_fmt": ca["end_fmt"],
             "title": f"Section {i+1}", "summary": ca["observations"],
             }
            for i, ca in enumerate(chunk_analyses)
        ]

    return data, all_extractions


def extract_knowledge(segments, model, duration, keyframes, title="", log=print):
    """Main analysis function: 2-pass knowledge extraction.

    Pass 1: Chunked vision analysis → observations + extractions per chunk.
    Pass 2: Synthesis → brief + chapters + key_extractions.

    Returns a complete knowledge dict ready for JSON output.
    """
    use_vision = _check_vision_support(model)
    log(f"  Model: {model} | Vision: {'YES' if use_vision else 'no (text-only fallback)'}")

    total_words = len(" ".join(s["text"] for s in segments).split())
    log(f"  Transcript: {total_words} words | Keyframes: {len(keyframes)}")

    # Pass 1: Chunked analysis
    chunks = _make_time_chunks(duration)
    log(f"  Pass 1/2: Analyzing {len(chunks)} chunks...")
    chunk_analyses = []
    for i, (start, end) in enumerate(chunks):
        ca = _analyze_chunk(i, len(chunks), start, end, segments, keyframes,
                            model, use_vision, log)
        chunk_analyses.append(ca)

    # Pass 2: Synthesis
    log("  Pass 2/2: Synthesizing knowledge document...")
    synthesis, all_extractions = _synthesize_knowledge(
        chunk_analyses, model, duration, title, log)

    return {
        "synthesis": synthesis,
        "chunk_extractions": all_extractions,
        "chunk_analyses": chunk_analyses,
        "vision_used": use_vision and any(ca.get("had_images") for ca in chunk_analyses),
    }


# ── Keyframe Matching ────────────────────────────────────────────────────────



# ── Render ───────────────────────────────────────────────────────────────────


# ── Extraction Deduplication ─────────────────────────────────────────────


KNOWN_MODIFIERS = {'meta', 'ctrl', 'control', 'alt', 'shift', 'super', 'win'}
MOD_NORMALIZE = {'control': 'ctrl', 'win': 'meta', 'super': 'meta'}
MOD_ORDER = {'meta': 0, 'ctrl': 1, 'alt': 2, 'shift': 3}

VALID_EXTRACTION_TYPES = {
    "command", "setting", "error", "tool", "resource", "takeaway", "fact",
}


def _normalize_shortcut(text):
    """Normalize keyboard shortcut to canonical form: Meta+Ctrl+Shift+Key, lowercased."""
    normalized = text.lower().strip()
    # Split concatenated modifiers: "metashift" → "meta+shift"
    for m1 in KNOWN_MODIFIERS:
        for m2 in KNOWN_MODIFIERS:
            normalized = normalized.replace(m1 + m2, m1 + "+" + m2)
    # Split on + or spaces
    parts = [p.strip() for p in re.split(r'[+\s]+', normalized) if p.strip()]
    # Normalize modifier names
    parts = [MOD_NORMALIZE.get(p, p) for p in parts]
    # Sort modifiers, then key
    mods = sorted([p for p in parts if p in MOD_ORDER], key=lambda x: MOD_ORDER[x])
    keys = [p for p in parts if p not in MOD_ORDER]
    return "+".join(mods + keys)


def _normalize_extraction_key(ext):
    """Return normalized (type, content) for dedup comparison."""
    etype = ext.get("type", "").strip().lower()
    content = ext.get("content", "").strip()

    content_lower = content.lower().strip()

    # Shortcut-like content: has modifier+key pattern — normalize regardless of type
    if re.search(r'(?i)(meta|ctrl|control|alt|shift)\s*[+\s]', content):
        content_lower = _normalize_shortcut(content)
    else:
        content_lower = re.sub(r'\s+', ' ', content_lower).strip()

    # Ignore type for dedup key — same content = same extraction
    return content_lower


def _deduplicate_extractions(extractions):
    """Deduplicate extractions with fuzzy shortcut matching and substring containment."""
    # Group by normalized key, keep entry with longest detail
    groups = {}
    for ext in extractions:
        key = _normalize_extraction_key(ext)
        if key not in groups:
            groups[key] = []
        groups[key].append(ext)

    deduped = []
    for key, group in groups.items():
        best = max(group, key=lambda e: len(e.get("detail", "")))
        deduped.append(best)

    # Remove entries whose content is a substring of another entry (cross-type)
    final = []
    for i, ext in enumerate(deduped):
        content = re.sub(r'\s+', ' ', ext.get("content", "").strip().lower())
        is_substring = False
        for j, other in enumerate(deduped):
            if i == j:
                continue
            other_content = re.sub(r'\s+', ' ', other.get("content", "").strip().lower())
            if content != other_content and content in other_content:
                is_substring = True
                break
        if not is_substring:
            final.append(ext)

    # Filter out invalid types (e.g. "screen_content" that snuck through)
    final = [e for e in final if e.get("type", "").strip().lower() in VALID_EXTRACTION_TYPES]

    return final



def _compress_image_b64(frame_path, max_width=800, quality=60):
    """Read an image, resize it, and return compressed base64."""
    try:
        from PIL import Image
        import io

        img = Image.open(frame_path)
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(frame_path, "rb") as f:
            return base64.b64encode(f.read()).decode()



# ── Visual Log ───────────────────────────────────────────────────────────────


def _thin_frames(keyframes, min_gap=5):
    """Reduce keyframes to ~5-second intervals for visual log."""
    thinned = []
    last_ts = -min_gap
    for ts, path in keyframes:
        if ts - last_ts >= min_gap:
            thinned.append((ts, path))
            last_ts = ts
    return thinned


def generate_visual_log(keyframes, model, duration, log=print):
    """Generate frame-by-frame visual log — what's on screen at every moment.

    Batches 5 frames per vision call. Returns list of {"time": "M:SS", "screen": "..."}.
    """
    thinned = _thin_frames(keyframes, min_gap=5)
    log(f"  {len(keyframes)} keyframes -> {len(thinned)} after thinning (5s gap)")

    visual_log = []
    batch_size = 5
    batches = [thinned[i:i + batch_size] for i in range(0, len(thinned), batch_size)]

    for batch_idx, batch in enumerate(batches):
        timestamps = [fmt_time(ts) for ts, _ in batch]
        images_b64 = []
        for ts, path in batch:
            try:
                images_b64.append(_compress_image_b64(path, max_width=800, quality=60))
            except Exception:
                images_b64.append(None)

        # Skip batch if no images loaded
        valid_images = [img for img in images_b64 if img is not None]
        if not valid_images:
            for ts, _ in batch:
                visual_log.append({"time": fmt_time(ts), "screen": "(no frame)"})
            continue

        prompt = f"""You are watching a screen recording. For each screenshot, write ONE sentence describing what is visible.
Focus on: terminal commands/output, application names and state, settings panels and values, error dialogs and messages, file contents, code visible on screen.
Skip: wallpaper descriptions, mouse cursor position, UI chrome that isn't the subject.

Screenshots at: [{', '.join(timestamps)}]

Return ONLY a JSON array of {len(batch)} strings (one per screenshot, same order):
["Terminal showing pacman -S output with 3 packages installed", "System Settings > Appearance with Global Theme selected", ...]"""

        msg = {"role": "user", "content": prompt, "images": valid_images}

        try:
            response = ollama_chat([msg], model, num_ctx=8192, num_predict=2048, timeout=120)
            descriptions = parse_json_array_response(response)

            if len(descriptions) == len(batch):
                for (ts, _), desc in zip(batch, descriptions):
                    visual_log.append({"time": fmt_time(ts), "screen": str(desc).strip()})
            else:
                for j, (ts, _) in enumerate(batch):
                    if j < len(descriptions):
                        visual_log.append({"time": fmt_time(ts), "screen": str(descriptions[j]).strip()})
                    else:
                        visual_log.append({"time": fmt_time(ts), "screen": "(parse error)"})
        except Exception as e:
            log(f"    Warning: Visual log batch {batch_idx+1}/{len(batches)} failed ({e})")
            for ts, _ in batch:
                visual_log.append({"time": fmt_time(ts), "screen": "(failed)"})

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
            log(f"    Batch {batch_idx+1}/{len(batches)} done ({len(visual_log)} entries)")

        # Throttle to avoid cloud API rate limits
        if batch_idx < len(batches) - 1:
            _time.sleep(1)

    # Post-process: dedup consecutive similar entries and filter filler
    raw_count = len(visual_log)
    visual_log = _postprocess_visual_log(visual_log)
    log(f"  Post-processed: {raw_count} -> {len(visual_log)} entries ({raw_count - len(visual_log)} removed)")
    return visual_log


def _postprocess_visual_log(entries):
    """Remove consecutive near-duplicate entries and filter low-value filler."""
    if not entries:
        return entries

    # Filter filler entries (wallpaper-only, cursor-only, generic desktop)
    FILLER_PATTERNS = re.compile(
        r'(?i)^(desktop\s+(with|showing)\s+(default\s+)?wallpaper'
        r'|(?:just\s+)?(?:a\s+)?(?:default\s+)?desktop\s+(?:background|wallpaper)'
        r'|mouse\s+cursor\s+(?:on|over|hovering)'
        r'|empty\s+desktop'
        r'|same\s+(?:as|view)\b'
        r'|\(no frame\)|\(failed\)|\(parse error\))'
    )
    filtered = []
    for entry in entries:
        screen = entry.get("screen", "").strip()
        if not screen or FILLER_PATTERNS.search(screen):
            continue
        filtered.append(entry)

    # Dedup consecutive entries with very similar descriptions
    deduped = [filtered[0]] if filtered else []
    for i in range(1, len(filtered)):
        prev_screen = deduped[-1]["screen"].lower().strip()
        curr_screen = filtered[i]["screen"].lower().strip()
        # Skip if identical or one contains the other
        if curr_screen == prev_screen:
            continue
        if len(curr_screen) > 20 and len(prev_screen) > 20:
            # Check word overlap — if 80%+ words match, skip
            prev_words = set(prev_screen.split())
            curr_words = set(curr_screen.split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / min(len(prev_words), len(curr_words))
                if overlap > 0.8:
                    continue
        deduped.append(filtered[i])

    return deduped


def render_json(title, video_id, url, duration, model, knowledge, keyframes,
                output_dir, visual_log=None, log=print):
    """Render a single self-contained JSON file. No folders, no images, no extras.

    One file with everything Claude Code needs to fully understand the video.
    """
    log("  Rendering JSON...")
    from datetime import datetime

    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:80]
    json_path = output_dir / f"{safe_title}.json"

    synthesis = knowledge["synthesis"]
    vision_used = knowledge["vision_used"]

    # Deduplicate all extractions
    all_extractions = _deduplicate_extractions(knowledge.get("chunk_extractions", []))

    # Index extractions by time range for inline chapter assignment
    chunk_analyses = knowledge.get("chunk_analyses", [])
    def _extractions_for_range(start_sec, end_sec):
        """Get extractions that belong to a time range (from chunk analyses)."""
        result = []
        for ca in chunk_analyses:
            if ca["end"] <= start_sec or ca["start"] >= end_sec:
                continue
            result.extend(ca.get("extractions", []))
        return _deduplicate_extractions(result)

    # Build chapters with extractions inline (no cross-chapter duplicates)
    chapters = []
    assigned_keys = set()
    for i, ch in enumerate(synthesis.get("chapters", []), 1):
        start_sec = parse_time(ch.get("start_fmt", "0:00"))
        end_sec = parse_time(ch.get("end_fmt", "0:00"))

        # Get extractions for this chapter's time range, skip globally assigned
        ch_extractions = []
        for ext in _extractions_for_range(start_sec, end_sec):
            key = ext.get("content", "").strip().lower()
            if key not in assigned_keys:
                assigned_keys.add(key)
                ch_extractions.append(ext)

        chapters.append({
            "index": i,
            "start": ch.get("start_fmt", ""),
            "end": ch.get("end_fmt", ""),
            "title": ch.get("title", ""),
            "summary": ch.get("summary", ""),
            "extractions": ch_extractions,
        })

    # ── Build top-level structured reference sections from extractions ──
    # These make the file queryable, not just readable.

    # Flatten all chapter extractions for reference sections
    all_ch_extractions = []
    for ch in chapters:
        for ext in ch.get("extractions", []):
            all_ch_extractions.append({**ext, "_chapter": ch["index"], "_chapter_title": ch["title"]})

    # 1. Command Reference — all commands/shortcuts in one flat list
    commands_ref = []
    for ext in all_ch_extractions:
        if ext.get("type") == "command":
            commands_ref.append({
                "command": ext["content"],
                "action": ext.get("detail", ""),
                "outcome": ext.get("outcome", "info"),
            })

    # 2. Problems & Solutions — every error/failure paired with resolution
    problems = []
    for ext in all_ch_extractions:
        if ext.get("type") == "error" or ext.get("outcome") == "failed":
            problems.append({
                "problem": ext["content"],
                "detail": ext.get("detail", ""),
                "resolved": ext.get("outcome") != "failed",
                "chapter": ext["_chapter"],
            })

    # 3. Software Verdicts — every tool tested with one-line verdict
    software = []
    seen_tools = set()
    for ext in all_ch_extractions:
        if ext.get("type") == "tool":
            name = ext["content"].strip()
            if name.lower() not in seen_tools:
                seen_tools.add(name.lower())
                software.append({
                    "name": name,
                    "verdict": ext.get("detail", ""),
                    "outcome": ext.get("outcome", "info"),
                })

    # 4. Settings Changed — every setting modification
    settings = []
    for ext in all_ch_extractions:
        if ext.get("type") == "setting":
            settings.append({
                "setting": ext["content"],
                "detail": ext.get("detail", ""),
                "outcome": ext.get("outcome", "info"),
            })

    # 5. Key Takeaways — distilled lessons
    takeaways = []
    for ext in all_ch_extractions:
        if ext.get("type") == "takeaway":
            takeaways.append({
                "insight": ext["content"],
                "context": ext.get("detail", ""),
            })

    output = {
        "meta": {
            "title": title,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": duration,
            "duration_fmt": fmt_time(duration),
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "vision": vision_used,
        },
        "brief": synthesis.get("brief", ""),
        "reference": {
            "commands": commands_ref,
            "problems": problems,
            "software": software,
            "settings": settings,
            "takeaways": takeaways,
        },
        "chapters": chapters,
    }

    if visual_log:
        output["visual_log"] = visual_log

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log(f"  Output: {json_path}")
    return json_path


# ── Full Pipeline ────────────────────────────────────────────────────────────


def run_pipeline(url, model="kimi-k2.5:cloud", whisper_model="small.en",
                 scene_threshold=27.0, no_cache=False, visual_log=False, log=print):
    """Run the full knowledge extraction pipeline.

    Output: single .json file in output/summaries/
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")

    ok, err = check_ollama(model)
    if not ok:
        raise RuntimeError(err)

    # Pre-flight: wait for cloud API rate limits to clear
    if ":cloud" in model.lower():
        import time as _time
        for _wait_round in range(12):  # up to ~10 min
            _test = requests.post(f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                      "stream": False, "options": {"num_predict": 5}}, timeout=30)
            if _test.status_code != 429:
                break
            wait_s = 30 + _wait_round * 15  # 30, 45, 60, 75, ...
            log(f"  API rate-limited, waiting {wait_s}s before starting... ({_wait_round+1}/12)")
            _time.sleep(wait_s)
        else:
            raise RuntimeError("API still rate-limited after 10+ minutes. Try again later.")

    if no_cache:
        import shutil
        cache_path = CACHE_DIR / video_id
        if cache_path.exists():
            shutil.rmtree(cache_path)

    total_steps = 6 if visual_log else 5

    log(f"[1/{total_steps}] Download")
    video_path, audio_path, metadata = download_video(url, video_id, log)
    log(f"  Title: {metadata['title']}")
    log(f"  Duration: {fmt_time(metadata['duration'])}")

    log(f"[2/{total_steps}] Transcribe")
    transcript = transcribe_audio(audio_path, whisper_model, log)
    log(f"  Got {len(transcript)} segments")

    log(f"[3/{total_steps}] Scene Detection + Keyframes")
    scene_timestamps = detect_scenes(video_path, scene_threshold, log)
    keyframes = extract_keyframes(video_path, scene_timestamps,
                                  metadata["duration"], video_id, log)

    log(f"[4/{total_steps}] Knowledge Extraction")
    knowledge = extract_knowledge(transcript, model, metadata["duration"], keyframes,
                                  title=metadata["title"], log=log)
    ch_count = len(knowledge["synthesis"].get("chapters", []))
    ext_count = len(knowledge.get("chunk_extractions", []))
    log(f"  {ch_count} chapters, {ext_count} extractions")

    vlog = None
    if visual_log:
        log(f"[5/{total_steps}] Visual Log")
        vlog = generate_visual_log(keyframes, model, metadata["duration"], log)
        log(f"  {len(vlog)} visual log entries")

    log(f"[{total_steps}/{total_steps}] Render")
    json_path = render_json(
        metadata["title"], video_id, url, metadata["duration"], model,
        knowledge, keyframes, OUTPUT_DIR, visual_log=vlog, log=log)

    return json_path
