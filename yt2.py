"""YT2 CLI — Command-line interface for the summarizer pipeline."""

import sys
import click
from summarizer import run_pipeline


@click.command()
@click.argument("url")
@click.option("--model", default="kimi-k2.5:cloud", help="Ollama model for summarization")
@click.option("--whisper-model", default="small.en", help="Whisper model size")
@click.option("--scene-threshold", default=27.0, help="Scene detection sensitivity (lower = more scenes)")
@click.option("--no-cache", is_flag=True, help="Re-download and re-process everything")
@click.option("--visual-log", is_flag=True, help="Generate frame-by-frame visual log (slower)")
@click.option("--open", "open_browser", is_flag=True, help="Open result in browser")
def main(url, model, whisper_model, scene_threshold, no_cache, visual_log, open_browser):
    """Summarize a YouTube video locally with keyframe images.

    Example: python yt2.py https://www.youtube.com/watch?v=VIDEO_ID
    """
    click.echo(f"\n{'='*60}")
    click.echo(f"  YT2 — Summarizing")
    click.echo(f"{'='*60}\n")

    try:
        output_path = run_pipeline(
            url, model=model, whisper_model=whisper_model,
            scene_threshold=scene_threshold, no_cache=no_cache,
            visual_log=visual_log, log=click.echo,
        )
    except Exception as e:
        click.echo(f"\nERROR: {e}", err=True)
        sys.exit(1)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Done! Output: {output_path}")
    click.echo(f"{'='*60}\n")

    if open_browser:
        import webbrowser
        webbrowser.open(str(output_path))


if __name__ == "__main__":
    main()
