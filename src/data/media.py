"""ffmpeg wrapper for notebook playback."""

import subprocess


def run_ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True, capture_output=True)
