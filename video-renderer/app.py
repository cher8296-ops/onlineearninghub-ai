import json
import os
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="FlowChera21 Video Renderer", version="0.1.0")
MAX_PROJECT_BYTES = 100_000
MAX_AUDIO_BYTES = 80 * 1024 * 1024
MAX_SCENES = 80
MAX_DURATION_SECONDS = int(os.getenv("MAX_RENDER_SECONDS", "300"))
MAX_SCENE_SECONDS = 180
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
PALETTES = {
    "cinematic": ["0x10172e", "0x273d63"],
    "educational": ["0x123b35", "0x287a69"],
    "action": ["0x24132f", "0xa13c4d"],
    "documentary": ["0x29291f", "0x6c6548"],
    "professional": ["0x102b46", "0x28749a"],
}


def clean_text(value: Any, limit: int = 1000) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


def run_ffmpeg(args: list[str]) -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
            capture_output=True, text=True, timeout=240, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "Video rendering exceeded the processing time limit.") from exc
    if result.returncode:
        raise HTTPException(500, "FFmpeg rendering failed: " + (result.stderr or "unknown error")[-1200:])


def srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    whole, millis = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{whole:02},{millis:03}"


def remove_tree(path: str) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


@app.get("/health")
def health():
    try:
        check = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        available = check.returncode == 0
        version = (check.stdout.splitlines()[0] if available and check.stdout else "unknown")
    except Exception:
        available, version = False, "unavailable"
    return {"status": "healthy" if available else "degraded", "ffmpeg": available, "version": version,
            "max_render_seconds": MAX_DURATION_SECONDS, "audio_input": "mp3,wav,m4a,aac,ogg",
            "voice_generation": False, "note": "This service renders uploaded narration; it does not synthesize AI speech."}


@app.post("/render")
async def render(project: str = Form(...), audio: UploadFile | None = File(None)):
    if len(project.encode("utf-8")) > MAX_PROJECT_BYTES:
        raise HTTPException(413, "Project description is too large.")
    try:
        data = json.loads(project)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Project must be valid JSON.") from exc

    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes or len(scenes) > MAX_SCENES:
        raise HTTPException(422, f"Provide between 1 and {MAX_SCENES} scenes.")
    normalized = []
    total = 0
    for item in scenes:
        if not isinstance(item, dict):
            raise HTTPException(422, "Every scene must be an object.")
        seconds = item.get("seconds", 5)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise HTTPException(422, "Scene duration must be a number.")
        seconds = max(1, min(MAX_SCENE_SECONDS, int(seconds)))
        total += seconds
        normalized.append({
            "title": clean_text(item.get("title"), 120) or "Scene",
            "visual": clean_text(item.get("visual"), 240),
            "narration": clean_text(item.get("narration"), 1200),
            "seconds": seconds,
        })
    if total > MAX_DURATION_SECONDS:
        raise HTTPException(422, f"This renderer currently supports up to {MAX_DURATION_SECONDS // 60} minutes per video. Longer projects need queued rendering infrastructure.")

    locale = str(data.get("locale", "en")).lower()
    font = "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf" if locale == "am" else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not Path(font).exists():
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    palette = PALETTES.get(str(data.get("tone", "cinematic")), PALETTES["cinematic"])
    title = clean_text(data.get("title"), 120) or "FlowChera21 Video"
    workdir = tempfile.mkdtemp(prefix="flowchera21-")
    try:
        segments = []
        subtitle_entries = []
        cursor = 0
        for index, scene in enumerate(normalized):
            base = Path(workdir) / f"scene-{index:03}.mp4"
            title_file = Path(workdir) / f"title-{index:03}.txt"
            visual_file = Path(workdir) / f"visual-{index:03}.txt"
            title_file.write_text(title.replace("\n", " "), encoding="utf-8")
            visual_file.write_text(scene["visual"], encoding="utf-8")
            duration = scene["seconds"]
            bg = palette[index % len(palette)]
            vf = (
                f"drawbox=x='mod(t*100,1100)':y=565:w=180:h=6:color=0x9fe7d3@0.9:t=fill,"
                f"drawtext=fontfile={font}:textfile={title_file}:fontcolor=white:fontsize=44:x=70:y=110:box=1:boxcolor=black@0.20:boxborderw=18,"
                f"drawtext=fontfile={font}:textfile={visual_file}:fontcolor=0xd9e7ff:fontsize=28:x=75:y=300:box=1:boxcolor=black@0.22:boxborderw=14,"
                f"fade=t=in:st=0:d=0.45,fade=t=out:st={max(0, duration-0.45)}:d=0.45,format=yuv420p"
            )
            run_ffmpeg(["-f", "lavfi", "-i", f"color=c={bg}:s=1280x720:r=24:d={duration}",
                        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "24", "-t", str(duration), str(base)])
            segments.append(base)
            if scene["narration"]:
                subtitle_entries.append(
                    f"{len(subtitle_entries)+1}\n{srt_time(cursor)} --> {srt_time(cursor+duration)}\n{scene['narration']}\n"
                )
            cursor += duration

        concat_file = Path(workdir) / "concat.txt"
        concat_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8")
        joined = Path(workdir) / "joined.mp4"
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(joined)])

        final = Path(workdir) / "flowchera21-production.mp4"
        subtitle_file = Path(workdir) / "captions.srt"
        args = ["-i", str(joined)]
        if subtitle_entries:
            subtitle_file.write_text("\ufeff" + "\n".join(subtitle_entries), encoding="utf-8")
            args += ["-vf", f"subtitles={subtitle_file}:force_style='FontName=DejaVu Sans,FontSize=22,Outline=2,Shadow=1,Alignment=2'"]
        if audio is not None:
            suffix = Path(audio.filename or "").suffix.lower()
            if suffix not in ALLOWED_AUDIO:
                raise HTTPException(415, "Upload MP3, WAV, M4A, AAC or OGG audio.")
            audio_path = Path(workdir) / ("narration" + suffix)
            size = 0
            with audio_path.open("wb") as output:
                while True:
                    chunk = await audio.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_AUDIO_BYTES:
                        raise HTTPException(413, "Narration audio must be 80 MB or smaller.")
                    output.write(chunk)
            args += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-af", "apad"]
        else:
            args += ["-map", "0:v:0", "-an"]
        args += ["-t", str(total), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if audio is not None:
            args += ["-c:a", "aac", "-b:a", "160k"]
        args.append(str(final))
        run_ffmpeg(args)
        return FileResponse(
            final, media_type="video/mp4", filename="flowchera21-video.mp4",
            background=BackgroundTask(remove_tree, workdir),
        )
    except Exception:
        remove_tree(workdir)
        raise


@app.get("/")
def root():
    return {"service": "FlowChera21 Video Renderer", "status": "ready", "docs": "/docs"}
