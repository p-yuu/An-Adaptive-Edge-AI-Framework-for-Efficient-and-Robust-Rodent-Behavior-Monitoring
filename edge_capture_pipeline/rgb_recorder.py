import argparse
import csv
import json
import logging
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

STOP = False

def stop_handler(signum, frame):
    global STOP
    STOP = True


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def log_event(path, event, **kwargs):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now_iso(), "event": event, **kwargs}, ensure_ascii=False) + "\n")


def check_tools():
    missing = [name for name in ("ffmpeg", "ffprobe", "adb") if not shutil.which(name)]
    if missing:
        raise RuntimeError("Missing PATH commands: " + ", ".join(missing))
    

def forward_adb():
    r = subprocess.run(["adb", "forward", "tcp:8554", "tcp:554"], capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise RuntimeError("ADB forward failed: " + (r.stderr or r.stdout).strip())


def inspect_segments(session, csv_path, event_path):
    files = sorted(session.glob("rgb_*.mp4"))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "bytes", "duration_media_s", "frames", "width", "height", "avg_frame_rate", "decode_ok"])
        for path in files:
            try:
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
                    "-of", "json", str(path)], capture_output=True, text=True, timeout=90)
                if probe.returncode:
                    raise RuntimeError(probe.stderr.strip())
                info = json.loads(probe.stdout)
                stream = info.get("streams", [{}])[0]
                decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=300)
                writer.writerow([path.name, path.stat().st_size, info.get("format", {}).get("duration", ""),
                                 stream.get("nb_read_frames", ""), stream.get("width", ""), stream.get("height", ""),
                                 stream.get("avg_frame_rate", ""), decode.returncode == 0])
                log_event(event_path, "segment_checked", filename=path.name, decode_ok=decode.returncode == 0)
            except Exception as exc:
                writer.writerow([path.name, path.stat().st_size, "", "", "", "", "", False])
                log_event(event_path, "segment_check_failed", filename=path.name, error=str(exc))
    return len(files)


def main():
    global STOP
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="rgb_sessions", help="Root folder for session output")
    parser.add_argument("--segment-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=5)
    parser.add_argument("--url", default="rtsp://127.0.0.1:8554/live/0")
    parser.add_argument("--max-minutes", type=float, default=0, help="0 = until Ctrl+C")
    args = parser.parse_args()
    if args.segment_seconds <= 0 or args.retry_seconds <= 0 or args.max_minutes < 0:
        parser.error("segment-seconds/retry-seconds must be >0; max-minutes must be >=0")
    check_tools()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    session = Path(args.output).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    events = session / "events.jsonl"
    ffmpeg_log = session / "ffmpeg.log"
    metadata = {"created_local": now_iso(), "rtsp_url": args.url, "segment_target_s_media_time": args.segment_seconds,
                "note": "Host event timestamps are wall-clock, NOT sensor capture timestamps. MP4 PTS may not share thermal monotonic clock."}
    (session / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Session:", session, flush=True)
    log_event(events, "session_start")
    start = time.monotonic()
    attempt = 0
    try:
        while not STOP and (not args.max_minutes or time.monotonic() - start < args.max_minutes * 60):
            attempt += 1
            try:
                forward_adb()
            except Exception as exc:
                log_event(events, "adb_error", error=str(exc))
                print("ADB error:", exc, flush=True)
                time.sleep(args.retry_seconds)
                continue
            # Unique filenames on restart; never overwrite previously completed clips.
            pattern = str(session / f"rgb_run{attempt:03d}_%05d.mp4")
            cmd = ["ffmpeg", "-hide_banner", "-rtsp_transport", "tcp", "-i", args.url,
                   "-map", "0:v:0", "-c:v", "copy", "-an", "-f", "segment", "-segment_time", str(args.segment_seconds),
                   "-reset_timestamps", "1", pattern]
            log_event(events, "ffmpeg_start", attempt=attempt, output_pattern=pattern)
            print(f"FFmpeg attempt {attempt}: recording...", flush=True)
            with ffmpeg_log.open("a", encoding="utf-8") as logfile:
                logfile.write(f"\n===== attempt {attempt} {now_iso()} =====\n")
                logfile.flush()
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=logfile, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if STOP or (args.max_minutes and time.monotonic() - start >= args.max_minutes * 60):
                        # Ask FFmpeg to finalize the current MP4 before exiting.
                        try:
                            proc.stdin.write(b"q\n")
                            proc.stdin.flush()
                            proc.wait(timeout=15)
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            log_event(events, "graceful_stop_failed", attempt=attempt, error=str(exc))
                            if proc.poll() is None:
                                proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                        log_event(events, "stop_result", attempt=attempt, returncode=proc.returncode)
                        break
                    time.sleep(0.5)
                log_event(events, "ffmpeg_exit", attempt=attempt, returncode=proc.returncode)
            if STOP or (args.max_minutes and time.monotonic() - start >= args.max_minutes * 60):
                break
            print(f"FFmpeg exited ({proc.returncode}); retry in {args.retry_seconds}s", flush=True)
            time.sleep(args.retry_seconds)
    finally:
        log_event(events, "session_end")
        print("Checking segments (may take time)...", flush=True)
        count = inspect_segments(session, session / "segments.csv", events)
        print(f"Finished. {count} segments. Report: {session / 'segments.csv'}", flush=True)

if __name__ == "__main__":
    main()
