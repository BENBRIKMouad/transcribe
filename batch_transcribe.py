import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
NEMO_BIN = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\bin\nemo-speech.exe"
MODEL_PATH = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\models\NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf"

def find_ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    candidates = [
        r"C:\Program Files\digiKam\ffmpeg.exe",
        r"C:\Users\mouad_benbrik\AppData\Local\Programs\Stremio\ffmpeg.exe"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find ffmpeg.exe. Please install or add to PATH.")

def is_server_running(url=DEFAULT_SERVER_URL):
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False

def start_server_daemon(port=8080):
    print(f"[*] Starting local NeMo 30B server on port {port}...", flush=True)
    cmd = [
        NEMO_BIN, "serve",
        "--asr-model", MODEL_PATH,
        "--device", "cuda",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-ui",
        "--asr.salm.max_sequences", "1",
        "--asr.salm.n_ctx_per_sequence", "28192",
        "--asr.salm.max_context", "28192",
        "--asr.salm.mtp_draft", "0",
        "--asr.salm.n_batch", "1024",
        "--asr.salm.encoder_max_batch", "8"
    ]
    log_file = Path("server.log").resolve()
    with open(log_file, "a", encoding="utf-8") as out:
        subprocess.Popen(cmd, stdout=out, stderr=out, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    
    print("[*] Waiting for server to initialize 30B model...", end="", flush=True)
    for _ in range(60):
        time.sleep(2)
        print(".", end="", flush=True)
        if is_server_running(f"http://127.0.0.1:{port}"):
            print(" [READY!]", flush=True)
            return True
    print("\n[!] Timed out waiting for server to start.", flush=True)
    return False

def get_audio_duration(ffmpeg_bin, audio_file):
    cmd = [
        ffmpeg_bin, "-i", str(audio_file),
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
    if m:
        h, m_val, s = m.groups()
        return int(h) * 3600 + int(m_val) * 60 + float(s)
    # fallback to wave if uncompressed
    try:
        import wave
        with wave.open(str(audio_file), 'rb') as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        raise RuntimeError(f"Could not determine audio duration for {audio_file}")

def find_silence_cut_points(ffmpeg_bin, audio_file, total_duration, target_chunk_sec=1200, min_silence_sec=0.5):
    print(f"[*] Detecting natural speech pauses...", flush=True)
    cmd = [
        ffmpeg_bin, "-i", str(audio_file),
        "-af", f"silencedetect=noise=-30dB:d={min_silence_sec}",
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    
    silence_ends = []
    for line in res.stderr.splitlines():
        if "silence_end:" in line:
            m = re.search(r"silence_end:\s*([\d\.]+)", line)
            if m:
                silence_ends.append(float(m.group(1)))
                
    cuts = [0.0]
    current_target = target_chunk_sec
    while current_target < total_duration:
        candidates = [t for t in silence_ends if abs(t - current_target) <= 120]
        if candidates:
            best_cut = min(candidates, key=lambda t: abs(t - current_target))
        else:
            broader = [t for t in silence_ends if abs(t - current_target) <= 300]
            if broader:
                best_cut = min(broader, key=lambda t: abs(t - current_target))
            else:
                best_cut = current_target
        cuts.append(best_cut)
        current_target = best_cut + target_chunk_sec
        
    cuts.append(total_duration)
    clean_cuts = []
    for c in cuts:
        if not clean_cuts or (c - clean_cuts[-1] > 30.0):
            clean_cuts.append(c)
    return clean_cuts

def parse_time_srt(ts_str):
    h, m, s = ts_str.strip().split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def format_time_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def transcribe_audio_via_api(wav_path, server_url, response_format="srt"):
    boundary = "----WebKitFormBoundary" + hex(int(time.time() * 1000))[2:]
    body = bytearray()
    
    # format field
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
    body.extend(f"{response_format}\r\n".encode())
    
    # file field
    with open(wav_path, "rb") as f:
        file_bytes = f.read()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{Path(wav_path).name}"\r\n'.encode())
    body.extend(b'Content-Type: audio/wav\r\n\r\n')
    body.extend(file_bytes)
    body.extend(b'\r\n')
    body.extend(f"--{boundary}--\r\n".encode())
    
    req = urllib.request.Request(
        f"{server_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        return resp.read().decode("utf-8")

def process_file(audio_path, ffmpeg_bin, server_url, output_path=None, response_format="srt"):
    audio_path = Path(audio_path).resolve()
    print(f"\n=======================================================", flush=True)
    print(f"[*] Processing: {audio_path.name}", flush=True)
    
    total_duration = get_audio_duration(ffmpeg_bin, audio_path)
    print(f"[*] Duration: {total_duration:.1f}s ({total_duration / 60:.2f} min)", flush=True)
    
    ext = response_format if response_format != "text" else "txt"
    if not output_path:
        output_path = audio_path.with_suffix(f".{ext}")
    else:
        output_path = Path(output_path).resolve()
        
    temp_dir = Path("workspace_chunks") / audio_path.stem
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # If audio is under 20 minutes, transcribe in single direct API call
    if total_duration <= 1200:
        print("[*] Audio <= 20 min, sending directly to 30B server...", flush=True)
        wav_path = temp_dir / "full_16k.wav"
        if not wav_path.exists():
            subprocess.run([ffmpeg_bin, "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        res_text = transcribe_audio_via_api(wav_path, server_url, response_format)
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(res_text)
        print(f"[SUCCESS] Saved transcript: {output_path}", flush=True)
        return

    # Long audio: silence detection & chunking
    cuts = find_silence_cut_points(ffmpeg_bin, audio_path, total_duration, target_chunk_sec=1200)
    print(f"[*] Split into {len(cuts) - 1} chunks with zero word clipping.", flush=True)
    
    chunk_transcripts = []
    chunk_offsets = []
    
    for idx in range(len(cuts) - 1):
        start = cuts[idx]
        duration = cuts[idx + 1] - start
        chunk_wav = temp_dir / f"chunk_{idx:03d}.wav"
        chunk_txt = temp_dir / f"chunk_{idx:03d}.{ext}"
        
        chunk_offsets.append(start)
        
        if chunk_txt.exists() and chunk_txt.stat().st_size > 50:
            print(f"[*] [{idx+1}/{len(cuts)-1}] Chunk {idx:03d} cached, skipping...", flush=True)
            with open(chunk_txt, "r", encoding="utf-8") as f:
                chunk_transcripts.append(f.read())
            continue
            
        print(f"[*] [{idx+1}/{len(cuts)-1}] Extracting chunk {idx:03d} ({start:.1f}s -> {cuts[idx+1]:.1f}s)...", flush=True)
        if not chunk_wav.exists() or chunk_wav.stat().st_size < 1000:
            cmd = [
                ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(chunk_wav)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        print(f"[*] [{idx+1}/{len(cuts)-1}] Streaming chunk {idx:03d} to 30B server...", flush=True)
        t0 = time.time()
        res_text = transcribe_audio_via_api(chunk_wav, server_url, response_format)
        elapsed = time.time() - t0
        print(f"[+] [{idx+1}/{len(cuts)-1}] Chunk {idx:03d} completed in {elapsed:.1f}s", flush=True)
        
        with open(chunk_txt, "w", encoding="utf-8") as f:
            f.write(res_text)
        chunk_transcripts.append(res_text)

    # Merge SRT with shifted timestamps
    print(f"[*] Stitching {len(chunk_transcripts)} chunks into final output...", flush=True)
    if response_format == "srt":
        cue_idx = 1
        with open(output_path, "w", encoding="utf-8") as out_f:
            for text_data, offset in zip(chunk_transcripts, chunk_offsets):
                blocks = text_data.replace('\r\n', '\n').strip().split('\n\n')
                for b in blocks:
                    lines = b.strip().split('\n')
                    if len(lines) >= 2 and '-->' in lines[1]:
                        time_parts = [t.strip() for t in lines[1].split('-->')]
                        st_sec = parse_time_srt(time_parts[0]) + offset
                        en_sec = parse_time_srt(time_parts[1]) + offset
                        body_txt = "\n".join(lines[2:]).strip()
                        if body_txt:
                            out_f.write(f"{cue_idx}\n{format_time_srt(st_sec)} --> {format_time_srt(en_sec)}\n{body_txt}\n\n")
                            cue_idx += 1
    elif response_format in ["text", "txt"]:
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n\n".join(chunk_transcripts))
            
    print(f"[SUCCESS] Complete! Result written to: {output_path}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="High-Speed Batch Audio Transcriber using Nemotron-30B Server")
    parser.add_argument("input", help="Path to an audio/video file OR directory of files")
    parser.add_argument("--format", "-f", default="srt", choices=["srt", "vtt", "json", "text"], help="Subtitle/Transcript format")
    parser.add_argument("--output", "-o", default=None, help="Explicit output file or directory")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="NeMo serve URL (default: http://127.0.0.1:8080)")
    args = parser.parse_args()

    ffmpeg_bin = find_ffmpeg()
    
    if not is_server_running(args.server):
        print(f"[!] Server not detected at {args.server}. Auto-launching...")
        started = start_server_daemon()
        if not started:
            sys.exit(1)
            
    input_path = Path(args.input).resolve()
    if input_path.is_dir():
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov"}
        files = [p for p in input_path.iterdir() if p.suffix.lower() in audio_extensions]
        print(f"[*] Found {len(files)} media files in {input_path}")
        for f in files:
            process_file(f, ffmpeg_bin, args.server, response_format=args.format)
    elif input_path.is_file():
        process_file(input_path, ffmpeg_bin, args.server, output_path=args.output, response_format=args.format)
    else:
        print(f"Error: {input_path} is neither a valid file nor directory.")

if __name__ == "__main__":
    main()
