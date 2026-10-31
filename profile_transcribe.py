import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8080"
AUDIO_FILE = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\audio\audio2.wav"

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
    raise FileNotFoundError("Could not find ffmpeg.exe.")

def get_audio_duration(ffmpeg_bin, audio_file):
    cmd = [ffmpeg_bin, "-i", str(audio_file), "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
    if m:
        h, m_val, s = m.groups()
        return int(h) * 3600 + int(m_val) * 60 + float(s)
    import wave
    with wave.open(str(audio_file), 'rb') as w:
        return w.getnframes() / float(w.getframerate())

def find_silence_cut_points(ffmpeg_bin, audio_file, total_duration, target_chunk_sec=1200, min_silence_sec=0.5):
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

def transcribe_audio_via_api(wav_path, server_url):
    boundary = "----WebKitFormBoundary" + hex(int(time.time() * 1000))[2:]
    body = bytearray()
    
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
    body.extend(b'srt\r\n')
    
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
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return resp.read().decode("utf-8")

def main():
    profile = {
        "audio_file": AUDIO_FILE,
        "stages": {},
        "chunks": [],
        "summary": {}
    }
    
    total_start_time = time.time()
    ffmpeg_bin = find_ffmpeg()
    
    print("\n" + "="*70)
    print("PROFILING RUN: Nemotron-3.5-30B Transcription via Persistent Server")
    print("="*70)
    
    # Check server
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
            print(f"[+] NeMo 30B Server status: {data.get('status')} (version {data.get('version')})")
    except Exception as e:
        print(f"[!] Server not accessible: {e}")
        sys.exit(1)
        
    # Audio duration
    total_audio_duration = get_audio_duration(ffmpeg_bin, AUDIO_FILE)
    profile["total_audio_duration_sec"] = round(total_audio_duration, 2)
    print(f"[*] Audio Total Duration: {total_audio_duration:.2f}s ({total_audio_duration/60:.2f} min)")
    
    # Stage 1: Segmentation
    t0_seg = time.time()
    cuts = find_silence_cut_points(ffmpeg_bin, AUDIO_FILE, total_audio_duration, target_chunk_sec=1200)
    t_cuts = time.time() - t0_seg
    profile["stages"]["silence_detection_sec"] = round(t_cuts, 2)
    print(f"[+] Silence cuts detected in {t_cuts:.2f}s: {[round(c, 1) for c in cuts]}")
    
    work_dir = Path("workspace_chunks") / "audio2_profile"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Splitting
    t0_split = time.time()
    chunk_meta = []
    for idx in range(len(cuts) - 1):
        start = cuts[idx]
        duration = cuts[idx + 1] - start
        chunk_wav = work_dir / f"chunk_{idx:03d}.wav"
        if not chunk_wav.exists() or chunk_wav.stat().st_size < 1000:
            cmd = [
                ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-i", AUDIO_FILE,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(chunk_wav)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        chunk_meta.append({
            "index": idx,
            "path": chunk_wav,
            "start": start,
            "end": cuts[idx + 1],
            "duration": duration,
            "size_bytes": chunk_wav.stat().st_size
        })
    t_split = time.time() - t0_split
    profile["stages"]["chunk_extraction_sec"] = round(t_split, 2)
    print(f"[+] 7 chunks extracted in {t_split:.2f}s")
    
    # Stage 2: Server Transcription
    print("\n" + "-"*70)
    print(f"{'Chunk':<8} {'Audio Dur':<12} {'Infer Time':<12} {'Speedup (RTF)':<16} {'Cues':<8}")
    print("-"*70)
    
    total_infer_time = 0.0
    all_srt_results = []
    
    for c in chunk_meta:
        t0_chunk = time.time()
        srt_content = transcribe_audio_via_api(c["path"], SERVER_URL)
        infer_duration = time.time() - t0_chunk
        total_infer_time += infer_duration
        
        # Count cues
        cues = [b for b in srt_content.replace('\r\n', '\n').strip().split('\n\n') if '-->' in b]
        cue_count = len(cues)
        speedup = c["duration"] / infer_duration if infer_duration > 0 else 0
        rtf = infer_duration / c["duration"] if c["duration"] > 0 else 0
        
        print(f"#{c['index']:<7} {c['duration']:>8.1f}s    {infer_duration:>8.1f}s    {speedup:>5.2f}x (RTF {rtf:.3f})  {cue_count:>5}")
        
        c_prof = {
            "chunk_index": c["index"],
            "audio_duration_sec": round(c["duration"], 2),
            "infer_time_sec": round(infer_duration, 2),
            "speedup_factor": round(speedup, 2),
            "rtf": round(rtf, 4),
            "cue_count": cue_count
        }
        profile["chunks"].append(c_prof)
        all_srt_results.append((srt_content, c["start"]))
        
    print("-"*70)
    profile["stages"]["total_inference_sec"] = round(total_infer_time, 2)
    
    # Stage 3: Subtitle Stitching
    t0_stitch = time.time()
    final_output = Path("audio2_profiled.srt").resolve()
    total_cues = 0
    with open(final_output, "w", encoding="utf-8") as out_f:
        cue_idx = 1
        for srt_text, offset in all_srt_results:
            blocks = srt_text.replace('\r\n', '\n').strip().split('\n\n')
            for b in blocks:
                lines = b.strip().split('\n')
                if len(lines) >= 2 and '-->' in lines[1]:
                    time_parts = [t.strip() for t in lines[1].split('-->')]
                    st_sec = parse_time_srt(time_parts[0]) + offset
                    en_sec = parse_time_srt(time_parts[1]) + offset
                    body = "\n".join(lines[2:]).strip()
                    if body:
                        out_f.write(f"{cue_idx}\n{format_time_srt(st_sec)} --> {format_time_srt(en_sec)}\n{body}\n\n")
                        cue_idx += 1
        total_cues = cue_idx - 1
        
    t_stitch = time.time() - t0_stitch
    profile["stages"]["stitching_sec"] = round(t_stitch, 4)
    
    total_elapsed = time.time() - total_start_time
    overall_speedup = total_audio_duration / total_infer_time if total_infer_time > 0 else 0
    overall_rtf = total_infer_time / total_audio_duration if total_audio_duration > 0 else 0
    
    profile["summary"] = {
        "total_wall_clock_time_sec": round(total_elapsed, 2),
        "total_wall_clock_time_min": round(total_elapsed / 60, 2),
        "total_inference_time_min": round(total_infer_time / 60, 2),
        "total_audio_duration_min": round(total_audio_duration / 60, 2),
        "overall_speedup": round(overall_speedup, 2),
        "overall_rtf": round(overall_rtf, 4),
        "total_cues": total_cues,
        "output_file": str(final_output)
    }
    
    # Save JSON report
    with open("transcribe_profile_results.json", "w", encoding="utf-8") as jf:
        json.dump(profile, jf, indent=2)
        
    print(f"\n=======================================================")
    print(f"PROFILING SUMMARY")
    print(f"=======================================================")
    print(f"Audio Duration      : {total_audio_duration/60:.2f} min ({total_audio_duration:.1f}s)")
    print(f"Inference Time      : {total_infer_time/60:.2f} min ({total_infer_time:.1f}s)")
    print(f"Wall Clock Time     : {total_elapsed/60:.2f} min ({total_elapsed:.1f}s)")
    print(f"Overall Speedup     : {overall_speedup:.2f}x real-time (RTF: {overall_rtf:.3f})")
    print(f"Total Subtitle Cues : {total_cues}")
    print(f"Output File         : {final_output}")
    print(f"JSON Report         : transcribe_profile_results.json")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
