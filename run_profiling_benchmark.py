import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

NEMO_BIN = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\bin\nemo-speech.exe"
MODEL_PATH = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\models\NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf"
MANIFEST_PATH = Path(r"workspace_chunks\audio2\manifest.json")
PROFILE_JSON = Path("transcription_profile.json")

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

def get_audio_duration_wave(wav_path):
    import wave
    with wave.open(str(wav_path), 'rb') as w:
        return w.getnframes() / float(w.getframerate())

def run_chunk(idx):
    wav_path = Path(f"workspace_chunks/audio2/wavs/chunk_{idx:03d}.wav").resolve()
    out_path = Path(f"workspace_chunks/audio2/out_profile/chunk_{idx:03d}.srt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not wav_path.exists():
        print(f"Error: {wav_path} not found.")
        sys.exit(1)
        
    duration = get_audio_duration_wave(wav_path)
    file_size_mb = wav_path.stat().st_size / (1024 * 1024)
    
    print(f"===========================================================", flush=True)
    print(f"PROFILING CHUNK {idx:03d}", flush=True)
    print(f"File: {wav_path.name} ({file_size_mb:.2f} MB)", flush=True)
    print(f"Audio Duration: {duration:.2f}s ({duration/60:.2f} min)", flush=True)
    print(f"===========================================================", flush=True)
    
    cmd = [
        NEMO_BIN, "transcribe", str(wav_path),
        "--model", MODEL_PATH,
        "--device", "cuda",
        "--format", "srt",
        "--output", str(out_path),
        "--instruction", "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization, numbers in their standard written form. Output: transcript only.",
        "--asr.salm.max_sequences", "1",
        "--asr.salm.n_ctx_per_sequence", "28192",
        "--asr.salm.max_context", "28192",
        "--asr.salm.mtp_draft", "0",
        "--asr.salm.n_batch", "512",
        "--asr.salm.encoder_max_batch", "2"
    ]
    
    t_start = time.perf_counter()
    proc = subprocess.run(cmd)
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    if proc.returncode != 0:
        print(f"Failed with exit code: {proc.returncode}")
        sys.exit(proc.returncode)
        
    # Analyze output
    with open(out_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
        
    blocks = [b for b in content.replace('\r\n', '\n').split('\n\n') if '-->' in b]
    cue_count = len(blocks)
    speedup = duration / elapsed if elapsed > 0 else 0
    rtf = elapsed / duration if duration > 0 else 0
    
    chunk_stat = {
        "chunk_index": idx,
        "audio_duration_sec": round(duration, 2),
        "file_size_mb": round(file_size_mb, 2),
        "elapsed_sec": round(elapsed, 2),
        "speedup_factor": round(speedup, 2),
        "rtf": round(rtf, 4),
        "cue_count": cue_count,
        "output_srt": str(out_path)
    }
    
    # Save to JSON
    data = {}
    if PROFILE_JSON.exists():
        try:
            with open(PROFILE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    if "chunks" not in data:
        data["chunks"] = {}
    data["chunks"][str(idx)] = chunk_stat
    with open(PROFILE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"\n[+] Chunk {idx:03d} Result:", flush=True)
    print(f"    Elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"    Speedup: {speedup:.2f}x real-time (RTF: {rtf:.4f})", flush=True)
    print(f"    Cues Generated: {cue_count}", flush=True)

def merge_and_report():
    print(f"\n[*] Merging all chunks and generating profiling report...", flush=True)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    profile_data = {}
    if PROFILE_JSON.exists():
        with open(PROFILE_JSON, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            
    out_final = Path("audio2_profiled.srt").resolve()
    t0_merge = time.perf_counter()
    cue_index = 1
    
    with open(out_final, "w", encoding="utf-8") as out_f:
        for item in manifest:
            idx_name = Path(item["chunk"]).stem
            offset = item["offset"]
            srt_path = Path(f"workspace_chunks/audio2/out_profile/{idx_name}.srt")
            if not srt_path.exists():
                print(f"[!] Warning: missing {srt_path}")
                continue
            with open(srt_path, "r", encoding="utf-8", errors="replace") as sf:
                content = sf.read().strip()
            blocks = content.replace('\r\n', '\n').split('\n\n')
            for b in blocks:
                lines = b.strip().split('\n')
                if len(lines) >= 2 and '-->' in lines[1]:
                    parts = [t.strip() for t in lines[1].split('-->')]
                    st_sec = parse_time_srt(parts[0]) + offset
                    en_sec = parse_time_srt(parts[1]) + offset
                    body = "\n".join(lines[2:]).strip()
                    if body:
                        out_f.write(f"{cue_index}\n{format_time_srt(st_sec)} --> {format_time_srt(en_sec)}\n{body}\n\n")
                        cue_index += 1
                        
    t_merge = time.perf_counter() - t0_merge
    total_cues = cue_index - 1
    
    # Calculate summary metrics
    chunks = profile_data.get("chunks", {})
    total_audio_sec = sum(c["audio_duration_sec"] for c in chunks.values())
    total_infer_sec = sum(c["elapsed_sec"] for c in chunks.values())
    overall_speedup = total_audio_sec / total_infer_sec if total_infer_sec > 0 else 0
    overall_rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else 0
    
    summary = {
        "hardware": "NVIDIA GeForce RTX 4080 (16GB VRAM)",
        "model": "Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M",
        "total_audio_sec": round(total_audio_sec, 2),
        "total_audio_min": round(total_audio_sec / 60, 2),
        "total_inference_sec": round(total_infer_sec, 2),
        "total_inference_min": round(total_infer_sec / 60, 2),
        "overall_speedup": round(overall_speedup, 2),
        "overall_rtf": round(overall_rtf, 4),
        "stitching_time_sec": round(t_merge, 4),
        "total_cues": total_cues,
        "output_srt": str(out_final)
    }
    profile_data["summary"] = summary
    with open(PROFILE_JSON, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
        
    # Copy to original audio location
    mirror = Path(r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\audio\audio2.srt")
    try:
        shutil.copy2(out_final, mirror)
    except Exception:
        pass
        
    print(f"\n=======================================================")
    print(f"BENCHMARK & PROFILING SUMMARY")
    print(f"=======================================================")
    print(f"Total Audio Duration : {total_audio_sec/60:.2f} min ({total_audio_sec:.1f}s)")
    print(f"Total Inference Time : {total_infer_sec/60:.2f} min ({total_infer_sec:.1f}s)")
    print(f"Overall Speedup      : {overall_speedup:.2f}x real-time (RTF: {overall_rtf:.4f})")
    print(f"Subtitle Cues        : {total_cues}")
    print(f"Stitching Overhead   : {t_merge:.4f}s")
    print(f"Final Subtitle File  : {out_final}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    
    if args.chunk is not None:
        run_chunk(args.chunk)
    elif args.merge:
        merge_and_report()
    else:
        print("Usage: --chunk N or --merge")
