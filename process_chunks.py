import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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

def transcribe_one_chunk(idx, nemo_bin, model_path):
    wav_path = Path(f"workspace_chunks/audio2/wavs/chunk_{idx:03d}.wav").resolve()
    out_path = Path(f"workspace_chunks/audio2/out/chunk_{idx:03d}.srt").resolve()
    
    if not wav_path.exists():
        print(f"Error: {wav_path} does not exist.")
        sys.exit(1)
        
    if out_path.exists() and out_path.stat().st_size > 50:
        print(f"Chunk {idx:03d} already transcribed: {out_path} ({out_path.stat().st_size} bytes)")
        return
        
    cmd = [
        nemo_bin, "transcribe", str(wav_path),
        "--model", model_path,
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
    print(f"[*] Starting transcription of chunk {idx:03d}...", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error in chunk {idx:03d}: exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[+] Chunk {idx:03d} complete: {out_path}", flush=True)

def merge_all():
    manifest_path = Path("workspace_chunks/audio2/manifest.json")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    cue_index = 1
    out_final = Path("audio2.srt").resolve()
    
    with open(out_final, 'w', encoding='utf-8') as out_f:
        for item in manifest:
            chunk_name = Path(item["chunk"]).stem
            offset = item["offset"]
            srt_path = Path(f"workspace_chunks/audio2/out/{chunk_name}.srt")
            if not srt_path.exists():
                print(f"[!] Warning: Missing {srt_path}")
                continue
            with open(srt_path, 'r', encoding='utf-8', errors='replace') as sf:
                content = sf.read().strip()
            if not content:
                continue
            blocks = content.replace('\r\n', '\n').split('\n\n')
            for b in blocks:
                lines = b.strip().split('\n')
                if len(lines) >= 2 and '-->' in lines[1]:
                    time_line = lines[1]
                    parts = [t.strip() for t in time_line.split('-->')]
                    start_sec = parse_time_srt(parts[0]) + offset
                    end_sec = parse_time_srt(parts[1]) + offset
                    text = "\n".join(lines[2:]).strip()
                    if text:
                        out_f.write(f"{cue_index}\n{format_time_srt(start_sec)} --> {format_time_srt(end_sec)}\n{text}\n\n")
                        cue_index += 1
                        
    print(f"[SUCCESS] Merged all chunks into {out_final} ({cue_index - 1} cues)")
    
    # Mirror copy to original audio directory
    mirror_path = Path(r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\audio\audio2.srt")
    try:
        shutil.copy2(out_final, mirror_path)
        print(f"[+] Copied to: {mirror_path}")
    except Exception as e:
        print(f"[!] Note on mirror copy: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    
    nemo_bin = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\bin\nemo-speech.exe"
    model_path = r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\models\NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf"
    
    if args.chunk is not None:
        transcribe_one_chunk(args.chunk, nemo_bin, model_path)
    elif args.merge:
        merge_all()
    else:
        print("Usage: --chunk N or --merge")
