import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

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

def parse_time_vtt(ts_str):
    parts = ts_str.strip().split(':')
    if len(parts) == 3:
        h, m, s_part = parts
    else:
        h = 0
        m, s_part = parts
    s, ms = s_part.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def format_time_vtt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def get_audio_duration_wave(audio_file):
    import wave
    with wave.open(str(audio_file), 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)

def find_silence_cut_points(ffmpeg_bin, audio_file, total_duration, target_chunk_sec=1200, min_silence_sec=0.5):
    print(f"[*] Analyzing audio silence intervals with FFmpeg...", flush=True)
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

def split_audio(ffmpeg_bin, audio_file, cuts, temp_dir):
    chunk_paths = []
    chunk_offsets = []
    for idx in range(len(cuts) - 1):
        start = cuts[idx]
        duration = cuts[idx + 1] - start
        out_chunk = temp_dir / f"chunk_{idx:03d}.wav"
        chunk_paths.append(out_chunk)
        chunk_offsets.append(start)
        if out_chunk.exists() and out_chunk.stat().st_size > 1000:
            print(f"[*] Chunk {idx:03d} already exists, skipping split: {out_chunk.name}", flush=True)
            continue
        print(f"[*] Splitting chunk {idx:03d}: {start:.1f}s -> {cuts[idx+1]:.1f}s (duration {duration:.1f}s)", flush=True)
        cmd = [
            ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(audio_file),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out_chunk)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return chunk_paths, chunk_offsets

def merge_srt(chunk_outputs, chunk_offsets, final_path):
    cue_index = 1
    with open(final_path, 'w', encoding='utf-8') as out_f:
        for chunk_file, offset in zip(chunk_outputs, chunk_offsets):
            if not chunk_file.exists():
                print(f"[!] Warning: chunk file missing: {chunk_file}", flush=True)
                continue
            with open(chunk_file, 'r', encoding='utf-8', errors="replace") as f:
                content = f.read().strip()
            if not content:
                continue
            blocks = content.replace('\r\n', '\n').split('\n\n')
            for b in blocks:
                lines = b.strip().split('\n')
                if len(lines) >= 2 and '-->' in lines[1]:
                    time_line = lines[1]
                    start_str, end_str = [t.strip() for t in time_line.split('-->')]
                    start_sec = parse_time_srt(start_str) + offset
                    end_sec = parse_time_srt(end_str) + offset
                    text = "\n".join(lines[2:])
                    out_f.write(f"{cue_index}\n{format_time_srt(start_sec)} --> {format_time_srt(end_sec)}\n{text}\n\n")
                    cue_index += 1

def merge_vtt(chunk_outputs, chunk_offsets, final_path):
    with open(final_path, 'w', encoding='utf-8') as out_f:
        out_f.write("WEBVTT\n\n")
        for chunk_file, offset in zip(chunk_outputs, chunk_offsets):
            if not chunk_file.exists():
                continue
            with open(chunk_file, 'r', encoding='utf-8', errors="replace") as f:
                content = f.read().strip()
            blocks = content.replace('\r\n', '\n').split('\n\n')
            for b in blocks:
                lines = b.strip().split('\n')
                if not lines or lines[0].startswith("WEBVTT"):
                    continue
                time_idx = 0
                if '-->' not in lines[0] and len(lines) > 1 and '-->' in lines[1]:
                    time_idx = 1
                if '-->' in lines[time_idx]:
                    start_str, end_str = [t.strip().split()[0] for t in lines[time_idx].split('-->')]
                    start_sec = parse_time_vtt(start_str) + offset
                    end_sec = parse_time_vtt(end_str) + offset
                    text = "\n".join(lines[time_idx + 1:])
                    out_f.write(f"{format_time_vtt(start_sec)} --> {format_time_vtt(end_sec)}\n{text}\n\n")

def merge_json(chunk_outputs, chunk_offsets, final_path):
    merged = {
        "text": "",
        "duration": 0.0,
        "words": []
    }
    texts = []
    for chunk_file, offset in zip(chunk_outputs, chunk_offsets):
        if not chunk_file.exists():
            continue
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts.append(data.get("text", ""))
        merged["duration"] += data.get("duration", 0.0)
        for w in data.get("words", []):
            item = dict(w)
            if "start" in item:
                item["start"] = round(item["start"] + offset, 3)
            if "end" in item:
                item["end"] = round(item["end"] + offset, 3)
            merged["words"].append(item)
    merged["text"] = " ".join(texts).strip()
    with open(final_path, 'w', encoding='utf-8') as out_f:
        json.dump(merged, out_f, indent=2, ensure_ascii=False)

def merge_text(chunk_outputs, final_path):
    texts = []
    for chunk_file in chunk_outputs:
        if not chunk_file.exists():
            continue
        with open(chunk_file, 'r', encoding='utf-8', errors="replace") as f:
            texts.append(f.read().strip())
    with open(final_path, 'w', encoding='utf-8') as out_f:
        out_f.write("\n\n".join(texts))

def main():
    parser = argparse.ArgumentParser(description="Transcribe long audio using NeMo-speech with smart VAD chunking.")
    parser.add_argument("input_audio", help="Path to long audio file")
    parser.add_argument("--output", "-o", default=None, help="Path to final output file")
    parser.add_argument("--format", "-f", default="srt", choices=["srt", "vtt", "json", "text"], help="Output format")
    parser.add_argument("--nemo-bin", default=r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\bin\nemo-speech.exe")
    parser.add_argument("--model", default=r"C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\models\NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf")
    parser.add_argument("--chunk-target-min", type=float, default=20.0, help="Target chunk duration in minutes (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Only split audio, skip transcription")
    args = parser.parse_args()

    audio_path = Path(args.input_audio).resolve()
    ffmpeg_bin = find_ffmpeg()
    print(f"[*] Using FFmpeg binary: {ffmpeg_bin}", flush=True)

    if not args.output:
        ext = args.format if args.format != "text" else "txt"
        args.output = str(Path.cwd() / f"{audio_path.stem}.{ext}")

    work_dir = Path(__file__).parent / "workspace_chunks" / audio_path.stem
    chunks_dir = work_dir / "wavs"
    out_dir = work_dir / "out"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_duration = get_audio_duration_wave(audio_path)
    print(f"[*] Total duration: {total_duration:.1f}s ({total_duration / 60:.2f} min)", flush=True)

    target_sec = args.chunk_target_min * 60.0
    cuts = find_silence_cut_points(ffmpeg_bin, audio_path, total_duration, target_chunk_sec=target_sec)
    print(f"[*] Identified {len(cuts) - 1} silence-aligned cuts: {[round(c, 1) for c in cuts]}", flush=True)

    chunk_paths, chunk_offsets = split_audio(ffmpeg_bin, audio_path, cuts, chunks_dir)

    manifest_path = work_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump([{"chunk": p.name, "offset": off} for p, off in zip(chunk_paths, chunk_offsets)], f, indent=2)

    if args.dry_run:
        print("[*] Dry run enabled. Chunks prepared. Exiting.", flush=True)
        return

    ext = args.format if args.format != "text" else "txt"
    chunk_outputs = [out_dir / f"{p.stem}.{ext}" for p in chunk_paths]

    print(f"\n[*] Launching NeMo-speech for {len(chunk_paths)} chunks...", flush=True)
    nemo_cmd = [
        args.nemo_bin, "transcribe", str(chunks_dir),
        "--model", args.model,
        "--device", "cuda",
        "--format", args.format,
        "--output-dir", str(out_dir),
        "--concurrency", "1",
        "--instruction", "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization, numbers in their standard written form. Output: transcript only.",
        "--asr.salm.max_sequences", "1",
        "--asr.salm.n_ctx_per_sequence", "28192",
        "--asr.salm.max_context", "28192",
        "--asr.salm.mtp_draft", "0",
        "--asr.salm.n_batch", "512",
        "--asr.salm.encoder_max_batch", "2"
    ]
    if args.format == "json":
        nemo_cmd.append("--word-times")

    print("[*] Executing NeMo-speech batch...", flush=True)
    proc = subprocess.run(nemo_cmd)
    if proc.returncode != 0:
        print(f"[!] nemo-speech exited with error code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)

    print(f"\n[*] Merging {len(chunk_outputs)} outputs into {args.output}...", flush=True)
    final_output_path = Path(args.output).resolve()
    if args.format == "srt":
        merge_srt(chunk_outputs, chunk_offsets, final_output_path)
    elif args.format == "vtt":
        merge_vtt(chunk_outputs, chunk_offsets, final_path=final_output_path)
    elif args.format == "json":
        merge_json(chunk_outputs, chunk_offsets, final_path=final_output_path)
    elif args.format == "text":
        merge_text(chunk_outputs, final_output_path)

    # Also copy to original audio directory if different
    orig_dir_output = audio_path.parent / f"{audio_path.stem}.{ext}"
    if orig_dir_output.resolve() != final_output_path:
        try:
            shutil.copy2(final_output_path, orig_dir_output)
            print(f"[+] Also mirrored output to: {orig_dir_output}", flush=True)
        except Exception as e:
            print(f"[!] Note: Could not copy to original dir: {e}", flush=True)

    print(f"[SUCCESS] Transcription finished successfully: {final_output_path}", flush=True)

if __name__ == "__main__":
    main()
