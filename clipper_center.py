import os
import subprocess
import numpy as np
from faster_whisper import WhisperModel

VOD_URL = "https://www.youtube.com/watch?v=EXAMPLE_ID"
OUTPUT_DIR = "final_clips"
TEMP_DIR = "temp_chunks"
TARGET_CLIP_DURATION = 45
TOP_SPIKES_TO_EXTRACT = 3
PRE_HOOK_BUFFER = 5
MIN_SPIKE_GAP = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def download_vod(url, output_path):
    print("[+] Downloading VOD via yt-dlp...")
    cmd = [
        "yt-dlp", "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4", "-o", output_path, url
    ]
    subprocess.run(cmd, check=True)

def extract_audio_pcm(video_path, pcm_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-f", "s16le", pcm_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def detect_audio_spikes(pcm_path, top_n=3):
    data = np.fromfile(pcm_path, dtype=np.int16).astype(np.float32)
    sample_rate = 16000
    window_size = sample_rate * 2
    stride = sample_rate // 2
    num_windows = (len(data) - window_size) // stride
    rms_scores = []
    for i in range(num_windows):
        start = i * stride
        window = data[start:start + window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_scores.append((start / sample_rate, rms))
    rms_scores.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for ts, _ in rms_scores:
        if ts < PRE_HOOK_BUFFER or ts > (len(data)/sample_rate - TARGET_CLIP_DURATION): continue
        if all(abs(ts - s) >= MIN_SPIKE_GAP for s in selected):
            selected.append(ts)
            if len(selected) == top_n: break
    return sorted(selected)

def generate_srt(audio_chunk, srt_path):
    model = WhisperModel("small", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio_chunk, word_timestamps=True, language="en")
    words = [w for seg in segments for w in seg.words]
    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for i in range(0, len(words), 3):
            chunk = words[i:i+3]
            st, et = chunk[0].start, chunk[-1].end
            text = " ".join([w.word.strip().upper() for w in chunk])
            f.write(f"{idx}\n00:00:{st:06.3f} --> 00:00:{et:06.3f}\n{text}\n\n".replace(".", ","))
            idx += 1

def render_clip(video_path, start_time, output_path, srt_path):
    vf = f"crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920,subtitles={srt_path.replace(':', '\\:')}:force_style='Fontname=Impact,FontSize=18,PrimaryColour=&H0000FFFF,Outline=2,Alignment=2,MarginV=280'"
    cmd = ["ffmpeg", "-y", "-ss", str(max(0, start_time - PRE_HOOK_BUFFER)), "-t", str(TARGET_CLIP_DURATION), "-i", video_path, "-vf", vf, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22", "-c:a", "aac", output_path]
    subprocess.run(cmd, check=True)

def main():
    vod_path = os.path.join(TEMP_DIR, "source_vod.mp4")
    pcm_path = os.path.join(TEMP_DIR, "source_audio.pcm")
    if not os.path.exists(vod_path): download_vod(VOD_URL, vod_path)
    extract_audio_pcm(vod_path, pcm_path)
    spikes = detect_audio_spikes(pcm_path, top_n=TOP_SPIKES_TO_EXTRACT)
    for idx, ts in enumerate(spikes):
        chunk_audio = os.path.join(TEMP_DIR, f"chunk_{idx}.wav")
        srt_path = os.path.join(TEMP_DIR, f"sub_{idx}.srt")
        out_clip = os.path.join(OUTPUT_DIR, f"center_clip_{idx+1}.mp4")
        subprocess.run(["ffmpeg", "-y", "-ss", str(max(0, ts - PRE_HOOK_BUFFER)), "-t", str(TARGET_CLIP_DURATION), "-i", vod_path, "-vn", "-ac", "1", "-ar", "16000", chunk_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        generate_srt(chunk_audio, srt_path)
        render_clip(vod_path, ts, out_clip, srt_path)
        print(f"[✓] Rendered: {out_clip}")
if __name__ == "__main__": main()
