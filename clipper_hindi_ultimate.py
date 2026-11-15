import os
import re
import argparse
import subprocess
import numpy as np
from faster_whisper import WhisperModel
from chat_downloader import ChatDownloader

OUTPUT_DIR = "final_clips"
TEMP_DIR = "temp_chunks"

HINDI_LEXICON_WORD_BOUNDARY = {
    "BHAI": 3.0, "BHAII": 3.0, "BHAI BHAI": 3.5, "ARRE": 2.5, "AREY": 2.5,
    "KYA": 2.0, "KYAAA": 2.5, "KYA SCENE": 3.0, "KYA KAR RAHA": 3.0,
    "NAHI": 2.0, "NAHI YAAR": 2.5, "OH BHAI": 3.5, "OYE": 2.0, "DEKH": 2.0, "RUK": 2.0, "RUKO": 2.0,
    "PAGAL": 2.5, "CHOMU": 2.5, "NOOB": 2.5, "PRO": 2.5, "OP": 3.5, "OP BHAI": 4.0,
    "GOAT": 3.0, "LEGEND": 3.0, "KING": 3.0, "MAST": 2.5, "ZABARDAST": 3.0, "MAZA AA GAYA": 3.5,
    "KYA BAAT HAI": 3.0, "JALA DIYA": 3.0, "PEL DIYA": 3.5, "MAAR DIYA": 3.0, "BACH GAYA": 3.0,
    "GAYA": 2.0, "LAG GAYE": 3.5, "RIP": 2.5, "LOL": 2.0, "LMAO": 2.5, "HAHAHA": 2.5, "OMG": 2.5, "GG": 2.0
}
HINDI_SYMBOL_TRIGGERS = {"BH**N": 3.5, "😂": 3.0, "🤣": 3.0, "😭": 3.0, "💀": 3.5, "🔥": 3.5}
REGEX_SINGLE_LETTER_WAVE = re.compile(r'\b[WLF]\b', re.IGNORECASE)

KEYWORD_YELLOW_HI = ["BHAI", "OP", "GTA6", "LEGEND", "MAST", "KING", "GOAT", "ZABARDAST", "KYA"]
KEYWORD_RED_HI = ["LAG GAYE", "GAYA", "PEL DIYA", "RIP", "PAGAL", "NOOB", "FAIL", "WASTED"]
KEYWORD_GREEN_HI = ["BACH GAYA", "PRO", "WIN", "SUCCESS", "MAZA", "FREE", "BEST"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def download_vod(url, out_path):
    subprocess.run(["yt-dlp", "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "--merge-output-format", "mp4", "-o", out_path, url], check=True)

def normalize_hindi(t):
    t = re.sub(r'I{2,}', 'I', t.upper())
    t = re.sub(r'A{2,}', 'A', t)
    t = re.sub(r'E{2,}', 'E', t)
    return re.sub(r'Y{2,}', 'Y', t)

def extract_vocal_pcm(v_path, p_path):
    subprocess.run(["ffmpeg", "-y", "-i", v_path, "-af", "highpass=f=800,lowpass=f=3000", "-ac", "1", "-ar", "16000", "-f", "s16le", p_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def fetch_hindi_chat_velocity(url, duration_sec):
    try:
        chat = ChatDownloader().get_chat(url)
        events = []
        for m in chat:
            if 'time_in_seconds' in m and m['time_in_seconds'] is not None:
                raw = m.get('message', '').upper()
                norm = normalize_hindi(raw)
                score = 1.0
                for tok, w in HINDI_LEXICON_WORD_BOUNDARY.items():
                    if re.search(r'\b' + re.escape(tok) + r'\b', norm): score += w
                for sym, w in HINDI_SYMBOL_TRIGGERS.items():
                    if sym in raw: score += w
                score += len(REGEX_SINGLE_LETTER_WAVE.findall(raw)) * 2.5
                events.append((max(0, m['time_in_seconds'] - 5.0), max(0.0, score)))
        if len(events) < 15: return None
        bins = np.arange(0, duration_sec + 5, 5.0)
        counts = np.zeros(len(bins) - 1)
        for t, w in events:
            b = int(t // 5.0)
            if b < len(counts): counts[b] += w
        vel = {}
        for i, c in enumerate(counts):
            base = np.mean(counts[max(0, i-60):i+1]) if i > 0 else c
            vel[bins[i]] = c / (base + 0.1)
        return vel
    except Exception as e:
        print(f"[!] Hindi Chat warning: {e}")
        return None

def find_top_3_peaks_hindi(pcm_path, vod_path, url):
    data = np.fromfile(pcm_path, dtype=np.int16).astype(np.float32)
    dur = len(data) / 16000
    rms_candidates = []
    for start in range(0, len(data) - 32000, 8000):
        rms = np.sqrt(np.mean(data[start:start+32000]**2))
        rms_candidates.append((start / 16000, rms))
    rms_candidates.sort(key=lambda x: x[1], reverse=True)
    vel = fetch_hindi_chat_velocity(url, dur)
    scored = []
    for ts, rms in rms_candidates:
        if ts < 30 or ts > (dur - 45): continue
        mult = vel.get(round(ts / 5.0) * 5.0, 1.0) if vel else 1.0
        scored.append((ts, rms * (1.0 + np.log1p(mult))))
    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = []
    for ts, _ in scored:
        if all(abs(ts - existing) >= 60 for existing in top3):
            top3.append(ts)
            if len(top3) == 3: break
    return sorted(top3)

def generate_ass_hindi(audio_chunk, ass_path):
    model = WhisperModel("small", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio_chunk, word_timestamps=True)
    words = [{"start": w.start, "end": w.end, "word": w.word.strip().upper()} for seg in segments for w in seg.words]
    ass_h = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Impact,22,&H00FFFFFF,&H00000000,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,1,2,10,10,640,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    def ft(s): return f"{int(s//3600)}:{int((s%3600)//60):02d}:{int(s%60):02d}.{int((s%1)*100):02d}"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_h)
        for i in range(0, len(words), 2):
            c = words[i:i+2]
            line = " ".join([(r"{\c&H00FFFF&}" if w["word"] in KEYWORD_YELLOW_HI else r"{\c&H0000FF&}" if w["word"] in KEYWORD_RED_HI else r"{\c&H00FF00&}" if w["word"] in KEYWORD_GREEN_HI else r"{\c&H00FFFFFF&}") + w["word"] for w in c])
            f.write(f"Dialogue: 0,{ft(c[0]['start'])},{ft(c[-1]['end'])},Default,,0,0,0,,{{\\pos(540,640)\\fscx118\\fscy118\\t(0,80,\\fscx100\\fscy100)}}{line}\n")

def render_master_hindi(vod_path, peak, out_path, idx):
    ass_p = os.path.join(TEMP_DIR, f"captions_hi_{idx}.ass")
    aud_p = os.path.join(TEMP_DIR, f"audio_hi_{idx}.wav")
    m_st, m_dur = max(0, peak - 25), 26.5
    subprocess.run(["ffmpeg", "-y", "-ss", str(m_st), "-t", str(m_dur), "-i", vod_path, "-vn", "-ac", "1", "-ar", "16000", aud_p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    generate_ass_hindi(aud_p, ass_p)
    e_ass = ass_p.replace("\\", "/").replace(":", "\\:")
    fc = f"[0:v]crop=480:360:1440:720,scale=1080:640[cam];[0:v]crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1280,zoompan=z='if(eq(mod(on,105),0),1.08,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1280[game];[cam][game]vstack=inputs=2[stacked];[stacked]drawbox=x=0:y=0:w=1080:h=120:color=black@0.7:t=fill,drawtext=text='BHAI YE KYA HUA GTA 6 MEIN':fontfile=C\\\\:/Windows/Fonts/impact.ttf:fontsize=44:fontcolor=white:x=(w-text_w)/2:y=35:enable='between(t,0,3)'[banner];[banner]subtitles='{e_ass}'[vfinal];[0:a]dynaudnorm=f=150:g=15,acompressor=threshold=-18dB:ratio=4:attack=5:release=50[afinal]"
    hk_p, bd_p = os.path.join(TEMP_DIR, f"hk_hi_{idx}.mp4"), os.path.join(TEMP_DIR, f"bd_hi_{idx}.mp4")
    subprocess.run(["ffmpeg", "-y", "-ss", str(max(0, peak - 1.4)), "-t", "1.4", "-i", vod_path, "-vf", "crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-c:a", "aac", hk_p], check=True)
    subprocess.run(["ffmpeg", "-y", "-ss", str(m_st), "-t", str(m_dur), "-i", vod_path, "-filter_complex", fc, "-map", "[vfinal]", "-map", "[afinal]", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-c:a", "aac", bd_p], check=True)
    c_txt = os.path.join(TEMP_DIR, f"concat_hi_{idx}.txt")
    with open(c_txt, "w") as f: f.write(f"file 'hk_hi_{idx}.mp4'\nfile 'bd_hi_{idx}.mp4'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", c_txt, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path], check=True)
    print(f"[✓] Rendered Top Hindi Clip {idx}: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    v_path = os.path.join(TEMP_DIR, "source_vod.mp4")
    p_path = os.path.join(TEMP_DIR, "source_audio.pcm")
    if not os.path.exists(v_path): download_vod(args.url, v_path)
    extract_vocal_pcm(v_path, p_path)
    peaks = find_top_3_peaks_hindi(p_path, v_path, args.url)
    for i, peak in enumerate(peaks):
        render_master_hindi(v_path, peak, os.path.join(OUTPUT_DIR, f"hindi_clip_{i+1}.mp4"), i+1)

if __name__ == "__main__": main()
