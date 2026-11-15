import os
import glob
import re

TRANSCRIPT_DIR = "temp_chunks"
OUTPUT_FILE = "post_ready_captions.txt"
CAMPAIGN_TAGS = "#GTA6 #GTA6Gameplay #Whop #Ad #Gaming #ViralShorts"
HOOK_TEMPLATES = ["You won't believe what happened in GTA 6...", "This GTA 6 moment felt completely unscripted 😱", "Is this the craziest detail in GTA 6 yet?"]

def generate_metadata():
    srt_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.srt"))
    if not srt_files: return
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for idx, sf in enumerate(srt_files):
            with open(sf, "r", encoding="utf-8") as f: txt = " ".join([l.strip() for l in f if l.strip() and not l.isdigit() and "-->" not in l])
            words = txt.split()
            snippet = " ".join(words[:6]).capitalize() if len(words)>=6 else "Insane GTA 6 Moment"
            hook = HOOK_TEMPLATES[idx % len(HOOK_TEMPLATES)]
            out.write(f"--- CLIP {idx+1} ---\nTITLE: {snippet} | GTA 6 #shorts\nHOOK: {hook}\nPINNED: Follow for daily GTA 6 clips! 👇\nTAGS: {CAMPAIGN_TAGS}\n\n")
    print(f"[✓] Metadata written to {OUTPUT_FILE}")

if __name__ == "__main__": generate_metadata()
