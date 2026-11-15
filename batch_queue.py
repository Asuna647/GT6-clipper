import os
import csv
import subprocess
from datetime import datetime

URLS_FILE = "urls.txt"
MANIFEST_FILE = "clip_manifest.csv"
DEFAULT_LANGUAGE = "english"
SHUTDOWN_ON_COMPLETE = False

def initialize_manifest():
    if not os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Source_URL", "Language", "Status"])

def process_batch():
    initialize_manifest()
    if not os.path.exists(URLS_FILE):
        with open(URLS_FILE, "w") as f: f.write("# Format: URL or language|URL (e.g. hindi|https://...)\n")
        return
    with open(URLS_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    for idx, line in enumerate(lines):
        if "|" in line: lang, url = line.split("|", 1)
        else: lang, url = DEFAULT_LANGUAGE, line
        script = "clipper_english_ultimate.py" if lang.lower() == "english" else "clipper_hindi_ultimate.py"
        print(f"\n[+] Processing VOD {idx+1}/{len(lines)} ({lang}): {url}")
        try:
            subprocess.run(["python", script, "--url", url], check=True)
            with open(MANIFEST_FILE, "a", newline="", encoding="utf-8") as mf:
                csv.writer(mf).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), url, lang, "SUCCESS"])
        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            if os.path.exists("temp_chunks"):
                for fn in os.listdir("temp_chunks"):
                    try: os.remove(os.path.join("temp_chunks", fn))
                    except: pass
    if SHUTDOWN_ON_COMPLETE: os.system("shutdown /s /t 60")

if __name__ == "__main__": process_batch()
