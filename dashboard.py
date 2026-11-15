import os
import re
import glob
import subprocess
import cv2
import numpy as np
import streamlit as st
import pandas as pd

st.set_page_config(page_title="GTA 6 AI Clipper Studio", page_icon="🎮", layout="wide")

FINAL_DIR = "final_clips"
TEMP_DIR = "temp_chunks"
URLS_FILE = "urls.txt"
MANIFEST_FILE = "clip_manifest.csv"
CAPTIONS_FILE = "post_ready_captions.txt"
PREVIEW_FRAME = os.path.join(TEMP_DIR, "preview_frame.jpg")
SOURCE_VOD = os.path.join(TEMP_DIR, "source_vod.mp4")

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

if "cam_x" not in st.session_state:
    st.session_state.update({"cam_x": 1440, "cam_y": 720, "cam_w": 480, "cam_h": 360})

st.title("🎮 GTA 6 Local RTX 3060 Auto-Clipper Studio")
st.caption("Zero-API Cost • Live Facecam Calibrator • 800Hz–3000Hz Vocal Bandpass • NVENC 60FPS")

with st.sidebar:
    st.header("⚡ System & Cache Control")
    temp_files = glob.glob(os.path.join(TEMP_DIR, "*"))
    temp_size_mb = sum(os.path.getsize(f) for f in temp_files if os.path.isfile(f)) / (1024 * 1024)
    st.metric("Temp Cache Size", f"{temp_size_mb:.1f} MB", f"{len(temp_files)} files")

    if st.button("🗑️ Purge Temp Cache (Free SSD)", use_container_width=True):
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        st.success("Temp cache wiped!")
        st.rerun()

    st.divider()
    st.subheader("🎯 Active Facecam Crop")
    st.code(f"crop={st.session_state.cam_w}:{st.session_state.cam_h}:{st.session_state.cam_x}:{st.session_state.cam_y}")

    st.divider()
    st.subheader("📋 Quick Compliance Checklist")
    st.checkbox("Campaign budget bar < 90% full")
    st.checkbox("Added #ad / Paid Partnership toggle")
    st.checkbox("Submitted URL within 30 mins")

tab_cal, tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Facecam Crop Calibrator",
    "🚀 Single VOD Clipper",
    "📦 Batch Queue (urls.txt)",
    "🎬 Rendered Clips & Captions",
    "📊 Run History"
])

with tab_cal:
    st.subheader("🎯 Visual Facecam Crop Calibrator & Live 9:16 Preview")
    c_src1, c_src2, c_src3 = st.columns([3, 1, 1])
    with c_src1:
        cal_url = st.text_input("Stream/Video URL (leave blank to use cached temp_chunks/source_vod.mp4):", key="cal_url")
    with c_src2:
        sample_sec = st.number_input("Timestamp (sec):", min_value=1, max_value=36000, value=120, step=10)
    with c_src3:
        st.write("")
        st.write("")
        grab_btn = st.button("📸 Grab Frame", type="primary", use_container_width=True)

    if grab_btn:
        with st.spinner("Extracting 1080p frame..."):
            target_input = SOURCE_VOD if (not cal_url.strip() and os.path.exists(SOURCE_VOD)) else None
            if cal_url.strip():
                res = subprocess.run(
                    ["yt-dlp", "-f", "bestvideo[height<=1080]/best", "-g", cal_url.strip()],
                    capture_output=True, text=True
                )
                stream_url = res.stdout.strip().splitlines()[0] if res.stdout.strip() else None
                target_input = stream_url

            if not target_input:
                st.error("No cached source_vod.mp4 found and no valid URL provided.")
            else:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(sample_sec), "-i", target_input,
                    "-frames:v", "1", "-vf", "scale=1920:1080", "-q:v", "2", PREVIEW_FRAME
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(PREVIEW_FRAME):
                    st.success(f"Frame captured at {sample_sec}s!")
                else:
                    st.error("Failed to extract frame. Try a smaller timestamp or check the URL.")

    p1, p2, p3, p4 = st.columns(4)
    if p1.button("↘️ Bottom Right (Default)", use_container_width=True):
        st.session_state.update({"cam_x": 1440, "cam_y": 720, "cam_w": 480, "cam_h": 360})
    if p2.button("↙️ Bottom Left", use_container_width=True):
        st.session_state.update({"cam_x": 0, "cam_y": 720, "cam_w": 480, "cam_h": 360})
    if p3.button("↗️ Top Right", use_container_width=True):
        st.session_state.update({"cam_x": 1440, "cam_y": 0, "cam_w": 480, "cam_h": 360})
    if p4.button("↖️ Top Left", use_container_width=True):
        st.session_state.update({"cam_x": 0, "cam_y": 0, "cam_w": 480, "cam_h": 360})

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        cam_w = st.slider("Webcam Width (W)", min_value=160, max_value=960, value=st.session_state.cam_w, step=4)
    with s2:
        cam_h = st.slider("Webcam Height (H)", min_value=120, max_value=720, value=st.session_state.cam_h, step=4)
    with s3:
        max_x = max(0, 1920 - cam_w)
        cam_x = st.slider("X Offset (Left → Right)", min_value=0, max_value=max_x, value=min(st.session_state.cam_x, max_x), step=4)
    with s4:
        max_y = max(0, 1080 - cam_h)
        cam_y = st.slider("Y Offset (Top → Bottom)", min_value=0, max_value=max_y, value=min(st.session_state.cam_y, max_y), step=4)

    st.session_state.update({"cam_x": cam_x, "cam_y": cam_y, "cam_w": cam_w, "cam_h": cam_h})

    if os.path.exists(PREVIEW_FRAME):
        img_bgr = cv2.imread(PREVIEW_FRAME)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            annotated = img_rgb.copy()
            cv2.rectangle(annotated, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), (0, 255, 0), 4)
            cv2.rectangle(annotated, (656, 0), (1264, 1080), (255, 255, 0), 2)

            face_crop = img_rgb[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
            face_resized = cv2.resize(face_crop, (540, 320))
            game_crop = img_rgb[0:1080, 656:1264]
            game_resized = cv2.resize(game_crop, (540, 640))
            stacked_preview = np.vstack([face_resized, game_resized])
            cv2.line(stacked_preview, (0, 320), (540, 320), (0, 255, 255), 3)

            prev_col1, prev_col2 = st.columns([2, 1])
            with prev_col1:
                st.image(annotated, caption="1920x1080 Source (Green = Facecam | Yellow = Gameplay)", use_container_width=True)
            with prev_col2:
                st.image(stacked_preview, caption="Simulated 9:16 Split-Screen Output", width=300)
    else:
        st.info("Click 📸 Grab Frame above to load a live video frame for visual calibration.")

    new_crop_str = f"crop={cam_w}:{cam_h}:{cam_x}:{cam_y}"
    if st.button(f"💾 Apply {new_crop_str} to All Clipper Scripts", type="primary", use_container_width=True):
        patched = []
        for script_name in ["clipper_english_ultimate.py", "clipper_hindi_ultimate.py", "clipper_facecam_split.py"]:
            if os.path.exists(script_name):
                with open(script_name, "r", encoding="utf-8") as f:
                    code = f.read()
                updated = re.sub(r"crop=\d+:\d+:[^,]+,scale=1080:640", f"{new_crop_str},scale=1080:640", code)
                with open(script_name, "w", encoding="utf-8") as f:
                    f.write(updated)
                patched.append(script_name)
        st.success(f"Patched {new_crop_str} into: {', '.join(patched)}!")

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        vod_url = st.text_input("Paste YouTube or Twitch VOD URL:", placeholder="https://www.youtube.com/watch?v=...")
    with col2:
        engine = st.selectbox("Select Streamer Pipeline:", [
            "English Ultimate (66 Chat Tokens)",
            "Hindi / Hinglish Ultimate (50 Tokens)",
            "Fast 9:16 Center Crop (No Facecam)"
        ])

    clear_before_run = st.checkbox("Auto-clean temp_chunks before downloading new VOD", value=True)

    if st.button("🔥 Start GPU Auto-Clipping", type="primary", use_container_width=True):
        if not vod_url.strip():
            st.warning("Please paste a valid YouTube or Twitch URL first.")
        else:
            if clear_before_run:
                for f in glob.glob(os.path.join(TEMP_DIR, "*")):
                    if f != PREVIEW_FRAME:
                        try:
                            os.remove(f)
                        except Exception:
                            pass

            if "English" in engine:
                cmd = ["python", "clipper_english_ultimate.py", "--url", vod_url.strip()]
            elif "Hindi" in engine:
                cmd = ["python", "clipper_hindi_ultimate.py", "--url", vod_url.strip()]
            else:
                cmd = ["python", "clipper_center.py"]

            st.info(f"Running: {' '.join(cmd)} on RTX 3060...")
            log_box = st.empty()
            full_log = ""

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in proc.stdout:
                full_log += line
                log_box.code(full_log[-2500:], language="bash")
            proc.wait()

            if proc.returncode == 0:
                subprocess.run(["python", "meta_generator.py"], check=False)
                st.success("Done! Switch to the Rendered Clips & Captions tab to preview and download.")
            else:
                st.error(f"Process exited with code {proc.returncode}. Check the log above.")

with tab2:
    st.subheader("Queue Multiple VODs (urls.txt)")
    existing_urls = ""
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, "r", encoding="utf-8") as f:
            existing_urls = f.read()

    new_urls = st.text_area("Edit Queue:", value=existing_urls, height=200)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save Queue to urls.txt", use_container_width=True):
            with open(URLS_FILE, "w", encoding="utf-8") as f:
                f.write(new_urls)
            st.success("Saved urls.txt!")
    with c2:
        if st.button("▶️ Run Full Batch Queue Now", type="primary", use_container_width=True):
            with open(URLS_FILE, "w", encoding="utf-8") as f:
                f.write(new_urls)
            st.info("Executing batch_queue.py...")
            res = subprocess.run(["python", "batch_queue.py"], capture_output=True, text=True)
            st.code(res.stdout + "\n" + res.stderr)
            subprocess.run(["python", "meta_generator.py"], check=False)
            st.success("Batch queue complete!")

with tab3:
    col_vid, col_meta = st.columns([2, 1])
    with col_vid:
        st.subheader("🎬 Rendered Vertical Clips (final_clips/)")
        mp4_files = sorted(glob.glob(os.path.join(FINAL_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
        if not mp4_files:
            st.info("No clips rendered yet. Run a VOD in Tab 1!")
        else:
            cols = st.columns(3)
            for idx, clip_path in enumerate(mp4_files):
                with cols[idx % 3]:
                    st.video(clip_path)
                    st.caption(os.path.basename(clip_path))
                    with open(clip_path, "rb") as file_bytes:
                        st.download_button(
                            label="⬇️ Download MP4",
                            data=file_bytes,
                            file_name=os.path.basename(clip_path),
                            mime="video/mp4",
                            key=f"dl_{idx}",
                            use_container_width=True
                        )

    with col_meta:
        st.subheader("📝 Ready-to-Post Captions & #Ad Tags")
        if st.button("🔄 Regenerate Captions from Transcripts", use_container_width=True):
            subprocess.run(["python", "meta_generator.py"], check=False)
        if os.path.exists(CAPTIONS_FILE):
            with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
                st.code(f.read(), language="markdown")
        else:
            st.caption("No captions generated yet.")

with tab4:
    st.subheader("📊 Batch Execution Log (clip_manifest.csv)")
    if os.path.exists(MANIFEST_FILE):
        df = pd.read_csv(MANIFEST_FILE)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No batch history logged yet.")