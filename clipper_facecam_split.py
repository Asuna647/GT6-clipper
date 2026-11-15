import os
import subprocess

INPUT_VOD = "temp_chunks/source_vod.mp4"
OUTPUT_CLIP = "final_clips/split_facecam_clip.mp4"
START_TIME = 120
DURATION = 40
FACECAM_CORNER = "bottom_right"

def get_facecam_crop_filter(corner):
    cam_coords = {"bottom_right": (1440, 720), "bottom_left": (0, 720), "top_right": (1440, 0), "top_left": (0, 0)}
    cam_x, cam_y = cam_coords.get(corner, (1440, 720))
    return f"[0:v]crop=480:360:{cam_x}:{cam_y},scale=1080:640[top];[0:v]crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1280[bottom];[top][bottom]vstack=inputs=2[v]"

def render_split_screen(input_path, output_path, start, duration, corner):
    print(f"[+] Rendering Split-Screen ({corner})...")
    filter_complex = get_facecam_crop_filter(corner)
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", input_path, "-filter_complex", filter_complex, "-map", "[v]", "-map", "0:a", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-c:a", "aac", output_path]
    subprocess.run(cmd, check=True)
    print(f"[✓] Split-Screen clip saved: {output_path}")

if __name__ == "__main__":
    if os.path.exists(INPUT_VOD): render_split_screen(INPUT_VOD, OUTPUT_CLIP, START_TIME, DURATION, FACECAM_CORNER)
