import os
import cv2
import torch
import subprocess
import gradio as gr
import tempfile
import requests
from PIL import Image
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

FONT_URL = "https.github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf/DejaVuSans-Bold.ttf"
FONT_PATH = "DejaVuSans-Bold.ttf"
if not os.path.exists(FONT_PATH):
    try:
        r = requests.get(FONT_URL)
        r.raise_for_status()
        with open(FONT_PATH, "wb") as f:
            f.write(r.content)
    except:
        linux_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        FONT_PATH = linux_font_path if os.path.exists(linux_font_path) else "sans-serif"

def get_video_duration(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout)
    except:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        if duration == 0:
            raise ValueError("Could not determine video duration.")
        return duration

def run_ffmpeg_command(cmd):
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def run_video_pipeline(video_input_path, progress=gr.Progress()):
    if video_input_path is None:
        raise gr.Error("Please upload a video file.")

    temp_dir = tempfile.mkdtemp()
    VIDEO_PATH = video_input_path
    OUT_DIR = temp_dir
    TTS_DIR = os.path.join(OUT_DIR, "tts_audio")
    os.makedirs(TTS_DIR, exist_ok=True)

    try:
        progress(0.05, desc="Detecting scenes...")
        video = open_video(VIDEO_PATH)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=10.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        if not scene_list:
            raise gr.Error("No scenes detected.")

        progress(0.15, desc="Extracting keyframes...")
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        scene_frames = []
        for i, (start, end) in enumerate(scene_list):
            mid_time = (start.get_seconds() + end.get_seconds()) / 2
            frame_num = int(mid_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret: continue
            frame_path = os.path.join(OUT_DIR, f"scene_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            scene_frames.append((frame_path, mid_time))
        cap.release()
        if not scene_frames:
            raise gr.Error("Failed to extract frames.")

        progress(0.25, desc="Generating captions...")
        captions = []
        total_frames = len(scene_frames)
        for i, (frame_path, ts) in enumerate(scene_frames):
            progress(0.25 + (i / total_frames) * 0.3)
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append((ts, caption))

        progress(0.55, desc="Cleaning captions...")
        clean_captions = []
        prev_text = None
        last_time = -0.5
        for ts, caption in captions:
            text = caption.strip().lower()
            if text and text != prev_text and (ts - last_time) > 0.5 and len(text) > 5:
                clean_captions.append((ts, caption))
                last_time = ts
            prev_text = text
        if not clean_captions:
            raise gr.Error("No captions left after cleaning.")

        progress(0.65, desc="Generating TTS audio...")
        tts_files = []
        total_captions = len(clean_captions)
        for i, (ts, caption) in enumerate(clean_captions):
            progress(0.65 + (i / total_captions) * 0.1)
            tts_path = os.path.join(TTS_DIR, f"tts_{i:03d}.mp3")
            gTTS(caption, lang='en').save(tts_path)
            duration_sec = len(AudioSegment.from_file(tts_path)) / 1000.0
            tts_files.append((ts, tts_path, duration_sec))
        if not tts_files:
            raise gr.Error("Failed to generate TTS.")

        progress(0.75, desc="Merging audio track...")
        video_duration = get_video_duration(VIDEO_PATH)
        final_audio = AudioSegment.silent(duration=video_duration * 1000)
        for ts, audio_path, _ in tts_files:
            tts_audio = AudioSegment.from_file(audio_path)
            final_audio = final_audio.overlay(tts_audio, position=int(ts*1000))
        final_tts_path = os.path.join(OUT_DIR, "final_tts.mp3")
        final_audio.export(final_tts_path, format="mp3")

        progress(0.85, desc="Merging video with audio...")
        final_video_path = os.path.join(OUT_DIR, "final_video.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", VIDEO_PATH,
            "-i", final_tts_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v",
            "-map", "1:a",
            "-shortest",
            final_video_path
        ]
        run_ffmpeg_command(cmd)

        progress(0.95, desc="Adding captions...")
        final_video_with_subs = os.path.join(OUT_DIR, "final_with_captions.mp4")
        drawtext_filters = []
        for i, (ts, caption) in enumerate(clean_captions):
            tts_data = tts_files[i] if i < len(tts_files) else (ts, None, 2.5)
            start = tts_data[0]
            end = start + (tts_data[2] if tts_data[2] else 2.5) + 0.2
            safe_caption = caption.replace("'", "\\'").replace(":", " ").replace(",", " ").replace('"', " ").strip()
            drawtext_filters.append(
                f"drawtext=text='{safe_caption}':fontfile='{FONT_PATH}':fontsize=36:fontcolor=white:"
                f"bordercolor=black:borderw=2:x=(w-text_w)/2:y=h-(text_h*2):enable='between(t,{start},{end})'"
            )
        filter_complex = ",".join(drawtext_filters)
        cmd_subs = [
            "ffmpeg", "-y",
            "-i", final_video_path,
            "-vf", filter_complex,
            "-c:a", "copy",
            final_video_with_subs
        ]
        run_ffmpeg_command(cmd_subs)
        progress(1.0)
        return final_video_with_subs

    except Exception as e:
        raise gr.Error(f"An error occurred: {e}")

with gr.Blocks(title="Video AI Dubber", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Video AI Dubber & Subtitler
    Upload a video file and generate captions, TTS voiceover, and final video.
    """)
    
    with gr.Row():
        video_in = gr.Video(label="Upload Video", source="upload")
        video_out = gr.Video(label="Processed Video")
    
    btn = gr.Button("Start Processing")
    btn.click(run_video_pipeline, inputs=[video_in], outputs=[video_out])

demo.launch(share=True)
