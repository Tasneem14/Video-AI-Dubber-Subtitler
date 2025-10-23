# Video AI Dubber & Subtitler

This project automatically converts any video into a version with **captions** and **voice-over** using AI.

---

##  Features

- Detect scenes and split the video into meaningful clips.
- Extract a keyframe from each scene.
- Generate English captions for each frame using BLIP.
- Convert captions to speech using gTTS.
- Merge the new audio with the original video.
- Burn captions onto the video as subtitles.

---

##  Requirements

- Python >= 3.8
- PyTorch
- Transformers
- OpenCV
- PySceneDetect
- gTTS
- pydub
- FFmpeg
- Gradio

## How to Run
  
```bash
pip install gradio pyscenedetect "pyscenedetect[opencv]" transformers torch gtts pydub requests
git clone <Tasneem14/Video-AI-Dubber-Subtitler>
cd <project-folder>
python main.py
```

## Demo Video

You can see a demo of the project in the video below:

[demo.mp4](blip_demo.mp4)


