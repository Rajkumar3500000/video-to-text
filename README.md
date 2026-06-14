# Video → Text Converter
### Flask + Whisper + SpeechRecognition | No API Key Required

Convert any video's speech to text, generate a summary, and download a PDF — fully offline.

---

## Features

- 🤫 **Whisper (offline)** — OpenAI's local speech model, works without internet
- 🌐 **Google Speech** — online fallback with language selection
- 🌍 **Multi-language** — Auto-detect or choose: English, Tamil, Hindi, French, German, Spanish, Chinese, Arabic, Japanese
- 📋 **Auto Summary** — extractive summarization (no ML model, no API)
- 📄 **PDF Download** — formatted report with transcript + summary
- 🎬 **Video Preview** — play the video in-browser before converting

---

## Project Structure

```
video-to-text/
├── app.py               ← Flask backend
├── requirements.txt     ← Python packages
├── README.md
├── uploads/             ← Temp files (auto-created, auto-deleted)
└── templates/
    └── index.html       ← Frontend UI
```

---

## Setup (English)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 2. Install packages
pip install -r requirements.txt

# 3. Run
python app.py
```

Open browser → http://localhost:5000

---

## Setup (Tamil / தமிழ்)

```bash
# 1. Virtual environment உருவாக்கு
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

# 2. Package install செய்
pip install -r requirements.txt

# 3. App ஓட்டு
python app.py
```

Browser-ல் http://localhost:5000 திற.

---

## How It Works

```
Video Upload → Extract Audio (MoviePy)
           → Transcribe (Whisper / Google SR)
           → Summarize (Extractive, no API)
           → Generate PDF (ReportLab)
           → Display in Browser
```

---

## Supported Formats

`mp4  mov  avi  mkv  webm  flv  wmv`

---

## Notes

- Whisper `base` model downloads ~140 MB on first run (cached after that)
- For better accuracy, change `whisper.load_model("base")` → `"medium"` in `app.py`
- Google Speech requires an internet connection
- All uploaded files are deleted automatically after processing
