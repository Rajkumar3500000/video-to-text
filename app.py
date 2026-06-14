from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
import speech_recognition as sr
import whisper
import os

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"}

# Load once at startup — "base" is fast; change to "medium" for better accuracy
print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")
print("Whisper ready.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(video_path: str, audio_path: str) -> str | None:
    """Extract audio WAV from video. Returns error string or None on success."""
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        return f"Could not open video: {e}"

    if clip.audio is None:
        clip.close()
        return "This video has no audio track."

    if clip.audio.duration is None or clip.audio.duration < 1:
        clip.close()
        return "Audio is too short to transcribe."

    clip.audio.write_audiofile(
        audio_path,
        fps=16000,
        codec="pcm_s16le",
        ffmpeg_params=["-ac", "1"],
        logger=None
    )
    clip.close()

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return "Audio extraction produced an empty file."

    return None  # success


def transcribe_whisper(audio_path: str, language: str) -> dict:
    """Use OpenAI Whisper locally — no internet needed."""
    kwargs = {"fp16": False}
    if language != "auto":
        kwargs["language"] = language

    result = whisper_model.transcribe(audio_path, **kwargs)
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
        "method": "Whisper (offline)"
    }


def transcribe_google_sr(audio_path: str, language: str) -> dict:
    """Use SpeechRecognition with Google Web Speech (needs internet)."""
    lang_map = {"en": "en-IN", "ta": "ta-IN", "hi": "hi-IN", "auto": "en-IN"}
    google_lang = lang_map.get(language, "en-IN")

    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)

    text = r.recognize_google(audio_data, language=google_lang)
    return {
        "text": text.strip(),
        "language": language,
        "method": "Google Speech Recognition"
    }


def summarize_text(text: str) -> str:
    """
    Simple extractive summarizer — no API, no ML model needed.
    Picks top sentences by word-frequency scoring.
    """
    import re
    from collections import Counter

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 3:
        return text  # already short

    # Remove stopwords
    stopwords = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could","should",
        "may","might","shall","can","and","or","but","so","yet","for","nor",
        "in","on","at","to","of","by","with","from","this","that","these",
        "those","it","its","i","we","you","he","she","they","my","our","your",
        "his","her","their","not","no","as","if","then","than","also","just",
        "been","into","about","which","who","what","when","where","how"
    }

    words = re.findall(r'\b[a-z]+\b', text.lower())
    freq = Counter(w for w in words if w not in stopwords)

    if not freq:
        return text[:800]

    max_freq = max(freq.values())
    freq = {w: v / max_freq for w, v in freq.items()}

    def score(sentence):
        s_words = re.findall(r'\b[a-z]+\b', sentence.lower())
        return sum(freq.get(w, 0) for w in s_words) / (len(s_words) + 1)

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    # Take top 40% of sentences, preserve order
    top_n = max(3, len(sentences) // 3)
    top_indices = sorted(i for i, _ in scored[:top_n])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary


def create_pdf(filename_base: str, original_text: str, summary: str,
               language: str, method: str) -> str:
    """Generate a formatted PDF with transcript + summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"result_{timestamp}.pdf"
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

    WIDTH, HEIGHT = A4
    MARGIN = 50
    LINE_H = 18
    MAX_CHARS = 95

    c = canvas.Canvas(pdf_path, pagesize=A4)

    def new_page():
        c.showPage()
        return HEIGHT - MARGIN

    def write_wrapped(text_block: str, y: float, font="Helvetica", size=11) -> float:
        c.setFont(font, size)
        words_list = text_block.split()
        line = ""
        for word in words_list:
            if len(line + word) < MAX_CHARS:
                line += word + " "
            else:
                c.drawString(MARGIN, y, line.strip())
                y -= LINE_H
                if y < MARGIN + 30:
                    y = new_page()
                    c.setFont(font, size)
                line = word + " "
        if line.strip():
            c.drawString(MARGIN, y, line.strip())
            y -= LINE_H
        return y

    # ── Header ──
    c.setFillColorRGB(0.08, 0.08, 0.22)
    c.rect(0, HEIGHT - 70, WIDTH, 70, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(MARGIN, HEIGHT - 42, "Video → Text Report")
    c.setFont("Helvetica", 10)
    c.drawString(MARGIN, HEIGHT - 60,
                 f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}   |   "
                 f"Engine: {method}   |   Language: {language.upper()}")

    y = HEIGHT - 90
    c.setFillColorRGB(0, 0, 0)

    # ── Summary section ──
    c.setFillColorRGB(0.95, 0.97, 1.0)
    c.rect(MARGIN - 8, y - (LINE_H * (len(summary) // 90 + 4)) - 10,
           WIDTH - 2 * MARGIN + 16, LINE_H * (len(summary) // 90 + 4) + 20,
           fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.1, 0.1, 0.6)
    c.drawString(MARGIN, y, "📋  Summary")
    y -= LINE_H + 4
    c.setFillColorRGB(0, 0, 0)
    y = write_wrapped(summary, y, font="Helvetica", size=11)
    y -= 16

    # Divider
    c.setStrokeColorRGB(0.8, 0.8, 0.9)
    c.setLineWidth(1)
    c.line(MARGIN, y, WIDTH - MARGIN, y)
    y -= 20

    # ── Full Transcript ──
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.1, 0.1, 0.6)
    c.drawString(MARGIN, y, "📝  Full Transcript")
    y -= LINE_H + 4
    c.setFillColorRGB(0, 0, 0)

    if y < MARGIN + 30:
        y = new_page()

    y = write_wrapped(original_text, y, font="Helvetica", size=11)

    # ── Footer on last page ──
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(MARGIN, 30, "Generated by Video-to-Text App — Whisper + SpeechRecognition (No API key required)")

    c.save()
    return pdf_filename


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/converter", methods=["POST"])
def convert():
    result = {
        "text": "",
        "summary": "",
        "message": "",
        "pdf_filename": "",
        "language": "",
        "method": "",
        "success": False
    }

    # ── Validate upload ──
    if "video" not in request.files or request.files["video"].filename == "":
        result["message"] = "Please select a video file."
        return jsonify(result), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        result["message"] = f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        return jsonify(result), 400

    engine = request.form.get("engine", "whisper")         # whisper | google
    language = request.form.get("language", "auto")        # auto | en | ta | hi

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
    audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{timestamp}.wav")

    try:
        file.save(video_path)

        # ── Extract audio ──
        err = extract_audio(video_path, audio_path)
        if err:
            result["message"] = err
            return jsonify(result), 422

        # ── Transcribe ──
        if engine == "google":
            try:
                transcription = transcribe_google_sr(audio_path, language)
            except sr.UnknownValueError:
                result["message"] = "Google could not understand the audio. Try Whisper instead."
                return jsonify(result), 422
            except sr.RequestError:
                result["message"] = "Google Speech service unavailable. Try Whisper (offline)."
                return jsonify(result), 503
        else:
            transcription = transcribe_whisper(audio_path, language)

        text = transcription["text"]
        if not text:
            result["message"] = "No speech detected in the video."
            return jsonify(result), 422

        summary = summarize_text(text)
        pdf_filename = create_pdf(
            filename, text, summary,
            transcription["language"], transcription["method"]
        )

        result.update({
            "text": text,
            "summary": summary,
            "message": f"✓ Done! Detected language: {transcription['language'].upper()} · Engine: {transcription['method']}",
            "pdf_filename": pdf_filename,
            "language": transcription["language"],
            "method": transcription["method"],
            "success": True
        })
        return jsonify(result)

    except Exception as e:
        result["message"] = f"Unexpected error: {str(e)}"
        return jsonify(result), 500

    finally:
        for p in [video_path, audio_path]:
            if os.path.exists(p):
                os.remove(p)


@app.route("/download-pdf/<filename>")
def download_pdf(filename):
    pdf_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(pdf_path):
        return "PDF not found.", 404
    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
