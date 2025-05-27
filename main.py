import logging
import os
import shutil
import uuid
import time
import threading
from enum import Enum
from typing import List

import ffmpeg
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, File, UploadFile
from pathlib import Path

FFMPEG_PATH = Path(__file__).parent / "ffmpeg" / "ffmpeg.exe"
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image
from rembg import remove
from werkzeug.utils import secure_filename

# ───────────────── Chat‑related imports ─────────────────
from chat import get_org_chain, get_general_chain, clear_memory, _llm

# ─────────────────── logging ───────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────── FastAPI APP ───────────────────
app = FastAPI(title="TTT‑Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────── Chat models ───────────────────
class Mode(str, Enum):
    org = "org"
    general = "general"

class ChatRequest(BaseModel):
    question: str
    session_id: str = Field("default")
    mode: Mode = Field(Mode.org)

class ChatResponse(BaseModel):
    answer: str
    source_documents: list[str] | None = None  # org เท่านั้น

# ─────────────────── Chat endpoints ───────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """LLM‑powered Q&A endpoint.  Supports two retrieval chains: org / general."""
    try:
        chain = (
            get_org_chain(req.session_id)
            if req.mode == Mode.org
            else get_general_chain(req.session_id)
        )
        result = chain.invoke(
            {"question": req.question},
            config={"configurable": {"session_id": req.session_id}},
        )

        answer = (
            result.get("answer")
            or result.get("output")
            or result.get("text")
            or ""
        )
        if req.mode == Mode.org and not answer.strip():
            answer = (
                "ขอโทษนะคะ ไม่พบข้อมูลขององค์กรที่เกี่ยวข้องกับคำถามนี้ "
                "ลองถามในโหมดทั่วไปดูนะคะ"
            )

        src_docs = (
            [d.page_content for d in result.get("source_documents", [])]
            if req.mode == Mode.org
            else None
        )
        return ChatResponse(answer=answer, source_documents=src_docs)

    except Exception as exc:
        logger.exception("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/clear")
async def clear_chat(session_id: str = "default"):
    """Clear vector‑store memory for a given session id."""
    clear_memory(session_id)
    return {"status": "success", "message": f"Cleared session '{session_id}'"}

# ─────────────────── Media‑processing configuration ───────────────────
UPLOAD_FOLDER = "TEMP"
UPLOAD_FOLDER_VIDEO = "uploadsVideo2Text/"
UPLOAD_FOLDER_IMG_INPUT = "uploadsIMG/input"
UPLOAD_FOLDER_IMG_OUTPUT = "uploadsIMG/output"
MIN_FREE_SPACE = 500 * 1024 * 1024  # 500 MB

for folder in (
    UPLOAD_FOLDER,
    UPLOAD_FOLDER_VIDEO,
    UPLOAD_FOLDER_IMG_INPUT,
    UPLOAD_FOLDER_IMG_OUTPUT,
):
    os.makedirs(folder, exist_ok=True)

# ─────────────────── Utility helpers ───────────────────

def check_disk_space() -> bool:
    """Return *True* if there is ≥ MIN_FREE_SPACE available inside *UPLOAD_FOLDER*."""
    total, used, free = shutil.disk_usage(UPLOAD_FOLDER)
    return free >= MIN_FREE_SPACE


def clean_up_temp_files(paths: List[str]) -> None:
    """Delete the temporary *paths* if they still exist."""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Deleted temporary file: %s", path)
        except OSError as exc:
            logger.error("Error deleting %s: %s", path, exc)


def clean_folder(folder_path: str) -> None:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
        except OSError:
            pass

# ─────────────────── Internal STT helper ───────────────────

def _speech_to_text(wav_path: str) -> str:
    """Google Speech Recognition (TH). Return raw transcript (lowercase)."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source, duration=220)
        try:
            return recognizer.recognize_google(audio, language="th").lower()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"ไม่สามารถติดต่อบริการถอดเสียงได้: {exc}",
            )

# ─────────────────── Media endpoints ───────────────────
@app.post("/api/NewS2T_V2")
async def new_s2t_v2(Profile: List[UploadFile] = File(...)):
    """Video (MP4) → single‑pass speech‑to‑text (TH) using Google STT."""
    if not Profile:
        raise HTTPException(status_code=400, detail="No file selected")

    unique_filename = f"{uuid.uuid4()}.mp4"
    file_path = os.path.join(UPLOAD_FOLDER_VIDEO, secure_filename(unique_filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(Profile[0].file, buffer)

    try:
        wav_path = os.path.join(UPLOAD_FOLDER_VIDEO, f"{uuid.uuid4()}.wav")
        ffmpeg.input(file_path).output(wav_path).run(overwrite_output=True, quiet=True, cmd=str(FFMPEG_PATH))
        text = _speech_to_text(wav_path)
        return {"text": text}
    finally:
        clean_up_temp_files([file_path, wav_path])
        clean_folder(UPLOAD_FOLDER_VIDEO)


@app.post("/api/ContentVideo2Text")
async def content_video_to_text(Profile: List[UploadFile] = File(...)):
    """Video (MP4) → MP3 → WAV → text.  Useful when Google rejects direct MP4."""
    if not Profile:
        raise HTTPException(status_code=400, detail="No file selected")

    mp4_path = os.path.join(UPLOAD_FOLDER_VIDEO, secure_filename(f"{uuid.uuid4()}.mp4"))
    with open(mp4_path, "wb") as buffer:
        shutil.copyfileobj(Profile[0].file, buffer)

    try:
        mp3_path = os.path.join(UPLOAD_FOLDER_VIDEO, f"{uuid.uuid4()}.mp3")
        wav_path = os.path.join(UPLOAD_FOLDER_VIDEO, f"{uuid.uuid4()}.wav")
        ffmpeg.input(mp4_path).output(mp3_path).run(overwrite_output=True, quiet=True, cmd=str(FFMPEG_PATH))
        ffmpeg.input(mp3_path).output(wav_path).run(overwrite_output=True, quiet=True, cmd=str(FFMPEG_PATH))
        text = _speech_to_text(wav_path)
        return {"text": text}
    finally:
        clean_up_temp_files([mp4_path, mp3_path, wav_path])
        clean_folder(UPLOAD_FOLDER_VIDEO)


@app.post("/api/RemoveBG")
async def remove_bg(Profile: List[UploadFile] = File(...)):
    """Remove background from a *single* PNG/JPG and return a transparent PNG."""
    if not check_disk_space():
        raise HTTPException(status_code=500, detail="พื้นที่ดิสก์ไม่เพียงพอ")
    if not Profile:
        raise HTTPException(status_code=400, detail="ไม่มีไฟล์ถูกเลือก")

    file = Profile[0]
    if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
        raise HTTPException(status_code=400, detail=f"ไฟล์ '{file.filename}' ไม่ใช่รูปภาพที่รองรับ")

    input_path = os.path.join(UPLOAD_FOLDER_IMG_INPUT, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with Image.open(input_path) as image:
        try:
            output_image = remove(image)
        except OSError as exc:
            if "No space left on device" in str(exc):
                raise HTTPException(status_code=500, detail="ไม่มีพื้นที่ว่างเพียงพอในการจัดเก็บไฟล์")
            raise
        output_path = os.path.join(UPLOAD_FOLDER_IMG_OUTPUT, f"output_{file.filename}.png")
        output_image.save(output_path, format="PNG")

    def _cleanup():
        time.sleep(1)
        clean_folder(UPLOAD_FOLDER_IMG_INPUT)
        clean_folder(UPLOAD_FOLDER_IMG_OUTPUT)
    threading.Thread(target=_cleanup, daemon=True).start()

    return FileResponse(path=output_path, media_type="image/png", filename=f"output_{file.filename}")


# ─────────────────── Media → Summary endpoint ───────────────────
@app.post("/api/MediaSummary")
async def media_summary(Profile: List[UploadFile] = File(...), session_id: str = "default"):
    """Accept video/audio, convert to text with STT, then summarize via LLM."""
    if not Profile:
        raise HTTPException(status_code=400, detail="No file selected")

    # 1️⃣ Detect & save
    uploaded = Profile[0]
    ext = os.path.splitext(uploaded.filename)[1].lower()
    raw_path = os.path.join(UPLOAD_FOLDER_VIDEO, secure_filename(f"{uuid.uuid4()}{ext}"))
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(uploaded.file, buffer)

    temp_paths: List[str] = [raw_path]
    wav_path = None

    try:
        # 2️⃣ Convert → WAV if necessary
        if ext in (".wav",):
            wav_path = raw_path
        else:
            wav_path = os.path.join(UPLOAD_FOLDER_VIDEO, f"{uuid.uuid4()}.wav")
            ffmpeg.input(raw_path).output(wav_path).run(overwrite_output=True, quiet=True, cmd=str(FFMPEG_PATH))
            temp_paths.append(wav_path)

        # 3️⃣ STT
        text = _speech_to_text(wav_path)
        if not text:
            return {"summary": "", "raw_text": ""}

        # 4️⃣ Summarize
        summarizer_prompt = (
            "สรุปประเด็นสำคัญภาษาไทยจากข้อความถอดเสียงต่อไปนี้ :\n" + text
        )
        summary = _llm().predict(summarizer_prompt)
        return {"summary": summary.strip(), "raw_text": text.strip()}

    finally:
        clean_up_temp_files(temp_paths)
        clean_folder(UPLOAD_FOLDER_VIDEO)

# ─────────────────── Root health check ───────────────────
@app.get("/")
def root():
    return {"message": "TTT‑Assistant backend is running"}
