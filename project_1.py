import io
import os
import re
import hashlib
import tempfile
import subprocess
import numpy as np
import imageio_ffmpeg
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
import streamlit as st

load_dotenv()

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WHISPER_SIZE = "base"

st.title("YouTube ChatBot")


@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0,
        reasoning_effort="low",
    )


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def decode_audio(path=None, data=None):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", path if path else "pipe:0",
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def clean(text):
    text = re.sub(r"\[[^\]]*\]|♪", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def captions_text(video_id):
    api = YouTubeTranscriptApi()
    try:
        data = api.fetch(video_id, languages=["en", "hi"])
    except NoTranscriptFound:
        data = next(iter(api.list(video_id))).fetch()
    return clean(" ".join(c.text for c in data))


def whisper_text(video_id):
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp, "audio.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        audio_file = os.path.join(tmp, os.listdir(tmp)[0])
        segments, _ = get_whisper().transcribe(decode_audio(path=audio_file), vad_filter=True)
        return clean(" ".join(seg.text for seg in segments))


def fetch_text(video_id):
    try:
        text = captions_text(video_id)
        if len(text) >= 50:
            return text
    except (TranscriptsDisabled, NoTranscriptFound, StopIteration):
        pass

    text = whisper_text(video_id)
    if len(text) < 20:
        raise ValueError("Could not get any speech or lyrics from this video.")
    return text


@st.cache_resource(show_spinner=False)
def get_retriever(video_id):
    transcript = fetch_text(video_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(docs, get_embeddings())
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def transcribe_mic(audio_bytes):
    segments, _ = get_whisper().transcribe(decode_audio(data=audio_bytes), vad_filter=True)
    return clean(" ".join(seg.text for seg in segments))


@st.cache_data(show_spinner=False)
def speak(text):
    from gtts import gTTS
    text = re.sub(r"[*_`#>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:3000]
    fp = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(fp)
    return fp.getvalue()


def extract_video_id(text):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else text.strip()


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


prompt = PromptTemplate.from_template(
    "You are answering questions about a song or video using its transcript.\n"
    "Always reply in English.\n"
    "- Explain meaning, themes, mood and story in your own words.\n"
    "- Do not reproduce the full lyrics; short phrases are fine.\n"
    "- for song provide full lyrics; if ask provide context related to song.\n"
    "- If the transcript doesn't cover the question, say so briefly.\n"
    "- If the question is unrelated to the video, reply exactly: I don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


video_input = st.text_input("Enter YouTube video URL or ID:")

if st.button("Load Video"):
    video_id = extract_video_id(video_input)
    try:
        with st.spinner("Loading video (no captions = transcribing audio, may take a few minutes)..."):
            get_retriever(video_id)
        st.session_state.video_id = video_id
        st.session_state.messages = []
        st.success("Video loaded successfully!")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error: {e}")


if "video_id" in st.session_state:
    retriever = get_retriever(st.session_state.video_id)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_model()
        | StrOutputParser()
    )

    with st.sidebar:
        st.header("Voice")
        tts_on = st.toggle("Read answers aloud", value=False)
        mic_audio = st.audio_input("Ask with your voice")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question:")

    if not query and mic_audio:
        data = mic_audio.getvalue()
        audio_hash = hashlib.md5(data).hexdigest()
        if audio_hash != st.session_state.get("last_audio"):
            st.session_state.last_audio = audio_hash
            with st.spinner("Listening..."):
                query = transcribe_mic(data)
            if not query:
                st.warning("Couldn't hear anything. Please try again.")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            answer = st.write_stream(chain.stream(query))
            if tts_on and answer:
                try:
                    with st.spinner("Generating voice..."):
                        st.audio(speak(answer), format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.warning(f"Voice failed (needs internet): {e}")
        st.session_state.messages.append({"role": "assistant", "content": answer})