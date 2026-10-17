"""
Voice-Activated Home Assistant Robot for Bedridden Patients
-------------------------------------------------------------
Flow: Voice Input -> Language Detect+Translate -> Command Parsing ->
      Execute Action -> Translate Reply -> Speak -> Wait

Now supports MULTIPLE LANGUAGES:
- Patient can speak in Hindi, Tamil, Telugu, Spanish, etc.
- We detect the language and translate to English internally so our
  keyword rules (light/water/caregiver) still work no matter what
  language was spoken.
- The spoken reply is translated back and read out in the SAME
  language the patient used.
"""

import os
import re
import json
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Home Assistant Robot",
    page_icon="🤖",
    layout="centered"
)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY was not found in the .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

LLM_MODEL = "openai/gpt-oss-120b"

# --------------------------------------------------------------------------
# Wake word settings
# --------------------------------------------------------------------------
# Streamlit's mic recorder is "push to record", not always-on listening.
# So we simulate a wake word: the patient says the wake phrase AND the
# command in the same recording, e.g. "Hey Robo, turn on the light".
# We check the raw transcript for the wake phrase before doing anything else.
WAKE_PHRASES = ["hey robo", "hi robo", "ok robo", "okay robo", "hey, robo"]


def strip_wake_word(raw_text):
    """
    Checks if the recording starts with (or contains near the start) a
    wake phrase. Returns (wake_word_found, remaining_text_after_wake_word).
    """
    lowered = raw_text.lower().strip()
    for phrase in WAKE_PHRASES:
        if phrase in lowered:
            # Remove the wake phrase (case-insensitive) and any leftover
            # punctuation like a comma right after it.
            pattern = re.compile(re.escape(phrase) + r"[,]?\s*", re.IGNORECASE)
            remainder = pattern.sub("", raw_text, count=1).strip()
            return True, remainder
    return False, raw_text

# --------------------------------------------------------------------------
# Session state: device/robot status + history (persists across interactions)
# --------------------------------------------------------------------------
if "devices" not in st.session_state:
    st.session_state.devices = {
        "light": "OFF",
        "water_requested": False,
        "caregiver_alert": False,
    }

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: command, intent, response, time

if "stage" not in st.session_state:
    st.session_state.stage = "Wait"  # tracks current flowchart stage for display

# --------------------------------------------------------------------------
# Stage 1: Voice Input -> Speech to Text
# --------------------------------------------------------------------------
def transcribe_audio(audio_bytes):
    """
    Converts the patient's recorded voice into raw text, in whatever
    language they spoke. We don't force English here on purpose, so
    accents/other languages come through as-is.
    """
    transcription = client.audio.transcriptions.create(
        file=("recording.wav", audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="json",
        temperature=0.0
    )
    return transcription.text.strip()


# --------------------------------------------------------------------------
# NEW: Language detection + translation to English
# --------------------------------------------------------------------------
def detect_and_translate_to_english(text):
    """
    Uses the LLM to figure out what language the patient spoke in and
    translate it to English, so our simple keyword rules (light/water/
    caregiver) keep working no matter what language came in.

    Returns: (language_code, language_name, english_text)
    e.g. ("hi", "Hindi", "turn on the light")
    """
    prompt = f"""Identify the language of the text below and translate it to English.
Respond with ONLY valid JSON, no other words, no markdown fences, in this exact shape:
{{"language_code": "two-letter ISO 639-1 code, e.g. en, hi, ta, te, es, fr", "language_name": "language name in English, e.g. Hindi", "english_text": "the English translation"}}

Text: \"\"\"{text}\"\"\""""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
        temperature=0,
        max_tokens=300
    )

    raw = completion.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        return (
            data.get("language_code", "en"),
            data.get("language_name", "English"),
            data.get("english_text", text)
        )
    except (json.JSONDecodeError, AttributeError):
        # If anything goes wrong parsing, just fall back to treating it as English
        return "en", "English", text


def translate_from_english(text, language_code, language_name):
    """
    Translates our reply back into the patient's language before we
    speak it out loud. If they spoke English, we skip this entirely.
    """
    if language_code == "en":
        return text

    prompt = (
        f"Translate the following text into {language_name}. "
        f"Respond with ONLY the translation, nothing else, no quotes.\n\n"
        f"Text: {text}"
    )
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
        temperature=0,
        max_tokens=300
    )
    return completion.choices[0].message.content.strip()


# --------------------------------------------------------------------------
# Stage 2: Command Parsing -> classify the (English) text into an intent
# --------------------------------------------------------------------------
INTENT_PATTERNS = {
    "light_on": r"\b(turn on|switch on|put on)\b.*\blight\b",
    "light_off": r"\b(turn off|switch off|put off)\b.*\blight\b",
    "fetch_water": r"\b(water|thirsty|drink)\b",
    "call_caregiver": r"\b(help|nurse|caregiver|emergency|call someone)\b",
}


def parse_command(text):
    """
    Rule-based intent detection first (fast, predictable, no API cost).
    Falls back to the LLM only when no keyword pattern matches, so the
    robot can still respond sensibly to open-ended questions.
    Expects English text (already translated if needed).
    """
    lowered = text.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, lowered):
            return intent
    return "general_query"


# --------------------------------------------------------------------------
# Stage 3: Execute Action -> update simulated device state / generate reply
# --------------------------------------------------------------------------
def execute_action(intent, english_text):
    devices = st.session_state.devices

    if intent == "light_on":
        devices["light"] = "ON"
        return "Sure, I've turned the light on for you."

    if intent == "light_off":
        devices["light"] = "OFF"
        return "Okay, I've turned the light off."

    if intent == "fetch_water":
        devices["water_requested"] = True
        return "I'm bringing you a glass of water right now."

    if intent == "call_caregiver":
        devices["caregiver_alert"] = True
        return "I've alerted your caregiver. Help is on the way."

    # general_query -> fall back to a conversational AI response
    return generate_ai_response(english_text)


def generate_ai_response(question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a caring home-assistant robot speaking to a "
                    "bedridden patient. Answer clearly, warmly, and concisely "
                    "using simple language."
                )
            },
            {"role": "user", "content": question}
        ],
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=300
    )
    return chat_completion.choices[0].message.content


# --------------------------------------------------------------------------
# Stage 4/5 helper: Complete Task -> Text to Speech confirmation
# --------------------------------------------------------------------------
def convert_text_to_speech(text, lang_code="en"):
    """
    Speaks the reply out loud in the given language. gTTS supports most
    common language codes directly (en, hi, ta, te, es, fr, etc.).
    If an unsupported code sneaks through, we fall back to English so
    the app doesn't crash.
    """
    audio_buffer = BytesIO()
    try:
        gTTS(text=text, lang=lang_code).write_to_fp(audio_buffer)
    except ValueError:
        gTTS(text=text, lang="en").write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🤖 Home Assistant Robot")
st.caption("Voice-activated support for bedridden patients — start your recording with "
           "\"Hey Robo\" followed by your command, in any language, "
           "e.g. \"Hey Robo, turn on the light\" or \"Hey Robo, bring me water\".")

# --- Live status dashboard (reflects the "Wait" / idle state) ---
with st.sidebar:
    st.subheader("🏠 Home Status")
    d = st.session_state.devices
    st.write(f"💡 Light: **{d['light']}**")
    st.write(f"🚰 Water requested: **{'Yes' if d['water_requested'] else 'No'}**")
    if d["caregiver_alert"]:
        st.error("🚨 Caregiver has been alerted!")
    else:
        st.write("🚨 Caregiver alert: **None**")

    if st.button("Reset device status"):
        st.session_state.devices = {
            "light": "OFF", "water_requested": False, "caregiver_alert": False
        }
        st.rerun()

    st.divider()
    st.subheader("🕓 Command History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**{item['time']}** — _{item['intent']}_ ({item.get('language', 'English')})  \n"
                        f"\"{item['command']}\"")
    else:
        st.write("No commands yet.")

st.markdown(f"**Current stage:** `{st.session_state.stage}`")

audio_value = st.audio_input("🎙️ Record your command — say \"Hey Robo\" first (any language)")

if audio_value is not None:
    st.session_state.stage = "Voice Input"
    st.audio(audio_value)

    with st.spinner("Converting speech to text..."):
        transcribed_text = transcribe_audio(audio_value.getvalue())

    if not transcribed_text:
        st.warning("No speech was detected. Please record your command again.")
    else:
        st.subheader("You said")
        st.write(transcribed_text)

        # NEW STAGE: Wake word check
        st.session_state.stage = "Wake Word Check"
        wake_found, transcribed_text = strip_wake_word(transcribed_text)

        if not wake_found:
            st.info("🤖 I only respond when you start with \"Hey Robo\". "
                     "Try again, e.g. \"Hey Robo, turn on the light\".")
            st.session_state.stage = "Wait"
        elif not transcribed_text.strip():
            st.info("🤖 Yes? I'm listening — go ahead and tell me what you need.")
            st.session_state.stage = "Wait"
        else:
            st.caption("✅ Wake word detected")

            # NEW STAGE: Language Detect + Translate to English
            st.session_state.stage = "Language Detection"
            with st.spinner("Detecting language..."):
                lang_code, lang_name, english_text = detect_and_translate_to_english(transcribed_text)
            st.caption(f"Detected language: `{lang_name}` ({lang_code})")

            # Stage 2: Command Parsing (on the English version)
            st.session_state.stage = "Command Parsing"
            intent = parse_command(english_text)
            st.caption(f"Detected intent: `{intent}`")

            # Stage 3: Execute Action
            st.session_state.stage = "Execute Action"
            with st.spinner("Executing action..."):
                response_text_en = execute_action(intent, english_text)

            # Translate the reply back into the patient's language
            with st.spinner("Preparing reply..."):
                response_text = translate_from_english(response_text_en, lang_code, lang_name)

            # Stage 4: Complete Task
            st.session_state.stage = "Complete Task"
            st.subheader("🤖 Assistant")
            st.write(response_text)

            with st.spinner("Generating voice response..."):
                response_audio = convert_text_to_speech(response_text, lang_code)
            st.audio(response_audio, format="audio/mp3")

            st.session_state.history.append({
                "command": transcribed_text,
                "intent": intent,
                "response": response_text,
                "language": lang_name,
                "time": datetime.now().strftime("%H:%M:%S"),
            })

            # Stage 5: Wait (ready for next command)
            st.session_state.stage = "Wait"
            st.success("Ready for the next command.")