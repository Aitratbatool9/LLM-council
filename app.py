import json
import random
import re
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Council — Complete 3-Stage Pipeline",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 LLM Council — Multi-Agent Clinical Pipeline")
st.markdown("""
*Methodology:*
1. **Stage 1 (Generation):** 4 free LLMs generate independent clinical plans.
2. **Stage 2 (Anonymous Peer Review):** All responses are masked (A, B, C, D) and evaluated by all models.
3. **Stage 3 (Council Chair Synthesis):** A Council Chair model synthesizes all plans + peer critiques to declare the winning plan and final consensus recommendation.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Paste sk-or-v1-...", key="openrouter_key")
    
    st.subheader("Council Member Models")
    model_1 = st.text_input("Model 1", "openrouter/free", key="m1_s3_v2")
    model_2 = st.text_input("Model 2", "meta-llama/llama-3.2-3b-instruct:free", key="m2_s3_v2")
    model_3 = st.text_input("Model 3", "google/gemma-2-9b-it:free", key="m3_s3_v2")
    model_4 = st.text_input("Model 4", "openrouter/auto", key="m4_s3_v2")
    
    st.subheader("👑 Council Chair Model")
    chair_model = st.selectbox(
        "Select Chair Model",
        ["openrouter/auto", "google/gemma-2-9b-it:free", "openrouter/free"],
        index=0,
        key="chair_select_v2"
    )

COUNCIL_MODELS = [model_1, model_2, model_3, model_4]

# Resilient Bulletproof JSON Extractor
def extract_json(text):
    if not text:
        raise ValueError("Model returned empty response.")
        
    # Attempt 1: Direct JSON parsing
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Attempt 2: Strip out markdown formatting (```json ... ```)
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("
