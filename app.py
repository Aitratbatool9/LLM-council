import json
import random
import re
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Council - Stage 1 & Stage 2 Pipeline",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 LLM Council — Multi-Agent Clinical Pipeline")
st.markdown("""
*Methodology:*
1. **Stage 1 (Generation):** 4 free LLMs generate independent clinical plans.
2. **Stage 2 (Anonymous Peer Review):** All responses are masked (Response A, B, C, D) and re-submitted to all models for peer critique and scoring.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Paste sk-or-v1-...", key="openrouter_key")
    
    st.subheader("Free Council Member Models")
    model_1 = st.text_input("Model 1", "openrouter/free", key="m1_stage2_v2")
    model_2 = st.text_input("Model 2", "meta-llama/llama-3.2-3b-instruct:free", key="m2_stage2_v2")
    model_3 = st.text_input("Model 3", "google/gemma-2-9b-it:free", key="m3_stage2_v2")
    model_4 = st.text_input("Model 4", "openrouter/auto", key="m4_stage2_v2")

COUNCIL_MODELS = [model_1, model_2, model_3, model_4]

# Robust Helper Function to Extract JSON
def extract_json(text):
    if not text:
        raise ValueError("Model returned empty response.")
        
    # Attempt 1: Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract code block content if wrapped in ```json ... ```
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Attempt 3: General greedy match for any JSON object structure
    greedy_match = re.search(r"\{.*\}", text, re.DOTALL)
    if greedy_match:
        try:
            return json.loads(greedy_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Model output could not be parsed into valid JSON.")

# Resilient Call function using OpenRouter Fallback Array & Enforced JSON Format
def call_openrouter_with_fallback(primary_model, messages, api_key_val):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_val,
    )
    fallback_models = [primary_model, "openrouter/free", "openrouter/auto"]
    
    response = client.chat.completions.create(
        model=primary_model,
        messages=messages,
        temperature=0.7,  # Slightly lower temperature for structured outputs
        top_p=1.0,
        max_tokens=2000,
        response_format={"type": "json_object"},
        extra_body={
            "models": fallback_models
        }
    )
    return response.choices[0].message.content

# Masking & Anonymizing Helper
def mask_stage1_responses(stage1_results):
    labels = ["Response A", "Response B", "Response C", "Response D"]
    
    combined = list(stage1_results.items())
    random.shuffle(combined)
    
    mapping = {}
    masked_blocks = []
    
    for idx, (model_name, response_data) in enumerate(combined):
        label = labels[idx]
        mapping[label] = model_name
        
        formatted_block = f"""---
### {label}
Primary Diagnosis: {response_data.get('primary_diagnosis', 'N/A')}
Differential Diagnoses: {', '.join(response_data.get('differential_diagnoses', []))}
Management Plan:
{chr(10).join(['  - ' + step for step in response_data.get('management_plan', [])])}
"""
        masked_blocks.append(formatted_block)
        
    return "\n".join(masked_blocks), mapping

# ------------------------------------------------------------------------------
# CLINICAL CASE INPUT
# ------------------------------------------------------------------------------
st.header("Enter Clinical Case Vignette")
default_vignette = """Patient presents with missing maxillary right first molar (#16).
CBCT reveals 4.5 mm vertical residual bone height below the maxillary sinus floor.
Sinus anatomy: straight floor, no septa, normal membrane thickness.
Bone quality: D3-D4 trabecular pattern.
Adjacent teeth (#15, #17) clinically healthy.
Medical history: Non-smoker, ASA II."""

case_vignette = st.text_area("Vignette Text", value=default_vignette, height=140, key="vignette_input")

# ------------------------------------------------------------------------------
# PIPELINE EXECUTION (STAGE 1 & STAGE 2)
# ------------------------------------------------------------------------------
if st.button("🚀 Run Pipeline (Stage 1 & Stage 2)", type="primary"):
    if not api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
        st.stop()

    # --- STAGE 1: INDEPENDENT GENERATION ---
    st.subheader("📍 Stage 1: Independent Generation")
    
    stage1_prompt = f"""You are an expert clinical council member. Evaluate the following case vignette.

Case Vignette:
{case_vignette}

You MUST output your response strictly in valid JSON format:
{{
  "primary_diagnosis": "Most likely primary diagnosis string",
  "differential_diagnoses": ["Diff 1", "Diff 2", "Diff 3", "Diff 4"],
  "management_plan": [
    "1. Surgical approach and sinus floor considerations",
    "2. Bone grafting material & membrane protocol",
    "3. Implant dimensions, primary stability, and insertion torque fallback rules",
    "4. Prosthodontic / occlusal design",
    "5. Periodontal biotype management & follow-up"
  ]
}}
"""

    stage1_results = {}
    cols1 = st.columns(4)

    for idx, model in enumerate(COUNCIL_MODELS):
        with cols1[idx]:
            st.markdown(f"### Model {idx+1}")
            st.caption(f"`{model}`")
            with st.spinner("Generating Plan..."):
                try:
                    messages = [{"role": "user", "content": stage1_prompt}]
                    raw_out = call_openrouter_with_fallback(model, messages, api_key)
                    parsed_json = extract_json(raw_out)
                    stage1_results[f"Model {idx+1} ({model})"] = parsed_json
                    
                    st.success("Stage 1 Complete")
                    st.write(f"**Diagnosis:** {parsed_json.get('primary_diagnosis', 'N/A')}")
                    with st.expander("Full Plan"):
                        st.json(parsed_json)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    # --- STAGE 2: ANONYMOUS MASKED PEER REVIEW ---
    st.subheader("📍 Stage 2: Anonymous Masked Peer Review")
    
    if len(stage1_results) < 2:
        st.error("Stage 2 requires at least 2 successful Stage 1 responses.")
        st.stop()

    # Perform Shuffling & Masking
    masked_text, label_mapping = mask_stage1_responses(stage1_results)
    
    with st.expander("🔍 Developer View: Anonymous Masking Key"):
        st.write("**Random Label Mapping for this run:**", label_mapping)
        st.text(masked_text)

    stage2_prompt = f"""You are an expert clinical council member participating in Stage 2: Peer Review.
Below are 4 anonymized clinical treatment plans (Response A, Response B, Response C, Response D).

Case Vignette:
{case_vignette}

Anonymized Plans:
{masked_text}

Critique ALL 4 responses objectively. Evaluate clinical accuracy, surgical safety, and grafting protocol completeness.
You MUST output your response strictly in valid JSON format:
{{
  "critique_response_a": {{"strengths": "Short summary of strengths", "weaknesses": "Short summary of weaknesses", "score_out_of_10": 8}},
  "critique_response_b": {{"strengths": "Short summary of strengths", "weaknesses": "Short summary of weaknesses", "score_out_of_10": 7}},
  "critique_response_c": {{"strengths": "Short summary of strengths", "weaknesses": "Short summary of weaknesses", "score_out_of_10": 9}},
  "critique_response_d": {{"strengths": "Short summary of strengths", "weaknesses": "Short summary of weaknesses", "score_out_of_10": 6}}
}}
"""

    cols2 = st.columns(4)

    for idx, model in enumerate(COUNCIL_MODELS):
        with cols2[idx]:
            st.markdown(f"### Reviewer {idx+1}")
            st.caption(f"`{model}`")
            with st.spinner("Evaluating Anonymous Plans..."):
                try:
                    messages = [{"role": "user", "content": stage2_prompt}]
                    raw_out = call_openrouter_with_fallback(model, messages, api_key)
                    parsed_critique = extract_json(raw_out)
                    
                    st.success("Review Complete")
                    for resp_key in ["critique_response_a", "critique_response_b", "critique_response_c", "critique_response_d"]:
                        item = parsed_critique.get(resp_key, {})
                        if item:
                            clean_label = resp_key.replace("critique_", "").replace("_", " ").title()
                            st.markdown(f"**{clean_label}** (`{item.get('score_out_of_10', 'N/A')}/10`)")
                            st.caption(f"👍 {item.get('strengths', 'N/A')}")
                            st.caption(f"⚠️ {item.get('weaknesses', 'N/A')}")
                except Exception as e:
                    st.error(f"Critique Error: {str(e)}")
