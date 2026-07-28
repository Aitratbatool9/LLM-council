import json
import re
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Council - Stage 1 (Fail-Safe Council)",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 LLM Council — Stage 1: Independent Generation")
st.markdown("""
*Methodology:* Submits the clinical case simultaneously to **4 free LLM council members** 
via OpenRouter using standardized hyperparameters (`temperature=1.0`, `top_p=1.0`) with automated fallback routing.
""")

# Sidebar for API key & Model Selection
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Paste sk-or-v1-...", key="openrouter_key")
    
    st.subheader("Free Council Member Models")
    model_1 = st.text_input("Model 1", "openrouter/free", key="m1_fallback_v6")
    model_2 = st.text_input("Model 2", "meta-llama/llama-3.2-3b-instruct:free", key="m2_fallback_v6")
    model_3 = st.text_input("Model 3", "google/gemma-2-9b-it:free", key="m3_fallback_v6")
    model_4 = st.text_input("Model 4", "openrouter/auto", key="m4_fallback_v6")

COUNCIL_MODELS = [model_1, model_2, model_3, model_4]

# Helper function to extract JSON cleanly
def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Model output could not be parsed into valid JSON.")

# Resilient Call function using OpenRouter's Native Extra Body Fallback Array
def call_openrouter_with_fallback(primary_model, messages, api_key_val):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_val,
    )
    
    # Fallback list for OpenRouter's extra_body model routing
    fallback_models = [primary_model, "openrouter/free", "openrouter/auto"]
    
    response = client.chat.completions.create(
        model=primary_model,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1500,
        extra_body={
            "models": fallback_models
        }
    )
    return response.choices[0].message.content

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

case_vignette = st.text_area("Vignette Text", value=default_vignette, height=150, key="vignette_input")

# ------------------------------------------------------------------------------
# STAGE 1 EXECUTION
# ------------------------------------------------------------------------------
if st.button("🚀 Run Stage 1 (Generate 4 Independent Plans)", type="primary"):
    if not api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
        st.stop()

    st.subheader("Executing Stage 1 Parallel Generation...")
    
    stage1_prompt = f"""You are an expert clinical council member. Evaluate the following case vignette.

Case Vignette:
{case_vignette}

You MUST output your response strictly in the following JSON format without any extra markdown wrapper text outside the JSON object:
{{
  "primary_diagnosis": "Most likely primary diagnosis string",
  "differential_diagnoses": [
    "Differential 1",
    "Differential 2",
    "Differential 3",
    "Differential 4"
  ],
  "management_plan": [
    "1. Surgical approach and sinus floor considerations",
    "2. Bone grafting material & membrane protocol",
    "3. Implant dimensions, primary stability, and insertion torque fallback rules",
    "4. Prosthodontic / occlusal design",
    "5. Periodontal biotype management & follow-up"
  ]
}}
"""

    cols = st.columns(4)

    for idx, model in enumerate(COUNCIL_MODELS):
        with cols[idx]:
            st.markdown(f"### Model {idx+1}")
            st.caption(f"`{model}`")
            with st.spinner("Generating..."):
                try:
                    messages = [{"role": "user", "content": stage1_prompt}]
                    raw_out = call_openrouter_with_fallback(model, messages, api_key)
                    parsed_json = extract_json(raw_out)
                    
                    st.success("Complete!")
                    st.markdown("**Primary Diagnosis:**")
                    st.write(parsed_json.get("primary_diagnosis", "N/A"))
                    
                    st.markdown("**Differential Diagnoses:**")
                    for diff in parsed_json.get("differential_diagnoses", []):
                        st.write(f"• {diff}")

                    st.markdown("**Management Array:**")
                    for step in parsed_json.get("management_plan", []):
                        st.write(f"- {step}")

                    with st.expander("Raw Output"):
                        st.json(parsed_json)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
