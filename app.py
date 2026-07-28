import json
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Council - Stage 1",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 LLM Council — Stage 1: Independent Generation")
st.markdown("""
*Methodology:* Submits the clinical case simultaneously to **4 individual LLM council members** 
using standardized hyperparameters (temp=1.0, top_p=1.0) and enforces structured JSON output.
""")

# Sidebar for API key & Model Selection
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Paste sk-or-v1-...", key="openrouter_key")
    
    st.subheader("Council Member Models")
    model_1 = st.text_input("Model 1", "anthropic/claude-3.5-sonnet", key="m1_input")
    model_2 = st.text_input("Model 2", "openai/gpt-4o", key="m2_input")
    model_3 = st.text_input("Model 3", "google/gemini-pro-1.5", key="m3_input")
    model_4 = st.text_input("Model 4", "meta-llama/llama-3.3-70b-instruct", key="m4_input")
# Function to execute OpenRouter API call
def call_openrouter(model_name, messages, api_key_val):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_val,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1500,  # Prevents token allocation/402 credit errors
        response_format={"type": "json_object"}
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

You MUST output your response strictly in the following JSON format:
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
                    raw_out = call_openrouter(model, messages, api_key)
                    parsed_json = json.loads(raw_out)
                    
                    st.success("Complete!")
                    st.markdown("**Primary Diagnosis:**")
                    st.write(parsed_json.get("primary_diagnosis", "N/A"))
                    
                    st.markdown("**Differential Diagnoses:**")
                    for diff in parsed_json.get("differential_diagnoses", []):
                        st.write(f"• {diff}")

                    st.markdown("**Management Array:**")
                    for step in parsed_json.get("management_plan", []):
                        st.write(f"- {step}")

                    with st.expander("Raw JSON Output"):
                        st.json(parsed_json)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
