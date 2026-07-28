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
    model_1 = st.text_input("Model 1", "openrouter/free", key="m1_s3")
    model_2 = st.text_input("Model 2", "meta-llama/llama-3.2-3b-instruct:free", key="m2_s3")
    model_3 = st.text_input("Model 3", "google/gemma-2-9b-it:free", key="m3_s3")
    model_4 = st.text_input("Model 4", "openrouter/auto", key="m4_s3")
    
    st.subheader("👑 Council Chair Model")
    chair_model = st.selectbox(
        "Select Chair Model",
        ["openrouter/auto", "google/gemma-2-9b-it:free", "openrouter/free"],
        index=0,
        key="chair_select"
    )

COUNCIL_MODELS = [model_1, model_2, model_3, model_4]

# Extremely Forgiving JSON Extractor
def extract_json(text):
    if not text:
        raise ValueError("Model returned empty response.")
        
    # Standard JSON attempt
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Clean potential triple backticks or markdown preamble
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract first '{' to last '}'
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError("Could not parse JSON response.")

# OpenRouter API call without rigid JSON response_format
def call_openrouter_with_fallback(primary_model, messages, api_key_val, max_tokens=2000):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_val,
    )
    fallback_models = [primary_model, "openrouter/auto", "openrouter/free"]
    
    response = client.chat.completions.create(
        model=primary_model,
        messages=messages,
        temperature=0.3,  # Low temperature for precise JSON generation
        top_p=0.9,
        max_tokens=max_tokens,
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
# PIPELINE EXECUTION (STAGES 1, 2, & 3)
# ------------------------------------------------------------------------------
if st.button("🚀 Run Full Pipeline (Stages 1, 2 & 3)", type="primary"):
    if not api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
        st.stop()

    # ==========================================================================
    # STAGE 1: INDEPENDENT GENERATION
    # ==========================================================================
    st.subheader("📍 Stage 1: Independent Generation")
    
    stage1_prompt = f"""You are an expert clinical council member. Output ONLY raw JSON matching this structure. Do NOT include markdown text outside JSON.

Case Vignette:
{case_vignette}

Format:
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
                    stage1_results[f"Model {idx+1}"] = parsed_json
                    
                    st.success("Stage 1 Complete")
                    st.write(f"**Diagnosis:** {parsed_json.get('primary_diagnosis', 'N/A')}")
                    with st.expander("Full Plan"):
                        st.json(parsed_json)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    # ==========================================================================
    # STAGE 2: ANONYMOUS MASKED PEER REVIEW
    # ==========================================================================
    st.subheader("📍 Stage 2: Anonymous Masked Peer Review")
    
    if len(stage1_results) < 2:
        st.error("Stage 2 requires at least 2 successful Stage 1 responses.")
        st.stop()

    masked_text, label_mapping = mask_stage1_responses(stage1_results)
    
    with st.expander("🔍 Developer View: Anonymous Masking Key"):
        st.write("**Random Label Mapping for this run:**", label_mapping)
        st.text(masked_text)

    stage2_prompt = f"""You are an expert clinical council member performing Stage 2 Peer Review.
Evaluate the 4 anonymized clinical treatment plans below.
Output ONLY raw JSON format without any conversational intro text or markdown outside JSON.

Case Vignette:
{case_vignette}

Anonymized Plans:
{masked_text}

JSON Format Required:
{{
  "critique_response_a": {{"strengths": "Short strengths", "weaknesses": "Short weaknesses", "score_out_of_10": 8}},
  "critique_response_b": {{"strengths": "Short strengths", "weaknesses": "Short weaknesses", "score_out_of_10": 7}},
  "critique_response_c": {{"strengths": "Short strengths", "weaknesses": "Short weaknesses", "score_out_of_10": 9}},
  "critique_response_d": {{"strengths": "Short strengths", "weaknesses": "Short weaknesses", "score_out_of_10": 6}}
}}
"""

    stage2_reviews = {}
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
                    stage2_reviews[f"Reviewer {idx+1}"] = parsed_critique
                    
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

    st.divider()

    # ==========================================================================
    # STAGE 3: COUNCIL CHAIR SYNTHESIS
    # ==========================================================================
    st.subheader("👑 Stage 3: Council Chair Synthesis & Final Decision")
    
    with st.spinner(f"Council Chair (`{chair_model}`) synthesizing Stage 1 plans & Stage 2 peer critiques..."):
        try:
            stage3_prompt = f"""You are the Chair of the Clinical Council. 
Synthesize a final consensus decision based on the case, Stage 1 plans, and Stage 2 critiques.
Output ONLY raw JSON format.

Case Vignette:
{case_vignette}

Stage 1 Independent Plans:
{json.dumps(stage1_results, indent=2)}

Stage 2 Peer Critiques & Scores:
{json.dumps(stage2_reviews, indent=2)}

Masking Key used in Stage 2:
{json.dumps(label_mapping, indent=2)}

JSON Format Required:
{{
  "winning_response_label": "e.g., Response C (Model 3)",
  "selection_rationale": "Detailed explanation of why this plan was chosen as best based on peer scores and safety.",
  "critique_reconciliation": "How key weakness points or risks identified during Stage 2 peer review were resolved.",
  "final_consensus_diagnosis": "Final consolidated primary diagnosis",
  "final_consensus_plan": [
    "1. Surgical approach and sinus floor considerations",
    "2. Bone grafting material & membrane protocol",
    "3. Implant dimensions, primary stability, and insertion torque fallback rules",
    "4. Prosthodontic / occlusal design",
    "5. Periodontal biotype management & follow-up"
  ]
}}
"""
            messages = [{"role": "user", "content": stage3_prompt}]
            raw_chair_out = call_openrouter_with_fallback(chair_model, messages, api_key, max_tokens=2500)
            chair_decision = extract_json(raw_chair_out)

            st.success("👑 Council Chair Synthesis Complete!")
            
            st.info(f"🏆 **Winning Plan Selected:** {chair_decision.get('winning_response_label', 'N/A')}")
            
            st.markdown("### 📝 Chair Selection Rationale")
            st.write(chair_decision.get("selection_rationale", "N/A"))
            
            st.markdown("### ⚖️ Reconciliation of Peer Critiques")
            st.write(chair_decision.get("critique_reconciliation", "N/A"))
            
            st.markdown("### 🎯 Final Consensus Diagnosis")
            st.write(f"**{chair_decision.get('final_consensus_diagnosis', 'N/A')}**")
            
            st.markdown("### 📋 Final Consensus Management Plan")
            for step in chair_decision.get("final_consensus_plan", []):
                st.write(f"- {step}")

            with st.expander("Raw Chair Decision JSON"):
                st.json(chair_decision)

        except Exception as e:
            st.error(f"Stage 3 Chair Synthesis Error: {str(e)}")
