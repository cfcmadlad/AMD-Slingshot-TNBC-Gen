import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import glob

# --- Configuration ---
MODEL_ID  = "runwayml/stable-diffusion-v1-5"
CKPT_DIR  = "./checkpoints"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Page Setup ---
st.set_page_config(page_title="TNBC Pathology Slide Generator", layout="centered")
st.title("TNBC Pathology Slide Generator")
st.markdown(
    "Generate synthetic histopathology slides for **Triple-Negative Breast Cancer** "
    "using a LoRA fine-tuned Stable Diffusion model trained on PathVQA data."
)

# --- Checkpoint Detection ---
def find_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None
    final = os.path.join(CKPT_DIR, "lora_tnbc_final.pt")
    if os.path.exists(final):
        return final
    candidates = sorted(glob.glob(os.path.join(CKPT_DIR, "lora_tnbc_step*.pt")))
    return candidates[-1] if candidates else None

# --- Pipeline Loader ---
@st.cache_resource(show_spinner="Loading model. This may take up to 60 seconds on first run.")
def load_pipeline(ckpt_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False
    ).to(DEVICE)

    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        # Strip PEFT prefix if present
        cleaned = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
        pipe.unet.load_state_dict(cleaned, strict=False)
        status = f"LoRA checkpoint loaded: {os.path.basename(ckpt_path)}"
    else:
        status = "No LoRA checkpoint found. Running base Stable Diffusion model."

    return pipe, status

ckpt_path      = find_checkpoint()
pipe, ckpt_msg = load_pipeline(ckpt_path)

# --- Sidebar ---
st.sidebar.header("Settings")
st.sidebar.info(ckpt_msg)

num_steps      = st.sidebar.slider("Inference Steps",  20, 50,  30)
guidance_scale = st.sidebar.slider("Guidance Scale",   5.0, 15.0, 7.5, step=0.5)
num_images     = st.sidebar.selectbox("Images to Generate", [1, 2, 4], index=0)

# --- Language Presets ---
SAMPLE_PROMPTS = {
    "English": (
        "Histopathology slide of triple-negative breast cancer demonstrating "
        "high mitotic rate, geographic necrosis, and pushing border invasion."
    ),
    "Hindi":   "ट्रिपल-नेगेटिव स्तन कैंसर की हिस्टोपैथोलॉजी स्लाइड, उच्च माइटोटिक दर दर्शाती है।",
    "Telugu":  "అధిక మైటోటిక్ రేట్ చూపించే ట్రిపుల్-నెగటివ్ బ్రెస్ట్ క్యాన్సర్ హిస్టోపాథాలజీ స్లైడ్.",
    "Tamil":   "அதிக மைட்டோடிக் விகிதம் காட்டும் டிரிபிள்-நெகடிவ் மார்பக புற்றுநோய் ஸ்லைடு.",
}

# --- Prompt Input ---
st.subheader("Clinical Prompt")
col1, col2 = st.columns([1, 3])
with col1:
    language = st.selectbox("Language", list(SAMPLE_PROMPTS.keys()))

prompt = st.text_area(
    "Enter a clinical description:",
    value=SAMPLE_PROMPTS[language],
    height=110
)

neg_prompt = st.text_area(
    "Negative prompt (optional):",
    value="blurry, artifact, watermark, text, low quality",
    height=60
)

# --- Generation ---
if st.button("Generate Slide", use_container_width=True):
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    else:
        with st.spinner("Generating histopathology slide..."):
            with torch.inference_mode():
                results = pipe(
                    [prompt] * num_images,
                    negative_prompt=[neg_prompt] * num_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                ).images

        os.makedirs("outputs", exist_ok=True)
        cols = st.columns(num_images if num_images <= 2 else 2)

        for i, img in enumerate(results):
            save_path = f"outputs/generated_{i:02d}.png"
            img.save(save_path)
            with cols[i % len(cols)]:
                st.image(img, caption=f"Generated Slide {i + 1}", use_column_width=True)

        st.success(f"{num_images} image(s) saved to outputs/")

# --- Footer ---
st.markdown("---")
st.caption(
    "Model: Stable Diffusion v1.5 + TNBC LoRA  |  "
    "Dataset: PathVQA  |  "
    "For research use only."
)