import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, mobilenet_v2
from PIL import Image
import numpy as np
import base64
from io import BytesIO

CLASSES = ['nablus', 'bethlehem', 'jaffa']
IMG_SIZE = 224
DEVICE = torch.device('cpu')

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="🇵🇸 Palestinian Thobe Classifier", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;1,400&display=swap');

html, body, [class*="css"] {
    background-color: #5C1A1A !important;
}

.stApp {
    background: #5C1A1A;
}

/* Animated background particles */
@keyframes floatUp {
    0%   { transform: translateY(0) scale(1); opacity: 0.5; }
    100% { transform: translateY(-800px) scale(0.2); opacity: 0; }
}

.title-box {
    text-align: center;
    padding: 1.5rem 0 1rem;
}

.title-main {
    font-family: 'Amiri', serif;
    font-size: 2.6rem;
    font-style: italic;
    color: #F5E6C8;
    letter-spacing: 1px;
    animation: glow 3s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { color: #F5E6C8; text-shadow: 0 0 10px rgba(212,175,130,0.2); }
    50%       { color: #FFD9A0; text-shadow: 0 0 20px rgba(212,175,130,0.5); }
}

.title-sub {
    font-size: 0.75rem;
    color: rgba(212,175,130,0.7);
    letter-spacing: 3px;
    margin-top: 4px;
}

.thobe-border {
    width: 100%;
    text-align: center;
    font-size: 1.1rem;
    color: #D4AF82;
    letter-spacing: 4px;
    margin: 0.5rem 0 1.5rem;
    animation: borderPulse 2.5s ease-in-out infinite;
}

@keyframes borderPulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
}

/* Cards */
div[data-testid="stSelectbox"] label,
div[data-testid="stFileUploader"] label {
    color: rgba(212,175,130,0.85) !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(92,26,26,0.6) !important;
    border: 0.5px solid rgba(212,175,130,0.4) !important;
    color: #F5E6C8 !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] > div {
    background: rgba(92,26,26,0.4) !important;
    border: 1px dashed rgba(212,175,130,0.4) !important;
    border-radius: 10px !important;
    color: #D4AF82 !important;
}

div[data-testid="stFileUploader"] p {
    color: rgba(212,175,130,0.7) !important;
}

.stButton > button {
    width: 100%;
    background: #8B2E2E !important;
    color: #F5E6C8 !important;
    border: 0.5px solid rgba(212,175,130,0.5) !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    padding: 0.6rem !important;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #A33535 !important;
    transform: translateY(-1px);
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #8B2E2E, #D4AF82) !important;
}

.stProgress > div > div {
    background: rgba(255,255,255,0.08) !important;
}

/* Success box */
.result-card {
    background: rgba(255,255,255,0.06);
    border: 0.5px solid rgba(212,175,130,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-class {
    font-family: 'Amiri', serif;
    font-size: 2rem;
    font-style: italic;
    color: #FFD9A0;
}

.result-conf {
    font-size: 0.85rem;
    color: rgba(212,175,130,0.75);
    margin-top: 4px;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    color: rgba(212,175,130,0.6);
    letter-spacing: 2px;
    margin: 1rem 0 0.3rem;
}

/* Image preview */
div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 0.5px solid rgba(212,175,130,0.3);
}

/* Bottom tatreez bar */
.tatreez-bar {
    text-align: center;
    font-size: 1.3rem;
    color: #D4AF82;
    letter-spacing: 6px;
    margin-top: 2rem;
    padding: 0.8rem 0;
    border-top: 0.5px solid rgba(212,175,130,0.2);
    animation: borderPulse 3s ease-in-out infinite;
}

/* Particles canvas */
#particles-js { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }

/* All text */
p, div, span, label { color: #D4AF82; }
h1, h2, h3 { color: #F5E6C8 !important; }

/* Streamlit override */
section[data-testid="stSidebar"] { display: none; }
</style>

<div class="title-box">
    <div class="title-main">🇵🇸 Palestinian Thobe Classifier</div>
    <div class="title-sub">NABLUS · BETHLEHEM · JAFFA</div>
</div>
<div class="thobe-border">◆ ✦ ◇ ✧ ◈ ✦ ◆ ◇ ✧ ◈ ✦ ◆ ◇ ✧ ◈ ✦ ◆</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name):
    if model_name == "EfficientNetB0":
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 3),
        )
        model.load_state_dict(torch.load("thobe_efficientnet_best.pt", map_location=DEVICE))
    else:
        model = mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 3),
        )
        model.load_state_dict(torch.load("thobe_mobilenet_best.pt", map_location=DEVICE))
    model.eval()
    return model


def predict(model, image):
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    return probs.numpy()


model_choice = st.selectbox("اختاري الموديل", ["EfficientNetB0", "MobileNetV2"])
uploaded_file = st.file_uploader("ارفعي صورة الثوب", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("جاري تحليل الثوب..."):
        model = load_model(model_choice)
        probs = predict(model, image)

    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    region_ar = {'nablus': 'نابلس', 'bethlehem': 'بيت لحم', 'jaffa': 'يافا'}

    st.markdown(f"""
    <div class="result-card">
        <div class="result-class">{region_ar[pred_class]}</div>
        <div class="result-conf">نسبة الثقة: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">توزيع الاحتمالات</div>', unsafe_allow_html=True)
    for cls, prob in zip(CLASSES, probs):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.markdown(f"<span style='color:#F5E6C8; font-size:13px;'>{region_ar[cls]} {prob*100:.1f}%</span>",
                        unsafe_allow_html=True)

st.markdown('<div class="tatreez-bar">◆ ✦ ✧ ◈ ✦ ◆ ✧ ◈ ✦ ◆ ◇ ✧ ◈</div>', unsafe_allow_html=True)