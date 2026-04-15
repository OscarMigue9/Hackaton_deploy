import io
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
TRAINED_ONNX_PATH = ROOT_DIR / "models" / "best.onnx"

st.set_page_config(page_title="Deteccion Cacao", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-main: #f6fff8;
        --bg-card: #ffffff;
        --green-soft: #dff7e3;
        --green-main: #2e7d32;
        --green-main-dark: #1f5f24;
        --text-main: #000000;
        --border-soft: #ccefd2;
    }

    .stApp {
        background: radial-gradient(circle at top right, #ebffe8 0%, var(--bg-main) 45%, #ffffff 100%);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #000000;
    }

    p, span, label, li, div {
        color: #000000;
    }

    .stMarkdown, .stText, .stCaption {
        color: #000000;
    }

    .stFileUploader label,
    .stButton label {
        color: #000000 !important;
    }

    .stFileUploader {
        color: #000000 !important;
    }

    .hero-box {
        background: linear-gradient(120deg, var(--bg-card) 0%, #f4fff3 100%);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(30, 90, 40, 0.08);
    }

    .hint-box {
        background: var(--green-soft);
        border: 1px solid #bde8c7;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }

    .stFileUploader {
        background: #ffffff;
        border: 1px dashed #9ed8a8;
        border-radius: 12px;
        padding: 0.3rem;
    }

    .stButton > button {
        background-color: var(--green-main);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.45rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: var(--green-main-dark);
    }

    .stDownloadButton > button {
        background-color: #ffffff;
        color: var(--green-main-dark);
        border: 1px solid #98d9a4;
        border-radius: 10px;
    }

    .stCaption {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_yolo_inference(model: YOLO, image: Image.Image) -> Image.Image:
    results = model.predict(image, verbose=False, conf=0.25, iou=0.6)
    if not results:
        raise RuntimeError("No se obtuvieron resultados de YOLO.")

    result = results[0]
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]
    return Image.fromarray(plotted_rgb)


@st.cache_resource
def load_trained_yolo_model(model_path: str) -> YOLO:
    return YOLO(model_path, task="detect")


st.markdown(
    """
    <div class="hero-box">
        <h1 style="margin:0;">Deteccion de granos de cacao</h1>
        <p style="margin:0.45rem 0 0 0; font-size:1.03rem;">
            Interfaz simplificada: solo sube imagenes y el modelo ONNX fine-tuned hace la inferencia.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Backend: ONNX (best.onnx)")
st.caption(f"Modelo en uso: {TRAINED_ONNX_PATH}")

uploaded_images = st.file_uploader(
    "Sube una o varias imagenes",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

st.markdown(
    """
    <div class="hint-box">
        Tip: usa imagenes enfocadas y con buena iluminacion para mejorar la deteccion.
    </div>
    """,
    unsafe_allow_html=True,
)

run_inference = st.button("Ejecutar inferencia", type="primary", disabled=not uploaded_images)

if run_inference:
    try:
        if not TRAINED_ONNX_PATH.exists():
            raise RuntimeError(f"No se encontro el modelo ONNX en: {TRAINED_ONNX_PATH}")

        with st.spinner("Cargando modelo ONNX..."):
            model = load_trained_yolo_model(str(TRAINED_ONNX_PATH))

        for img_file in uploaded_images:
            image = Image.open(img_file).convert("RGB")
            boxed_image = run_yolo_inference(model, image)

            st.subheader(f"Resultado: {img_file.name}")
            c1, c2 = st.columns(2)
            c1.image(image, caption="Original", use_container_width=True)
            c2.image(boxed_image, caption="Inferencia del modelo", use_container_width=True)

            png_buffer = io.BytesIO()
            boxed_image.save(png_buffer, format="PNG")
            st.download_button(
                label=f"Descargar imagen con cajas - {img_file.name}",
                data=png_buffer.getvalue(),
                file_name=f"boxed_{Path(img_file.name).stem}.png",
                mime="image/png",
            )

    except Exception as exc:
        st.error(f"Error en inferencia: {exc}")

st.markdown("---")
st.caption("Flujo simplificado: solo subes imagenes y el modelo best.onnx hace la deteccion automaticamente.")
