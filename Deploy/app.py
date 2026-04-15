import importlib.util
import io
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Deja estas rutas vacias por ahora. Luego podras poner tus rutas reales.
DEFAULT_CUSTOM_MODEL_PY = ""
DEFAULT_CUSTOM_WEIGHTS_PTH = ""
DEFAULT_CUSTOM_CLASS_NAME = ""
DEFAULT_CUSTOM_BUILDER_NAME = ""


st.set_page_config(page_title="Segmentacion Cacao", layout="wide")

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

    .stSelectbox label,
    .stFileUploader label,
    .stRadio label,
    .stNumberInput label,
    .stSlider label,
    .stTextInput label {
        color: #000000 !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stSlider,
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

    .section-card {
        background: #ffffff;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }

    .hint-box {
        background: var(--green-soft);
        border: 1px solid #bde8c7;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }

    .stRadio > div {
        background: #ffffff;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 0.4rem 0.8rem;
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


def pil_to_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    rgb = image.convert("RGB").resize((image_size, image_size))
    arr = np.array(rgb).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor.to(device)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    return Image.fromarray(mask_uint8, mode="L")


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.array(image.convert("RGB"))
    mask_resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize((base.shape[1], base.shape[0])))

    overlay = base.copy()
    red = np.zeros_like(base)
    red[..., 0] = 255

    mask_3d = np.stack([mask_resized] * 3, axis=-1).astype(bool)
    overlay[mask_3d] = (alpha * red[mask_3d] + (1 - alpha) * base[mask_3d]).astype(np.uint8)
    return Image.fromarray(overlay)


def dynamic_import_module(model_py_path: Path) -> object:
    module_name = model_py_path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(model_py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el archivo .py del modelo.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint_state_dict(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(str(ckpt_path), map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint

    raise RuntimeError("Checkpoint no compatible. Se esperaba un state_dict o dict con state_dict.")


def build_custom_model(module, class_name: str, builder_name: str):
    if builder_name:
        if not hasattr(module, builder_name):
            raise RuntimeError(f"La funcion '{builder_name}' no existe en el archivo .py.")
        builder = getattr(module, builder_name)
        model = builder()
        return model

    if class_name:
        if not hasattr(module, class_name):
            raise RuntimeError(f"La clase '{class_name}' no existe en el archivo .py.")
        cls = getattr(module, class_name)
        model = cls()
        return model

    raise RuntimeError("Debes indicar nombre de clase o funcion constructora para crear el modelo.")


def resolve_local_path(path_value: str) -> Path:
    raw = Path(path_value.strip()).expanduser()
    if raw.is_absolute():
        return raw
    return (APP_DIR / raw).resolve()


def predict_custom_segmentation(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    threshold: float,
    damaged_class: int,
    device: torch.device,
) -> np.ndarray:
    x = pil_to_tensor(image, image_size=image_size, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    if isinstance(out, dict):
        if "out" in out:
            out = out["out"]
        else:
            first_key = next(iter(out.keys()))
            out = out[first_key]

    if isinstance(out, (list, tuple)):
        out = out[0]

    if out.ndim == 4:
        out = out[0]

    if out.ndim == 3:
        channels = out.shape[0]
        if channels == 1:
            prob = torch.sigmoid(out[0])
            mask = (prob > threshold).cpu().numpy().astype(np.uint8)
            return mask

        pred_class = torch.argmax(out, dim=0)
        mask = (pred_class == damaged_class).cpu().numpy().astype(np.uint8)
        return mask

    if out.ndim == 2:
        prob = torch.sigmoid(out)
        mask = (prob > threshold).cpu().numpy().astype(np.uint8)
        return mask

    raise RuntimeError("Salida del modelo no compatible para segmentacion.")


def predict_yolo_boxes(model: YOLO, image: Image.Image) -> tuple[Image.Image, int]:
    results = model.predict(image, verbose=False)
    if not results:
        raise RuntimeError("No se obtuvieron resultados de YOLO.")

    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        return image.copy(), 0

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
    names = result.names if hasattr(result, "names") else {}

    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(46, 125, 50), width=3)

        cls_text = "obj"
        if classes is not None:
            cls_id = int(classes[idx])
            cls_text = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        conf_text = ""
        if confs is not None:
            conf_text = f" {confs[idx]:.2f}"

        label = f"{cls_text}{conf_text}"
        text_pos = (x1 + 4, max(2, y1 - 16))
        draw.text(text_pos, label, fill=(0, 0, 0))

    return annotated, len(boxes_xyxy)


@st.cache_resource
def load_default_yolo_model(model_name: str, models_dir: str) -> YOLO:
    target_path = Path(models_dir) / model_name
    if not target_path.exists():
        temp_model = YOLO(model_name)
        source_path = Path(getattr(temp_model, "ckpt_path", model_name))
        if source_path.exists() and source_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
    return YOLO(str(target_path if target_path.exists() else model_name))


st.markdown(
    """
    <div class="hero-box">
        <h1 style="margin:0;">Segmentacion de cacao</h1>
        <p style="margin:0.45rem 0 0 0; font-size:1.03rem;">
            Usa deteccion YOLO con cajas o tu modelo personalizado por rutas locales.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Tipo de modelo",
    options=["YOLO Deteccion (Ultralytics)", "PyTorch Personalizado (.py + .pth)"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.caption(f"Dispositivo detectado: {device}")

uploaded_images = st.file_uploader(
    "Sube una o varias imagenes",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

st.markdown(
    """
    <div class="hint-box">
        Tip: para mejores resultados usa imagenes enfocadas, con buena iluminacion y del area de interes bien visible.
    </div>
    """,
    unsafe_allow_html=True,
)

if mode == "YOLO Deteccion (Ultralytics)":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    default_model_name = st.selectbox(
        "Modelo YOLO de deteccion",
        options=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        index=0,
        help="Se descarga automaticamente en Deploy/models la primera vez.",
    )
    selected_weights_path = MODELS_DIR / default_model_name
    st.caption(f"Ruta de pesos YOLO: {selected_weights_path}")
    st.caption("En esta opcion solo subes imagenes. El resultado se muestra con bounding boxes.")
    st.markdown("</div>", unsafe_allow_html=True)

    run_yolo = st.button("Ejecutar inferencia YOLO", type="primary", disabled=not uploaded_images)

    if run_yolo:
        try:
            with st.spinner("Cargando modelo YOLO (primer uso puede tardar por descarga)..."):
                model = load_default_yolo_model(default_model_name, str(MODELS_DIR))

            for img_file in uploaded_images:
                image = Image.open(img_file).convert("RGB")
                boxed_image, total_boxes = predict_yolo_boxes(model, image)

                st.subheader(f"Resultado: {img_file.name}")
                c1, c2 = st.columns(2)
                c1.image(image, caption="Original", use_container_width=True)
                c2.image(boxed_image, caption=f"Deteccion (cajas): {total_boxes}", use_container_width=True)

                png_buffer = io.BytesIO()
                boxed_image.save(png_buffer, format="PNG")
                st.download_button(
                    label=f"Descargar imagen con cajas - {img_file.name}",
                    data=png_buffer.getvalue(),
                    file_name=f"boxed_{Path(img_file.name).stem}.png",
                    mime="image/png",
                )

        except Exception as exc:
            st.error(f"Error en inferencia YOLO: {exc}")

else:
    st.markdown("### Configuracion de modelo personalizado por rutas locales")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    model_py_path_text = st.text_input(
        "Ruta local al archivo del modelo (.py)",
        value=DEFAULT_CUSTOM_MODEL_PY,
        placeholder="Ejemplo: C:/ruta/al/modelo.py",
    )
    weights_pth_path_text = st.text_input(
        "Ruta local al archivo de pesos (.pth)",
        value=DEFAULT_CUSTOM_WEIGHTS_PTH,
        placeholder="Ejemplo: C:/ruta/al/pesos.pth",
    )

    class_name = st.text_input(
        "Nombre de clase del modelo (opcional si usas funcion constructora)",
        value=DEFAULT_CUSTOM_CLASS_NAME,
    )
    builder_name = st.text_input(
        "Nombre de funcion constructora (ejemplo: build_model)",
        value=DEFAULT_CUSTOM_BUILDER_NAME,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    image_size = col_a.number_input("Tamano de entrada", min_value=64, max_value=2048, value=512, step=32)
    threshold = col_b.slider("Umbral binario", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
    damaged_class = col_c.number_input("Clase danada", min_value=0, max_value=20, value=1, step=1)

    custom_paths_ready = bool(model_py_path_text.strip() and weights_pth_path_text.strip())

    if not custom_paths_ready:
        st.info(
            "Completa las rutas locales del modelo (.py) y pesos (.pth). "
            "Por ahora puedes dejarlas vacias hasta que tengas esos archivos."
        )

    run_custom = st.button(
        "Ejecutar inferencia personalizada",
        type="primary",
        disabled=not (custom_paths_ready and uploaded_images),
    )

    if run_custom:
        try:
            model_py_path = resolve_local_path(model_py_path_text)
            weights_pth_path = resolve_local_path(weights_pth_path_text)

            if not model_py_path.exists():
                raise RuntimeError(f"No existe el archivo de modelo: {model_py_path}")
            if not weights_pth_path.exists():
                raise RuntimeError(f"No existe el archivo de pesos: {weights_pth_path}")

            module = dynamic_import_module(model_py_path)
            model = build_custom_model(module, class_name=class_name.strip(), builder_name=builder_name.strip())

            state_dict = load_checkpoint_state_dict(weights_pth_path, device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            for img_file in uploaded_images:
                image = Image.open(img_file).convert("RGB")
                mask = predict_custom_segmentation(
                    model=model,
                    image=image,
                    image_size=int(image_size),
                    threshold=float(threshold),
                    damaged_class=int(damaged_class),
                    device=device,
                )

                mask_img = mask_to_image(mask)
                overlay = overlay_mask(image, mask)

                st.subheader(f"Resultado: {img_file.name}")
                c1, c2, c3 = st.columns(3)
                c1.image(image, caption="Original", use_container_width=True)
                c2.image(mask_img, caption="Mascara", use_container_width=True)
                c3.image(overlay, caption="Superposicion", use_container_width=True)

                png_buffer = io.BytesIO()
                mask_img.save(png_buffer, format="PNG")
                st.download_button(
                    label=f"Descargar mascara - {img_file.name}",
                    data=png_buffer.getvalue(),
                    file_name=f"mask_{Path(img_file.name).stem}.png",
                    mime="image/png",
                )

        except Exception as exc:
            st.error(f"Error en inferencia personalizada: {exc}")

st.markdown("---")
st.caption("Siguiente paso: cuando tengas tu modelo final, coloca las rutas en la seccion personalizada y ejecuta inferencia.")
