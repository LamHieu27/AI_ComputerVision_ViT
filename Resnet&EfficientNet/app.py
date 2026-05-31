import time
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_eff
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_res
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# ─── Paths 
BASE_DIR = Path(__file__).parent
CSS_FILE  = BASE_DIR / "style.css"
HTML_FILE = BASE_DIR / "html_deep.html"

CSS = CSS_FILE.read_text(encoding="utf-8")

def _parse_html_blocks(path: Path):
    blocks = {}
    current_key = None
    current_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("<!-- ── BLOCK:"):
            if current_key:
                blocks[current_key] = "\n".join(current_lines).strip()
            current_key = line.strip().removeprefix("<!-- ── BLOCK:").removesuffix("── -->").strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key:
        blocks[current_key] = "\n".join(current_lines).strip()

    return blocks

HTML = _parse_html_blocks(HTML_FILE)

# ─── Load models ───────────────────────────────────
print("Đang tải mô hình...")
model_eff = EfficientNetV2B0(weights="imagenet")
model_res = ResNet50(weights="imagenet")
print("✅Tải xong!")

# ─── Utils 
def safe_convert_rgb(img: Image.Image):
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def run_model(img_array, model, preprocess_fn):
    x = preprocess_fn(img_array.copy())
    t0 = time.perf_counter()
    preds = model.predict(x, verbose=0)
    elapsed = (time.perf_counter() - t0) * 1000
    results = decode_predictions(preds, top=3)[0]
    return results, elapsed


def fmt_preds(preds, ms, model_name):
    lines = [f"### {"Thời gian chờ:"}  `{ms:.0f} ms`\n"]
    ranks = ["🥇", "🥈", "🥉"]

    for i, (_, label, prob) in enumerate(preds):

        lines.append(f"{ranks[i]} **{label.replace('_', ' ')}** — `{prob*100:.2f}%`\n")
        bar = f"""
<div style="background:#374151; border-radius:6px; height:8px; width:100%; margin:6px 0;">
    <div style="
        width:{max(prob*100,5):.1f}%;
        background:{'#3b82f6' if model_name=='EfficientNetV2B0' else '#8b5cf6'};
        height:100%;
        border-radius:6px;
        transition: width 0.5s ease;">
    </div>
</div>
"""
        lines.append(bar)

    return "\n".join(lines)

# ─── Predict
def predict(img):
    if img is None:
        msg = "⚠️ Vui lòng upload ảnh!"
        return msg, msg, "", gr.update(visible=False)

    img = safe_convert_rgb(img).resize((224, 224))
    x = np.expand_dims(np.array(img), axis=0).astype(np.float32)

    eff_preds, eff_ms = run_model(x, model_eff, preprocess_eff)
    res_preds, res_ms = run_model(x, model_res, preprocess_res)

    eff_md = fmt_preds(eff_preds, eff_ms, "EfficientNetV2B0")
    res_md = fmt_preds(res_preds, res_ms, "ResNet50")

    eff_conf = eff_preds[0][2]
    res_conf = res_preds[0][2]

    faster = "EfficientNetV2B0" if eff_ms < res_ms else "ResNet50"
    higher = "EfficientNetV2B0" if eff_conf > res_conf else "ResNet50"

    analysis = f"""
    <div style="display:flex; gap:20px; align-items:flex-start;">
        <div style="flex:2;">
            <table style="width:100%; border-collapse:collapse; color: #e2e8f0;">
                <tr><th style="border:1px solid #475569; padding:8px; background:#1f2937;">Tiêu chí</th>
                    <th style="border:1px solid #475569; padding:8px; background:#1f2937;">EfficientNetV2B0</th>
                    <th style="border:1px solid #475569; padding:8px; background:#1f2937;">ResNet50</th>
                </tr>
                <tr>
                    <td style="border:1px solid #475569; padding:8px;">Khả năng cao</td>
                    <td style="border:1px solid #475569; padding:8px;">{eff_preds[0][1]}</td>
                    <td style="border:1px solid #475569; padding:8px;">{res_preds[0][1]}</td>
                </tr>
                <tr>
                    <td style="border:1px solid #475569; padding:8px;">Độ tin cậy</td>
                    <td style="border:1px solid #475569; padding:8px;">{eff_conf*100:.2f}%</td>
                    <td style="border:1px solid #475569; padding:8px;">{res_conf*100:.2f}%</td>
                </tr>
                <tr>
                    <td style="border:1px solid #475569; padding:8px;">Thời gian</td>
                    <td style="border:1px solid #475569; padding:8px;">{eff_ms:.0f} ms</td>
                    <td style="border:1px solid #475569; padding:8px;">{res_ms:.0f} ms</td>
                </tr>
            </table>
        </div>

        <div style="flex:1;">
            <div style="background:#1f2937; color:white; padding:12px; border-radius:8px;">
                <p style="text-align:center; font-weight:bold; color:#facc15">Đánh giá</p>
                <div style="color:green"><b>Nhanh hơn:</b> {faster}</div>
                <div style="margin-top:8px;color:green"><b>Chính xác hơn:</b> {higher}</div>
            </div>
        </div>
    </div>
    """

    return eff_md, res_md, analysis, gr.update(visible=True)

# ─── UI 
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.HTML(HTML["header"])

    with gr.Row():
        with gr.Column(scale=0.6):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Upload ảnh", elem_classes="upload-title")
                image_input = gr.Image(type="pil", height=220, show_label=False)
                gr.HTML(HTML["info_box"])
                btn = gr.Button("Phân tích", elem_classes="predict-btn")

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.HTML(HTML["result_label_eff"])
                    eff_output = gr.Markdown()
                with gr.Column():
                    gr.HTML(HTML["result_label_res"])
                    res_output = gr.Markdown()
            divider = gr.HTML(HTML["section_divider"], visible=False)
            analysis_output = gr.HTML()
    btn.click(
    fn=predict,
    inputs=image_input,
    outputs=[eff_output, res_output, analysis_output, divider]
)
if __name__ == "__main__":
    demo.queue(False).launch(css=CSS)