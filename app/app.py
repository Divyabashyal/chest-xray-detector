import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(
        torch.load(
            'model/best_model.pth',
            map_location=torch.device('cpu')
        )
    )
    model.eval()
    return model

model = load_model()
CLASS_NAMES = ['Normal', 'Pneumonia']

# ── Image transform ───────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── GradCAM ───────────────────────────────────────────────────
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)

    image_tensor = image_tensor.clone().requires_grad_(True)
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    handle_f.remove()
    handle_b.remove()

    grad = gradients[0].detach().cpu().numpy()[0]
    act  = activations[0].detach().cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    return cam, pred_class

# ── UI ────────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Pneumonia Detector")
st.markdown(
    "Upload a chest X-ray and the AI will detect whether it shows "
    "**Normal** lungs or **Pneumonia**."
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-Ray")
        st.image(image, use_column_width=True)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = probabilities.max(0)

    pred_label     = CLASS_NAMES[predicted.item()]
    confidence_pct = confidence.item() * 100

    # GradCAM
    cam, _ = generate_gradcam(model, image_tensor)

    # Overlay heatmap
    img_array   = np.array(image.resize((224, 224)))
    cam_resized = np.uint8(255 * cam)
    cam_resized = np.array(Image.fromarray(cam_resized).resize((224, 224)))
    heatmap     = cm.jet(cam_resized / 255.0)[:, :, :3]
    overlay     = (0.6 * img_array / 255.0 + 0.4 * heatmap)
    overlay     = np.clip(overlay, 0, 1)

    with col2:
        st.subheader("AI Attention Map")
        st.image(overlay, use_column_width=True)
        st.caption("🔴 Red areas = where the AI focused most")

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred_label == "Pneumonia":
        st.error(f"🔴 **{pred_label}** detected — Confidence: {confidence_pct:.1f}%")
    else:
        st.success(f"🟢 **{pred_label}** — Confidence: {confidence_pct:.1f}%")

    st.subheader("Confidence Breakdown")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name}**")
        st.progress(float(probabilities[i].item()))
        st.write(f"{probabilities[i].item()*100:.1f}%")

    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational purposes only "
        "and is not a substitute for medical diagnosis."
    )