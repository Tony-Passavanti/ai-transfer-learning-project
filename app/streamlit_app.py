"""Hammer Subtype Classifier — Transfer Learning Demo."""

import sys
from pathlib import Path

# Ensure project root is on the path so src imports work when run via streamlit
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights

from src.model import build_resnet18
from src.utils import CLASS_NAMES, get_device, get_eval_transform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MEDIA_DIR = _PROJECT_ROOT / "media"
MODELS_DIR = _PROJECT_ROOT / "models"

# Map filename prefixes to display names
_PREFIX_TO_LABEL = {
    "bp": "Ball-Peen Hammer",
    "ch": "Claw Hammer",
    "rm": "Rubber Mallet",
    "sh": "Sledge Hammer",
}

DISPLAY_NAMES = {name: label for name, label in zip(CLASS_NAMES, [
    "Ball-Peen Hammer", "Claw Hammer", "Rubber Mallet", "Sledge Hammer",
])}

# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load baseline (stock ImageNet ResNet18) and fine-tuned models."""
    device = get_device()
    finetuned_path = MODELS_DIR / "finetuned.pt"

    finetuned_ready = False

    # --- Baseline: completely unmodified pretrained ImageNet ResNet18 ---
    from torchvision import models
    baseline = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    baseline_labels = None  # signals ImageNet mode (1000 classes)

    # --- Fine-tuned ---
    if finetuned_path.exists():
        finetuned = build_resnet18(num_classes=len(CLASS_NAMES), freeze_backbone=False)
        finetuned.load_state_dict(torch.load(finetuned_path, map_location=device, weights_only=True))
        finetuned_labels = CLASS_NAMES
        finetuned_ready = True
    else:
        finetuned = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        finetuned_labels = None

    baseline.to(device).eval()
    finetuned.to(device).eval()

    return baseline, finetuned, baseline_labels, finetuned_labels, finetuned_ready, device


def _get_imagenet_class_name(idx: int) -> str:
    """Return the human-readable ImageNet class name for an index."""
    meta = ResNet18_Weights.DEFAULT.meta
    categories = meta.get("categories", None)
    if categories and idx < len(categories):
        return categories[idx]
    return f"class_{idx}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, image: Image.Image, labels, device):
    """Run a single image through a model and return top-3 predictions.

    Returns a list of (class_name, confidence) tuples sorted by confidence.
    """
    transform = get_eval_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

    top3_probs, top3_idxs = probs.topk(3)

    results = []
    for prob, idx in zip(top3_probs.tolist(), top3_idxs.tolist()):
        if labels is not None:
            name = DISPLAY_NAMES.get(labels[idx], labels[idx])
        else:
            name = _get_imagenet_class_name(idx)
        results.append((name, prob))

    return results


# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------

def _load_gallery_images():
    """Discover images in media/ and return sorted list of (path, display_name)."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for p in sorted(MEDIA_DIR.iterdir()):
        if p.suffix.lower() in extensions:
            # Derive display name from prefix
            prefix = "".join(c for c in p.stem if c.isalpha())
            display = _PREFIX_TO_LABEL.get(prefix, p.stem)
            images.append((p, display))
    return images


def _make_square_thumbnail(img: Image.Image, size: int = 150) -> Image.Image:
    """Center-crop an image to a square and resize to a uniform thumbnail."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Hammer Classifier Demo",
        page_icon="",
        layout="wide",
    )

    st.title("Hammer Subtype Classifier")
    st.markdown(
        "Compare a **baseline** model (stock ImageNet-pretrained ResNet18 "
        "with no hammer-specific training) against a **fine-tuned** model "
        "(all layers trained on hammer images). The baseline has never seen "
        "hammer subtypes and produces essentially random predictions, while "
        "the fine-tuned model learns the subtle differences between them."
    )

    # Load models
    baseline, finetuned, bl_labels, ft_labels, finetuned_ready, device = load_models()

    if not finetuned_ready:
        st.warning(
            "Fine-tuned model weights are missing. Place `finetuned.pt` in "
            "`models/` after training to see real fine-tuned results."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Image gallery
    # ------------------------------------------------------------------
    st.subheader("Select a Sample Image")

    gallery = _load_gallery_images()
    if not gallery:
        st.error(f"No images found in `{MEDIA_DIR}`. Add sample images to continue.")
        return

    # Display thumbnails in a compact, aligned grid
    # Use padding columns on each side to keep thumbnails small and centered
    cols_per_row = 4

    # Use session state to track selection
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None

    for row_start in range(0, len(gallery), cols_per_row):
        row_items = gallery[row_start : row_start + cols_per_row]
        # Pad with empty slots so every row has the same number of columns
        cols = st.columns(cols_per_row)
        for col, (img_path, display_name) in zip(cols, row_items):
            with col:
                img = Image.open(img_path)
                thumb = _make_square_thumbnail(img, size=150)
                st.image(thumb, use_container_width=True)
                if st.button(display_name, key=str(img_path), use_container_width=True):
                    st.session_state.selected_image = str(img_path)

    selected_path = st.session_state.selected_image

    # ------------------------------------------------------------------
    # Inference results
    # ------------------------------------------------------------------
    if selected_path is not None:
        st.divider()
        st.subheader("Prediction Results")

        image = Image.open(selected_path)

        # Show the selected image centered
        sel_col1, sel_col2, sel_col3 = st.columns([1, 2, 1])
        with sel_col2:
            st.image(image, caption="Selected Image", use_container_width=True)

        # Run inference on both models
        baseline_results = run_inference(baseline, image, bl_labels, device)
        finetuned_results = run_inference(finetuned, image, ft_labels, device)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Baseline Model (Stock ImageNet ResNet18)")
            _render_predictions(baseline_results)

        with col_right:
            st.markdown("#### Fine-Tuned Model (Trained on Hammer Images)")
            _render_predictions(finetuned_results)

    # ------------------------------------------------------------------
    # About section
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("About this demo"):
        st.markdown("""
**What is transfer learning?**

Instead of training a neural network from scratch, we start with a
model (ResNet18) that was already trained on 1.2 million images from
ImageNet. This model has already learned to recognize edges, textures,
and shapes — general visual features that transfer well to new tasks.

**Baseline model** — A stock ResNet18 with its original ImageNet-pretrained
weights and a randomly initialized 4-class classifier head. It has never
been trained on hammer images, so its predictions across the four hammer
subtypes are essentially random. This shows what a pretrained model looks
like *before* any task-specific training.

**Fine-tuned model** — We start from the same pretrained ResNet18 but
train all layers on our hammer dataset. This lets the network adapt its
learned features to the specific visual differences between hammer
subtypes — handle shape, head geometry, material texture — resulting in
accurate classification.
""")


def _render_predictions(results):
    """Render top-3 predictions with confidence bars."""
    if not results:
        st.write("No predictions available.")
        return

    top_name, top_conf = results[0]
    st.metric(label="Prediction", value=top_name, delta=f"{top_conf:.1%} confidence")

    st.markdown("**Top 3 Predictions:**")
    for name, conf in results:
        col_name, col_bar = st.columns([2, 3])
        with col_name:
            st.write(name)
        with col_bar:
            st.progress(conf, text=f"{conf:.1%}")


if __name__ == "__main__":
    main()
