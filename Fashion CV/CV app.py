import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# --- PAGE SETUP ---
st.set_page_config(
    page_title="HAUTE.AI | Autonomous Fashion Atelier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LUXURY VIBRANT FASHION THEME (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Editorial Fashion Header */
    .fashion-hero {
        background: linear-gradient(135deg, #FF007A 0%, #7928CA 50%, #4338CA 100%);
        padding: 36px 40px;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 15px 35px -5px rgba(121, 40, 202, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .brand-title {
        font-family: 'Cinzel', serif;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    .brand-tagline {
        font-size: 1.05rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        opacity: 0.95;
        margin-top: 8px;
    }
    
    /* Occasion & Style Personality Banner */
    .style-vibe-card {
        background: linear-gradient(120deg, #10B981 0%, #059669 100%);
        border-radius: 16px;
        padding: 22px 26px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
    }
    .vibe-tag {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        background: rgba(255,255,255,0.2);
        display: inline-block;
        padding: 4px 12px;
        border-radius: 30px;
        margin-bottom: 8px;
    }
    .vibe-title {
        font-family: 'Cinzel', serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: 0.03em;
    }
    .vibe-desc {
        font-size: 0.95rem;
        opacity: 0.95;
        line-height: 1.45;
    }

    /* Vibrant Feature Cards */
    .fashion-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .fashion-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.1);
    }
    
    .card-label-sub { color: #8B5CF6; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
    .card-label-art { color: #EC4899; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
    .card-label-col { color: #F59E0B; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
    
    .card-title-lg { font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 4px 0; }
    
    /* Taxonomy Badges */
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .chip-purple { background: #F3E8FF; color: #7E22CE; border: 1px solid #E9D5FF; }
    .chip-pink { background: #FCE7F3; color: #BE185D; border: 1px solid #FBCFE8; }
    .chip-amber { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    
    /* Color Swatch Pill */
    .palette-pill {
        display: inline-flex;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 6px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
        color: #1E293B;
        margin-right: 10px;
        margin-bottom: 8px;
    }
    .swatch-circle {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        margin-right: 8px;
        border: 2px solid #ffffff;
        box-shadow: 0 0 4px rgba(0,0,0,0.25);
    }

    /* Outfit Styling Block */
    .outfit-pairing-row {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-left: 5px solid #6366F1;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL DEFINITION ---
class MultiHeadFashionNet(nn.Module):
    def __init__(self, num_subcats, num_articles, num_colors):
        super().__init__()
        base_model = models.mobilenet_v3_large(weights=None)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        in_dim = 960
        self.shared_fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(p=0.25)
        )
        self.head_subcat = nn.Linear(512, num_subcats)
        self.head_article = nn.Linear(512, num_articles)
        self.head_color = nn.Linear(512, num_colors)

    def forward(self, x):
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        flat = torch.flatten(pooled, 1)
        feat = self.shared_fc(flat)
        return {
            "subCategory": self.head_subcat(feat),
            "articleType": self.head_article(feat),
            "baseColour": self.head_color(feat),
            "feat_map": feat_map
        }

# --- ASPECT RATIO PRESERVING LETTERBOX ---
class LetterboxResize:
    def __init__(self, target_size=(224, 224), fill_color=(255, 255, 255)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img):
        target_w, target_h = self.target_size
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        delta_w, delta_h = target_w - new_w, target_h - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(resized_img, padding, fill=self.fill_color)

# --- OCCASION & STYLE PERSONA ENGINE ---
def detect_occasion_and_vibe(subcat, article, colour):
    sub = subcat.lower()
    art = article.lower()
    col = colour.lower()
    
    # 1. Ethnic / Traditional
    if any(k in art for k in ["saree", "kurta", "kurtis", "lehenga", "nehru", "churidar", "dupatta", "sherwani"]) or "saree" in sub:
        return {
            "archetype": "Ethnic & Festive Atelier",
            "tag": "✨ Ceremonial & Festive Couture",
            "desc": "Timeless traditional elegance tailored for weddings, festive galas, religious ceremonies, and celebratory evening occasions.",
            "setting": "Weddings, Festivals & Receptions",
            "gradient": "linear-gradient(120deg, #F59E0B 0%, #D97706 100%)"
        }
        
    # 2. Formal / Corporate Executive
    if any(k in art for k in ["formal", "blazer", "suit", "tuxedo", "tie", "cufflinks", "waistcoat"]) or (art == "shirts" and any(c in col for c in ["white", "blue", "light blue", "black", "grey"])):
        return {
            "archetype": "Executive & Formal Business",
            "tag": "💼 Boardroom & Corporate Elite",
            "desc": "Sharp, structured silhouette projecting authority and professional confidence. Ideal for executive meetings, interviews, and corporate summits.",
            "setting": "Corporate Headquarters, Presentations & Conferences",
            "gradient": "linear-gradient(120deg, #3B82F6 0%, #1D4ED8 100%)"
        }
        
    # 3. Smart Casual / Modern Prep
    if art in ["shirts", "trousers", "chinos", "sweaters", "polo"] or (art == "dresses" and "casual" not in art):
        return {
            "archetype": "Smart Casual / Urban Chic",
            "tag": "🍸 Polished Day-To-Night",
            "desc": "Effortless sartorial balance between relaxed comfort and refined sophistication. Perfect for dinners, weekend pop-ups, and contemporary studios.",
            "setting": "Creative Workspaces, Dinners & Weekend Socials",
            "gradient": "linear-gradient(120deg, #8B5CF6 0%, #6D28D9 100%)"
        }
        
    # 4. Activewear / Performance
    if any(k in art for k in ["track", "sports", "running", "swimwear", "jersey", "shorts"]) or "sports" in sub:
        return {
            "archetype": "Performance Athleisure",
            "tag": "⚡ High-Mobility Activewear",
            "desc": "Ergonomically engineered for lightweight breathability, athletic performance, and seamless studio workout transitions.",
            "setting": "Gym Training, Morning Runs & Outdoor Athletics",
            "gradient": "linear-gradient(120deg, #10B981 0%, #047857 100%)"
        }
        
    # 5. Streetwear / Casual Lounge
    return {
        "archetype": "Modern Streetwear & Everyday",
        "tag": "🧢 Urban Casual Statement",
        "desc": "Laid-back, expressive aesthetic rooted in skate and modern street culture, prioritizing expressive styling and all-day comfort.",
        "setting": "Urban Outings, Coffee Shops & Casual Travel",
        "gradient": "linear-gradient(120deg, #EC4899 0%, #BE185D 100%)"
    }

# --- STYLING COORDINATOR ---
def get_curated_pairings(archetype, article, colour):
    pairings = []
    if "Formal" in archetype:
        pairings.append(("Bottomwear", "Tailored Pleated Trousers (Charcoal or Midnight Navy)"))
        pairings.append(("Footwear", "Burnished Leather Oxfords or Double Monk Straps"))
        pairings.append(("Accessories", "Brushed Silver Automatic Watch & Clean Dress Belt"))
    elif "Smart Casual" in archetype:
        pairings.append(("Bottomwear", "Slim Italian Chinos (Tan, Olive, or Raw Selvedge Denim)"))
        pairings.append(("Footwear", "Minimalist White Leather Sneakers or Suede Loafers"))
        pairings.append(("Layering", "Unstructured Wool-Linen Overcoat or Knit Cardigan"))
    elif "Ethnic" in archetype:
        pairings.append(("Bottomwear", "Silk Churidar or Contrast Draped Dhoti"))
        pairings.append(("Footwear", "Handmade Embroidered Juttis or Mojaris"))
        pairings.append(("Accessories", "Statement Pocket Square or Brocade Shawl"))
    elif "Performance" in archetype:
        pairings.append(("Coordinates", "Seamless Compression Tights or Performance Joggers"))
        pairings.append(("Footwear", "Carbon-Plated Cushion Road Trainers"))
        pairings.append(("Gear", "Matte Ergonomic Cap & Fitness Tracker"))
    else:
        pairings.append(("Bottomwear", "Wide-Leg Washed Denim or Heavyweight Cargo Pants"))
        pairings.append(("Layering", "Oversized Flannel Shirt or Boxy Utility Jacket"))
        pairings.append(("Footwear", "Chunky Retro Sneakers or Suede Desert Boots"))
    return pairings

# --- GRAD-CAM EXPLAINABLE AI ---
def generate_gradcam(model, input_tensor, target_idx):
    model.eval()
    feat_map = model.features(input_tensor)
    pooled = model.avgpool(feat_map)
    feat = model.shared_fc(torch.flatten(pooled, 1))
    logits = model.head_article(feat)
    
    score = logits[0, target_idx]
    grads = torch.autograd.grad(score, feat_map, retain_graph=False)[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    
    cam = torch.sum(weights * feat_map, dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# --- COLOR PALETTE EXTRACTOR ---
def extract_palette(pil_img, k=4):
    img = np.array(pil_img.resize((80, 80)))
    mask = ~((img[:, :, 0] > 235) & (img[:, :, 1] > 235) & (img[:, :, 2] > 235))
    pixels = img[mask]
    if len(pixels) < k:
        pixels = img.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    sorted_idx = np.argsort(counts)[::-1]
    
    palette = []
    for idx in sorted_idx:
        rgb = centers[idx].astype(int)
        hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}".upper()
        pct = (counts[idx] / len(labels)) * 100
        palette.append((hex_code, pct))
    return palette

# --- LOAD LOCAL ASSETS ---
@st.cache_resource
def load_fashion_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    json_path = base_dir / "label_mappings.json"
    model_path = base_dir / "best_fashion_model.pth"
    
    with open(json_path, "r") as f:
        mappings = json.load(f)
        
    id2sub = {v: k for k, v in mappings["subCategory"].items()}
    id2art = {v: k for k, v in mappings["articleType"].items()}
    id2col = {v: k for k, v in mappings["baseColour"].items()}
    
    model = MultiHeadFashionNet(len(id2sub), len(id2art), len(id2col)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    preprocess = transforms.Compose([
        LetterboxResize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, device, id2sub, id2art, id2col, preprocess

model, device, id2sub, id2art, id2col, preprocess = load_fashion_system()

# --- REFINED SIDEBAR (LUXURY ATELIER FILTERS ONLY) ---
with st.sidebar:
    st.markdown("### ✨ Haute Studio")
    st.markdown("Customize your visual analysis session:")
    
    input_mode = st.radio(
        "Garment Input Source",
        ["📁 Studio File Upload", "📸 Live Runway Camera"],
        index=0
    )
    
    st.divider()
    st.markdown("### 🎨 Wardrobe Collections")
    collection_filter = st.selectbox(
        "Current Season Context",
        ["Autumn / Winter Formal", "Spring / Summer Resort", "Urban Streetwear", "Bridal & Heritage"]
    )
    
    st.divider()
    st.markdown("### 💡 Style Director Tip")
    st.caption("Upload well-lit garment photos on plain neutral backgrounds to ensure maximum fabric texture fidelity and exact color matching.")

# --- EDITORIAL HERO BANNER ---
st.markdown("""
<div class="fashion-hero">
    <div class="brand-title">HAUTE.AI</div>
    <div class="brand-tagline">Intelligent Fashion Atelier • Multi-Tier Taxonomy • Automated Wardrobe Styling</div>
</div>
""", unsafe_allow_html=True)

# --- WORKSPACE ---
col_studio, col_runway = st.columns([1, 1.3], gap="large")

garment_img = None

with col_studio:
    st.markdown("### 1. Studio Upload")
    if "Studio File" in input_mode:
        uploaded_file = st.file_uploader("Select high-res product photo", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            garment_img = Image.open(uploaded_file).convert("RGB")
    else:
        cam_feed = st.camera_input("Capture product through camera")
        if cam_feed:
            garment_img = Image.open(cam_feed).convert("RGB")
            
    if garment_img is not None:
        letterbox_fn = LetterboxResize((224, 224))
        canvas_img = letterbox_fn(garment_img)
        
        tab_vogue, tab_cam, tab_raw = st.tabs(["✨ Standardized Canvas", "🔍 Model Attention (Grad-CAM)", "📷 Original File"])
        with tab_vogue:
            st.image(canvas_img, caption="Preserved Aspect-Ratio (224x224 Canvas)", use_container_width=True)
        with tab_raw:
            st.image(garment_img, caption=f"Original Dimensions: {garment_img.size}", use_container_width=True)

with col_runway:
    st.markdown("### 2. Style Intelligence & Analysis")
    
    if garment_img is not None:
        input_tensor = preprocess(garment_img).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        outputs = model(input_tensor)
        
        p_sub = F.softmax(outputs["subCategory"], dim=1)[0]
        p_art = F.softmax(outputs["articleType"], dim=1)[0]
        p_col = F.softmax(outputs["baseColour"], dim=1)[0]
        
        idx_sub = torch.argmax(p_sub).item()
        idx_art = torch.argmax(p_art).item()
        idx_col = torch.argmax(p_col).item()
        
        name_sub, conf_sub = id2sub[idx_sub], p_sub[idx_sub].item() * 100
        name_art, conf_art = id2art[idx_art], p_art[idx_art].item() * 100
        name_col, conf_col = id2col[idx_col], p_col[idx_col].item() * 100
        
        # Style Archetype & Occasion Engine
        style_vibe = detect_occasion_and_vibe(name_sub, name_art, name_col)
        
        # 1. Occasion Highlight Card
        st.markdown(f"""
        <div class="style-vibe-card" style="background: {style_vibe['gradient']};">
            <div class="vibe-tag">{style_vibe['tag']}</div>
            <div class="vibe-title">{style_vibe['archetype']}</div>
            <div class="vibe-desc">{style_vibe['desc']}</div>
            <div style="margin-top:10px; font-weight:700; font-size:0.85rem;">
                📍 Recommended For: <u>{style_vibe['setting']}</u>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Taxonomy Breadcrumbs
        st.markdown(f"""
        <div style="margin-bottom:15px;">
            <span class="chip chip-purple">Category: {name_sub}</span>
            <span class="chip chip-pink">Article: {name_art}</span>
            <span class="chip chip-amber">Colorway: {name_col}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Vibrant Metric Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="fashion-card">
                <div class="card-label-sub">Sub-Category</div>
                <div class="card-title-lg">{name_sub}</div>
                <strong style="color:#8B5CF6;">{conf_sub:.1f}% Confidence</strong>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="fashion-card">
                <div class="card-label-art">Article Type</div>
                <div class="card-title-lg">{name_art}</div>
                <strong style="color:#EC4899;">{conf_art:.1f}% Confidence</strong>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="fashion-card">
                <div class="card-label-col">Base Tone</div>
                <div class="card-title-lg">{name_col}</div>
                <strong style="color:#F59E0B;">{conf_col:.1f}% Confidence</strong>
            </div>
            """, unsafe_allow_html=True)

        # Grad-CAM in Tab 2
        cam = generate_gradcam(model, input_tensor, idx_art)
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_MAGMA)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * np.array(canvas_img) + 0.4 * heat)
        with tab_cam:
            st.image(overlay, caption=f"Model Focus on Silhouette & Lapel for '{name_art}'", use_container_width=True)

        st.divider()

        # 4. Color Palette & Styling Coordination
        col_swatches, col_pairings = st.columns([1, 1.2])
        
        with col_swatches:
            st.markdown("#### 🎨 Extracted Fabric Palette")
            palette = extract_palette(canvas_img, k=4)
            for hex_val, pct in palette:
                st.markdown(f"""
                <div class="palette-pill">
                    <span class="swatch-circle" style="background:{hex_val};"></span>
                    <code>{hex_val}</code> <span style="opacity:0.65; margin-left:6px;">({pct:.0f}%)</span>
                </div>
                """, unsafe_allow_html=True)

        with col_pairings:
            st.markdown("#### 👔 Curated Outfit Coordination")
            pairings = get_curated_pairings(style_vibe['archetype'], name_art, name_col)
            for role, item in pairings:
                st.markdown(f"""
                <div class="outfit-pairing-row">
                    <strong>{role}:</strong> {item}
                </div>
                """, unsafe_allow_html=True)

        # 5. Top 5 Probabilities
        with st.expander("📊 Fine-Grained Candidate Confidence Distribution"):
            top5_probs, top5_idx = torch.topk(p_art, 5)
            df_chart = pd.DataFrame({
                "Article": [id2art[i.item()] for i in top5_idx],
                "Confidence (%)": [p.item() * 100 for p in top5_probs]
            }).set_index("Article")
            st.bar_chart(df_chart, horizontal=True)

    else:
        st.info("👈 Upload a garment photo on the left studio panel to reveal its style archetype, occasion, and outfit pairings.")