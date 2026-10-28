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
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif; }

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
.outfit-pairing-row {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    border-left: 5px solid #6366F1;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    color: #1E293B;
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

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
        self.head_
