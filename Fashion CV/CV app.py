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
    .card-label-art { color: #EC4899; font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;
