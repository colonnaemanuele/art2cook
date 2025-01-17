"""Script with shared constants for most scripts"""
import os
from pathlib import Path

# Directory in which the project is stored
PROJ_DIR = str(Path(__file__).resolve().parent.parent)

# Data folders
DATA_DIR = os.path.join(PROJ_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
EXTRA_DATA_DIR = os.path.join(DATA_DIR, "extra")
PROC_DATA_DIR = os.path.join(DATA_DIR, "processed")
IMG_DIR = os.path.join(DATA_DIR, "images")
AUD_DIR = os.path.join(DATA_DIR, "audio")
ARTGRAPH_DIR = os.path.join(DATA_DIR, "images", "imagesf2")
FMA_DIR = os.path.join(DATA_DIR, "audio", "fma_large")

# Scripts folders
SRC_DIR = os.path.join(PROJ_DIR, "src")
AUDIOLDM_DIR = os.path.join(SRC_DIR, "audioldm")
AUDIOLDM2_DIR = os.path.join(SRC_DIR, "audioldm2")
IMAGEBIND_DIR = os.path.join(SRC_DIR, "imagebind")
SRC_DATA_DIR = os.path.join(SRC_DIR, "data")