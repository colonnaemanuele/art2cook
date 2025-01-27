import os
import sys
sys.path.append("src")
import torch
import numpy as np
import soundfile as sf

import conf
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR)

from test_art2mus import generate_audio
import art2mus.utils.train_test_utils as tu
from art2mus.art2mus_4_pipeline import AudioLDM2Pipeline
from art2mus.utils.imagebind_utils import load_model as load_image_bind
from art2mus.utils.imagebind_utils import generate_embeds, compute_similarity


MODEL_REPO_ID = tu.AUDIOLDM2_REPO_ID
NEG_PROMPT = tu.DEFAULT_NEGATIVE_PROMPT
CURRENT_FILE_PATH = os.path.abspath(__file__)
TEST_DATA_DIR = os.path.join(os.path.abspath(__file__).parent, 'data')
TEST_IMG_OUT_DIR = os.path.join(TEST_DATA_DIR, 'test_images')
TEST_AUD_OUT_DIR = os.path.join(TEST_DATA_DIR, 'test_audio')


def load_art2mus_pipeline(device):
    if device == 'cuda':
        print("Loading model with torch.float16, needed to work with GPU.")
        return AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=MODEL_REPO_ID, torch_dtype=torch.float16,
                                                 custom_pipeline=os.path.join(PROJ_DIR,"/src/art2mus/art2mus_4_pipeline.py"))
    else:
        print("Loading model without torch.float16, needed to work with CPU.")
        return AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=MODEL_REPO_ID, 
                                                 custom_pipeline=os.path.join(PROJ_DIR,"/src/art2mus/art2mus_4_pipeline.py"))


def test_music_file_not_empty(input_data):
    """
    Test to check if music file is non-empty.
    Input can be either a file path or a numpy array (tensor).
    """
    if isinstance(input_data, str):
        data, _ = sf.read(input_data)
    elif isinstance(input_data, np.ndarray):
        data = input_data
    else:
        raise ValueError("Input must be a file path or a numpy array.")
    
    assert np.any(data), "The audio data is empty"

    
def test_partial_image():
    """
    Test to check if music generated from a portion of the image is similar.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    og_img_path = os.path.join(TEST_IMG_OUT_DIR, 'og_thistles-on-orange.jpg')
    small_img_path = os.path.join(TEST_IMG_OUT_DIR, 'small_thistles-on-orange.jpg')
    
    imagebind = load_image_bind(False, use_cpu=True)
    art2mus = load_art2mus_pipeline(device)
    
    generate_audio(art2mus, img_path=og_img_path, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=1, 
                   output_dir=TEST_AUD_OUT_DIR, file_name=f"og_thistles_on_orange.wav")
    
    generate_audio(art2mus, img_path=small_img_path, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=1, 
                   output_dir=TEST_AUD_OUT_DIR, file_name=f"small_thistles_on_orange.wav")
    
    og_img_aud_path = os.path.join(TEST_AUD_OUT_DIR, 'og_thistles_on_orange.wav')
    small_img_aud_path = os.path.join(TEST_AUD_OUT_DIR, 'small_thistles_on_orange.wav')
    
    aud_paths = [og_img_aud_path, small_img_aud_path]
    imgs_embeds = generate_embeds(imagebind, curr_dev=device, audio_paths=aud_paths, extract_emb=True, emb_type='audio')
    
    # Remove generated audio files
    for path in aud_paths:
        if os.path.exists(path):
            os.remove(path)
    
    og_emb, variant_emb = imgs_embeds[0], imgs_embeds[1]
    similarity_val = compute_similarity(og_emb, variant_emb)
    
    assert similarity_val >= 0.8, "Music generated from a portion of the image is not similar enough"

    
    
def test_brightness_variation():
    """
    Test to check if music varies by changing brightness of the image.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    og_img_path = os.path.join(TEST_IMG_OUT_DIR, 'og_thistles-on-orange.jpg')
    dark_img_path = os.path.join(TEST_IMG_OUT_DIR, 'dark_thistles-on-orange.jpg')
    
    imagebind = load_image_bind(False, use_cpu=True)
    art2mus = load_art2mus_pipeline(device)
    
    generate_audio(art2mus, img_path=og_img_path, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=1, 
                   output_dir=TEST_AUD_OUT_DIR, file_name=f"og_thistles_on_orange.wav")
    
    generate_audio(art2mus, img_path=dark_img_path, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=1, 
                   output_dir=TEST_AUD_OUT_DIR, file_name=f"dark_thistles_on_orange.wav")
    
    og_img_aud_path = os.path.join(TEST_AUD_OUT_DIR, 'og_thistles_on_orange.wav')
    dark_img_aud_path = os.path.join(TEST_AUD_OUT_DIR, 'dark_thistles_on_orange.wav')
    
    aud_paths = [og_img_aud_path, dark_img_aud_path]
    imgs_embeds = generate_embeds(imagebind, curr_dev=device, audio_paths=aud_paths, extract_emb=True, emb_type='audio')
    
    # Remove generated audio files
    for path in aud_paths:
        if os.path.exists(path):
            os.remove(path)
    
    og_emb, variant_emb = imgs_embeds[0], imgs_embeds[1]
    similarity_val = compute_similarity(og_emb, variant_emb)
    
    assert similarity_val < 0.8, "Music generated from images with different brightness is similar"


def test_similar_music():
    """
    Test to check if music generated from similar images is similar.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    og_img_path = os.path.join(TEST_IMG_OUT_DIR, 'og_thistles-on-orange.jpg')
    sim_img_path = os.path.join(TEST_IMG_OUT_DIR, 'og_california_thistles.jpg')
    
    
    imagebind = load_image_bind(False, use_cpu=True)
    art2mus = load_art2mus_pipeline(device)
    
    aud_paths = [og_img_path, sim_img_path]
    
    generate_audio(art2mus, img_path=og_img_path, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=1, 
                   output_dir=TEST_AUD_OUT_DIR, file_name=f"og_thistles_on_orange.wav")
    
    generate_audio(art2mus, img_path=sim_img_path, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=1, 
                   output_dir=TEST_AUD_OUT_DIR, file_name=f"og_california_thistles.wav")
    
    og_img_aud_path = os.path.join(TEST_AUD_OUT_DIR, 'og_thistles_on_orange.wav')
    sim_img_aud_path = os.path.join(TEST_AUD_OUT_DIR, 'og_california_thistles.wav')
    
    aud_paths = [og_img_aud_path, sim_img_aud_path]
    imgs_embeds = generate_embeds(imagebind, curr_dev=device, audio_paths=aud_paths, extract_emb=True, emb_type='audio')
    
    # Remove generated audio files
    for path in aud_paths:
        if os.path.exists(path):
            os.remove(path)
    
    og_emb, variant_emb = imgs_embeds[0], imgs_embeds[1]
    similarity_val = compute_similarity(og_emb, variant_emb)
    
    assert similarity_val >= 0.8, "Music generated from similar images is not similar enough"