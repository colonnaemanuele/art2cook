import os
import sys 
import shutil             
import zipfile as zf
sys.path.append('src')
import conf


# Folders
AUD_DIR = conf.AUD_DIR
IMG_DIR = conf.IMG_DIR
RAW_DATA_DIR = conf.RAW_DATA_DIR
PROC_DATA_DIR = conf.PROC_DATA_DIR
EXTRA_DATA_DIR = conf.EXTRA_DATA_DIR

# Data
ZIP_DIR = os.path.join(RAW_DATA_DIR, 'art2cook_data.zip')

AUDIOS_ST = os.path.join(PROC_DATA_DIR, 'audios.safetensors')
AUDIOS_OUT_ST = os.path.join(EXTRA_DATA_DIR, 'audios.safetensors')

IMAGES_ST = os.path.join(PROC_DATA_DIR, 'images.safetensors')
IMAGES_OUT_ST = os.path.join(EXTRA_DATA_DIR, 'images.safetensors')

PAIRS_JSON = os.path.join(PROC_DATA_DIR, 'image_audio_subset_df.json')
PAIRS_OUT_JSON = os.path.join(EXTRA_DATA_DIR, 'image_audio_subset_df.json')

FMA_AUDIO_ZIP = os.path.join(PROC_DATA_DIR, 'fma_large.zip')
ARTGRAPH_IMGS_ZIP = os.path.join(PROC_DATA_DIR, 'imagesf2.zip')


def check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def extract_data():
    """ Method to extract data"""
    check_dir_exists(RAW_DATA_DIR)
    check_dir_exists(PROC_DATA_DIR)
    
    try:
        with zf.ZipFile(ZIP_DIR, 'r') as zip_ref:
            zip_ref.extractall(PROC_DATA_DIR)
            
        extracted_files = [f for f in os.listdir(PROC_DATA_DIR) if 'art2cook_data.zip' not in f]
        print("The files extracted from the zip file are:")
        for ef in extracted_files:
            print(f'- {ef}')   
        
        with zf.ZipFile(FMA_AUDIO_ZIP, 'r') as zip_ref:
            zip_ref.extractall(AUD_DIR)
        os.remove(FMA_AUDIO_ZIP)
            
        with zf.ZipFile(ARTGRAPH_IMGS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(IMG_DIR)
        os.remove(ARTGRAPH_IMGS_ZIP)

        # Delete old files if found
        if os.path.exists(AUDIOS_OUT_ST):
            os.remove(AUDIOS_OUT_ST)
            
        if os.path.exists(IMAGES_OUT_ST):
            os.remove(IMAGES_OUT_ST)
            
        if os.path.exists(PAIRS_OUT_JSON):
            os.remove(PAIRS_OUT_JSON)

        shutil.move(AUDIOS_ST, EXTRA_DATA_DIR)
        shutil.move(IMAGES_ST, EXTRA_DATA_DIR)
        shutil.move(PAIRS_JSON, EXTRA_DATA_DIR)
        
        return 'Data extracted succesfully.'
                   
    except FileNotFoundError:
        print("The zip file was not found.")
        return 'Error during data extraction.'


if __name__ == "__main__":
    extract_data()