import os
import sys
import scipy
import torch

CURR_FILE_PATH = os.path.abspath(__file__)
PROJ_DIR = os.path.join(CURR_FILE_PATH.split('art2cook/')[0], 'art2cook')
SRC_DIR = os.path.join(PROJ_DIR, 'src')
sys.path.append(PROJ_DIR)
sys.path.append(SRC_DIR)

import conf
PROJ_DIR = conf.PROJ_DIR

from art2mus.art2mus_4_pipeline import AudioLDM2Pipeline
import art2mus.utils.train_test_utils as tu


# Seed
SEED = 0
# Example negative prompt
NEG_PROMPT = "Low quality."
# HuggingFace's model repo id 
MODEL_REPO_ID = tu.AUDIOLDM2_REPO_ID
# Test audio output folder
OUT_DIR = os.path.join(PROJ_DIR, "test_music")
# Path of an example artwork to use to generate music
EXAMPLE_ARTWORK_PATH = 'test_images/erin-hanson_thistles-on-orange-2016.jpg'


def generate_audio(pipe, generator, prompt=None, img_path=None, img_emb=None, neg_prompt="Low quality.",
                   inf_steps=200, aud_len=10.0, waveforms=3,
                   output_dir=OUT_DIR, file_name="techno.wav"):
    
    print(f"Generating music based on the provided artwork....\n======================")
    
    audio = pipe(
        prompt=prompt,
        image_path=img_path,
        image_embeds=img_emb,
        negative_prompt=neg_prompt,
        num_inference_steps=inf_steps,
        audio_length_in_s=aud_len,
        num_waveforms_per_prompt=waveforms,
        generator=generator,
    ).audios
        
    out_file_path = os.path.join(output_dir, file_name)
    scipy.io.wavfile.write(out_file_path, rate=16000, data=audio[0])
    print(f"Music file stored at: {out_file_path}")


def generate_music(path_to_artwork=None, music_file_name="art2mus_example.wav"):
    
    # Assess if CUDA is available
    if torch.cuda.is_available():
        tot_gpu_mem = round(torch.cuda.mem_get_info()[1] / 1024 ** 3, 2)
        free_gpu_mem = round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 2)
        print(f"Free memory/Total memory: {free_gpu_mem}/{tot_gpu_mem} GIB")
        
        # At least 2.8GiB are needed to run Art2Mus on GPU
        if free_gpu_mem >= 3.0:
            print("Using CUDA!")
            device = "cuda"
        else:
            print("Not enough free GPU memory! Using CPU.")
            device = 'cpu'
    else:
        print("CUDA not found! Using CPU.")
        device = 'cpu'

    # Load Model
    if device == 'cuda':
        print("Loading model with torch.float16, needed to work with GPU.")
        pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=MODEL_REPO_ID, torch_dtype=torch.float16, 
                                                 custom_pipeline=os.path.join(PROJ_DIR, "src/art2mus/art2mus_4_pipeline.py"))
    else:
        print("Loading model without torch.float16, needed to work with CPU.")
        pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=MODEL_REPO_ID, 
                                                 custom_pipeline=os.path.join(PROJ_DIR, "src/art2mus/art2mus_4_pipeline.py"))
        
    pipe = pipe.to(device)
    print(f"Pipeline moved to: {device}!")

    generator = torch.Generator(device).manual_seed(SEED)

    file_name = 'art2mus_example.wav'
    
    # Generate music
    generate_audio(pipe=pipe, generator=generator, 
                   prompt=None, img_path=path_to_artwork, img_emb=None, neg_prompt=NEG_PROMPT,
                   inf_steps=200, aud_len=10.0, waveforms=3, 
                   output_dir=OUT_DIR, file_name=file_name)
    
    torch.cuda.empty_cache()
    
    out_file_path = os.path.join(OUT_DIR, file_name)
    
    if os.path.exists(out_file_path):
        return (out_file_path, True)
    else:
        return (None, False)
    

if __name__ == "__main__":
    generate_music(music_file_name="art2mus_example_TESTING.wav")