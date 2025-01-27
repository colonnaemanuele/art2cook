from metaflow import FlowSpec, step
from src.art2mus.art2mus_4_train import (
    ARTGRAPH_FOLDER,
    AUDIO_ST,
    FMA_FOLDER,
    IMAGE_AUDIO_JSON,
    IMAGE_ST,
)

import os
import sys
import math
import scipy
import shutil
import logging
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
# Diffusers
from diffusers.utils import is_wandb_available
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
# Accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# Torch
import torch
from torch.utils.data import Subset
import torch.nn.functional as torch_func

sys.path.append("src")
import conf

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
sys.path.append(PROJ_DIR + "/src")
sys.path.append(PROJ_DIR + "/src/audioldm")
sys.path.append(PROJ_DIR + "/src/art2mus")

import art2mus.utils.torch_tools as tt
from art2mus.utils.imagebind_utils import load_model
from art2mus.utils.train_test_argparse import parse_train_args

from my_dataset import ImageAudioDataset
import art2mus.utils.train_test_utils as tu
from src.art2mus.art2mus_4_pipeline import AudioLDM2Pipeline

# ImageBind stuff
from art2mus.utils.imagebind_utils import load_model

class Art2MusTrainFlow(FlowSpec):
    
    # TRAIN_CONFIG = tu.TrainingConfig()
    
    set_seed(0)
    device = 'cuda'
    using_cuda = True
    IMAGEBIND = load_model(False)
    stft = tu.load_stft()
    
    LOG_DIR = tu.LOG_DIR
    TMP_DIR_GT = tu.TMP_GT_DIR
    TMP_DIR_GEN = tu.TMP_GEN_DIR
    VAL_AUDIO_DIR = tu.VAL_AUDIO_DIR
    MODEL_OUT_DIR = tu.MODEL_OUT_DIR
    LAYER_WEIGHTS = tu.IMG_PROJ_LAYER_WEIGHTS
    
    AUDIOLDM_REPO = tu.AUDIOLDM2_REPO_ID
    CUSTOM_PIPE = tu.CUSTOM_PIPE_2
    EMBEDS_DTYPE = torch.float16
    
    use_cpu = False
    skip_train = False
    use_snr_gamma = False
    use_val_subset = False
    use_training_subset = False
    res_from_checkpoint = False
    use_large_batch_size = True

    num_epochs = 3
    BATCH_SIZE = 2
    max_grad_norm = 1.0
    audio_duration = 10
    guidance_scale = 3.5
    num_inference_steps = 200
    num_waveforms_per_prompt = 1
    audio_duration_in_seconds = 10.0
    max_eval_audios = 101 + (num_epochs * 100) 
    target_length = int(audio_duration * 102.4)
    NEGATIVE_PROMPT = tu.DEFAULT_NEGATIVE_PROMPT
    gradient_accumulation_steps = 2 // BATCH_SIZE
    
    accelerator_project_conf = ProjectConfiguration(MODEL_OUT_DIR, LOG_DIR)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              project_config=accelerator_project_conf,
                              cpu=use_cpu,
                              )
    
    TS_LIST = [10, 50, 100, 250, 500, 800]
    ts_loss_dict = {f'{ts_val}': [] for ts_val in TS_LIST}
    
    
    def prepare_negative_prompt(self, batch_size, neg_prompt=NEGATIVE_PROMPT):
        return [neg_prompt] * batch_size if batch_size > 1 else neg_prompt
    
    
    def process_audio(self, audio_path):
        try:
            mel, _, _ = tt.wav_to_fbank(audio_path, self.target_length, self.stft)
            mel = mel.unsqueeze(1).to(self.device).to(dtype=self.EMBEDS_DTYPE)
            return mel
        except Exception as e:
            print(f"Issues with {audio_path}: {e}")
            return None
        
    def generate_noisy_latents_and_noise(self, mel):
        with torch.no_grad():
            latents = self.pipe.vae.encode(mel).latent_dist.sample()
        latents *= self.pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
        ).long()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps).to(self.device)
        
        return noisy_latents, timesteps, noise
    
    
    def generate_noise(self, image_emb, negative_prompt, noisy_latents, timesteps):
        generated_noise, _ = self.pipe.__train__(
            image_embeds=image_emb,
            negative_prompt=negative_prompt,
            num_waveforms_per_prompt=self.num_waveforms_per_prompt,
            latents=noisy_latents,
            guidance_scale=self.guidance_scale,
            timesteps=timesteps,
        )
        return generated_noise
    
    
    # Does not handle SNR Gamma loss
    def compute_loss(self, generated_noise, target_noise, timesteps, ts_loss_dict):
        loss = torch_func.mse_loss(generated_noise.float(), target_noise.float(), reduction="none")
        for idts, ts in enumerate(timesteps):
            ts = str(ts.item())
            if ts in ts_loss_dict:
                ts_loss_dict[ts].append(loss[idts].mean().item())

        for ts_key, ts_losses in ts_loss_dict.items():
            if ts_losses:
                mean_loss = sum(ts_losses) / len(ts_losses)
                ts_loss_dict[ts_key] = []

        return loss.mean()
    
    
    def optimizer_step(self, loss):
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.pipe.img_project_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
    
    
    def generate_validation_audio(self, image_emb, negative_prompt, generator):
        gen_music = self.pipe(
            image_embeds=image_emb,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_inference_steps,
            audio_length_in_s=self.audio_duration_in_seconds,
            num_waveforms_per_prompt=self.num_waveforms_per_prompt,
            generator=generator,
            guidance_scale=self.guidance_scale,
        ).audios
        return gen_music 
    
    
    @step
    def start(self):
        
        self.dataset = ImageAudioDataset(
            json_file=IMAGE_AUDIO_JSON,
            images_dir=ARTGRAPH_FOLDER,
            img_emb_file=IMAGE_ST,
            audios_dir=FMA_FOLDER,
            audio_emb_file=AUDIO_ST,
        )

        self.train_data, self.val_data = self.dataset.train_val_test_split(val_size=0.2, random_state=0)

        self.pipe = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path=self.AUDIOLDM_REPO,
                                                      custom_pipeline=self.CUSTOM_PIPE,)
    
        self.pipe = self.pipe.to(self.device)
        self.generator = torch.Generator(self.device).manual_seed(0)

        self.pipe.img_project_model.requires_grad_(True) 
        self.pipe.projection_model.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        self.pipe.language_model.requires_grad_(False) 
        self.pipe.text_encoder.requires_grad_(False)    
        self.pipe.vocoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)       
    
        self.pipe.unet.eval()
        self.pipe.img_project_model.train()
        self.noise_scheduler = self.pipe.scheduler
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.BATCH_SIZE, 
                                                            shuffle=True,
                                                            num_workers=4,)
        
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, 
                                                          batch_size=self.BATCH_SIZE, 
                                                          shuffle=True,
                                                          num_workers=4,)
    
        update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
                self.max_train_steps = self.num_epochs * update_steps_per_epoch
        self.num_epochs = math.ceil(self.max_train_steps / update_steps_per_epoch)
    
        self.optimizer = torch.optim.AdamW(
            self.pipe.img_project_model.parameters(),
            lr=2e-5,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )

        self.lr_scheduler = get_scheduler(
                self.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.lr_warmup_steps * self.num_processes,
                num_training_steps=self.max_train_steps * self.accelerator.num_processes,
            )

        print('fac')
        self.pipe.img_project_model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.pipe.img_project_model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )
        print('iola3')

        self.stft = tu.load_stft()
        self.target_length = int(10 * 102.4)

        self.global_step = 0
        self.completed_val_steps = 0
        self.first_epoch = 0
        
        self.batch_indices = list(range(len(self.train_dataloader)))
        self.next(self.end)
    
    # @step
    # def train(self):
    #     no_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
    #     print(f"Will check if epochs has to end after {no_steps_per_epoch} steps.\n\n")
    #     self.progress_bar = tqdm(
    #         range(0, self.max_train_steps),
    #         initial=0,
    #         desc="Current step (w.r.t. max train steps)",
    #         disable=not self.accelerator.is_local_main_process,
    #     )
    #     self.num_epochs = list(range(self.first_epoch, self.num_epochs))
    #     self.next(self.train_epoch, foreach='num_epochs')
    
    # @step
    # def train_epoch(self):
    #     self.train_step_loss = 0.0
        
    #     for _, batch in enumerate(self.train_dataloader):
    #         image_emb, audio_path = batch
    #         negative_prompt = self.prepare_negative_prompt(image_emb.shape[0])
            
    #         # TODO: #1
    #         """ --- Convert audio to mel spectrogram --- """
    #         mel = self.process_audio(audio_path)
            
    #         """ --- Compute latents starting from the mel-spectrogram --- """
    #         noisy_latents, timesteps, noise = self.generate_noisy_latents_and_noise(mel)
    #         target = noise
    #         # TODO: #3
    #         """ --- Noise Generation Procedure --- """
    #         generated_noise = self.generate_noise(image_emb, 
    #                                                 negative_prompt, 
    #                                                 self.no_waveforms_per_prompt,
    #                                                 noisy_latents, 
    #                                                 self.guidance_scale,
    #                                                 timesteps)
    #         # Removed SNR Gamma loss computation (ease out the code)
    #         loss = self.compute_loss(generated_noise, target, timesteps, self.ts_loss_dict)
                
    #         # TODO: #5
    #         avg_loss = self.accelerator.gather(loss.repeat(self.BATCH_SIZE)).mean()
    #         self.train_step_loss += avg_loss.item() / self.gradient_accumulation_steps
    #         self.epoch_loss += self.train_step_loss
            
    #         self.optimizer_step(loss)
    #         self.progress_bar.update(1)
    #     self.next(self.join_training)
        
        
    # @step
    # def join_training(self, inputs):
    #     self.loss = sum(inputs.loss)
    #     self.next(self.end)
  
    @step
    def end(self):
        print(f'Loss: {self.loss}')
        print("Done training!")


if __name__ == "__main__":
    Art2MusTrainFlow()