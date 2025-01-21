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
# Wandb
import wandb 

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



class Art2MusTrainFlow(FlowSpec):
    
    AUDIOLDM_REPO = tu.AUDIOLDM2_REPO_ID
    CUSTOM_PIPE = tu.CUSTOM_PIPE_2
    EMBEDS_DTYPE = torch.float16
    use_snr_gamma = False
    use_large_batch_size = True
    use_training_subset = False
    use_val_subset = False
    use_cpu = False
    res_from_checkpoint = False
    skip_train = False
    set_seed(0)
    LAYER_WEIGHTS = tu.IMG_PROJ_LAYER_WEIGHTS
    num_epochs = 3
    MODEL_OUT_DIR = tu.MODEL_OUT_DIR
    LOG_DIR = tu.LOG_DIR

    
    accelerator_project_conf = ProjectConfiguration(MODEL_OUT_DIR, LOG_DIR)
    BATCH_SIZE = 2
    gradient_accumulation_steps = 2 // BATCH_SIZE
    
    max_eval_audios = 101 + (num_epochs * 100) 

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              project_config=accelerator_project_conf,
                              log_with="wandb",
                              cpu=use_cpu,
                              )
    
    device = 'cuda'
    using_cuda = True

    
    
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
    
        self.pipe.img_project_model.train()
        self.pipe.unet.eval()
        
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

        # Update the number of traning epochs based on the no. update steps per epoch
        self.num_epochs = math.ceil(self.max_train_steps / update_steps_per_epoch)
    
        self.optimizer_cls = torch.optim.AdamW
        
        self.optimizer = self.optimizer_cls(
            self.pipe.img_project_model.parameters(),
            lr=2e-5,
            betas=(0.9, 0.999),
            weight_decay=e-2,
            eps=1e-08,
        )

        self.lr_scheduler = get_scheduler(
                self.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.lr_warmup_steps * self.num_processes,
                num_training_steps=self.max_train_steps * self.accelerator.num_processes,
            )

        
        self.pipe.img_project_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = self.accelerator.prepare(
            self.pipe.img_project_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # Load Short-Time Fourier Transform (STFT) module 
        self.stft = tu.load_stft()
        self.target_length = int(10 * 102.4)

        # Number of completed train and validation steps 
        self.global_step = 0
        self.completed_val_steps = 0
        self.first_epoch = 0
        
        
        # Fan out to process each batch independently
        self.next(self.process_batch, foreach="batch_indices")

    @step
    def train(self):
        
        for epoch in tqdm(range(0, 3), desc="Training epochs"):
            
            # Training step loss
            train_step_loss = 0.0
            # Total epoch loss
            epoch_loss = 0.0
            
            validation_done = False
            
            noise_scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps)
            
            for _, batch in enumerate(train_dataloader):
                            
                # Skip first train epoch if needed
                if skip_train:
                    break
                
                with accelerator.accumulate(pipe.img_project_model):
                                    
                    image_emb, audio_path = batch
                    
                    # Repeat the Negative Prompt based on the batch size (image_emb.shape[0])
                    if image_emb.shape[0] > 1:
                        negative_prompt = [NEGATIVE_PROMPT] * image_emb.shape[0]
                    else:
                        negative_prompt = NEGATIVE_PROMPT
                    
                    """ --- Convert audio to latent space --- """
                    # Compute mel-spectrogram of the audio (this is our ground truth)
                    try:
                        # TODO: #1
                        mel, _, _ = tt.wav_to_fbank(audio_path, target_length, stft)
                    except Exception as e:
                        print(f"Issues with {audio_path}: {e}")
                        progress_bar.update(1)
                        global_step += 1
                        continue
                        
                    mel = mel.unsqueeze(1).to(device)
                    mel = mel.to(dtype=EMBEDS_DTYPE)
                    
                    """ --- Latents Computation --- """
                    # Compute latents starting from the mel-spectrogram
                    with torch.no_grad():
                        # TODO: #2
                        latents = pipe.vae.encode(mel).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    """ --- Noise Generation --- """
                    # Sample random noise to add to the latents
                    noise = torch.randn_like(latents)
                    target = noise
                    
                    """ --- Sample Timestep --- """
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, 
                                            (latents.shape[0],), device=latents.device)
                    timesteps = timesteps.long()
                    
                    """ --- Noisy Latents Computation --- """
                    # Add noise to previously computed latents (fed in input to the UNet)
                self.next(self.end)    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents = noisy_latents.to(device=device)
                    
                    # TODO: #3
                    """ --- Noise Generation Procedure --- """
                    generated_noise, _ = pipe.__train__(
                        image_embeds=image_emb,
                        negative_prompt=negative_prompt,
                        num_waveforms_per_prompt=self.no_waveforms_per_prompt,
                        latents=noisy_latents,
                        guidance_scale=self.guidance_scale,
                        timesteps=timesteps,
                    )

                    # TODO: #4
                    """ --- Loss Computation --- """
                    if not use_snr_gamma:
                        """
                        Standard Mean Squared Error (MSE).
                        'reduction' parameter values:
                            - 'none': no reduction applied to the loss;
                            - 'mean': the mean of the output will be taken;
                            - 'sum': the output will be summed.
                        """ 
                        loss = torch_func.mse_loss(generated_noise.float(), target.float(), reduction="none")
                        
                        # Check if the loss has been computed at a specific timestamp (timesteps_list)
                        for idts, ts in enumerate(timesteps):
                            ts = str(ts.item())
                            if ts in ts_loss_dict:
                                ts_loss_dict[ts].append(loss[idts].mean().item())
                                                
                        # Compute the mean loss for each timesteps' loss
                        for ts_key, ts_losses in ts_loss_dict.items():
                            if len(ts_losses) != 0:
                                loss_sum = sum(ts_losses)
                                mean_loss = loss_sum / len(ts_losses)
                                accelerator.log({f"train/step": global_step,
                                                f"train/timestep_{ts_key}_loss": mean_loss})
                                # Reset the key of the specific timestep
                                ts_loss_dict[ts_key] = []
                        
                        loss = loss.mean()
                        
                    else:
                        """
                        Signal to Noise Ratio Loss.
                        """
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        mse_loss_weights = mse_loss_weights / snr

                        loss = torch_func.mse_loss(generated_noise.float(), target.float(), reduction="none")
                        
                        # Check if the loss has been computed at a specific timestamp (timesteps_list)
                        for idts, ts in enumerate(timesteps):
                            ts = str(ts.item())
                            if ts in ts_loss_dict:
                                ts_loss_dict[ts].append(loss[idts].mean().item())
                                                
                        # Compute the mean loss for each timesteps' loss
                        for ts_key, ts_losses in ts_loss_dict.items():
                            if len(ts_losses) != 0:
                                loss_sum = sum(ts_losses)
                                mean_loss = loss_sum / len(ts_losses)
                                accelerator.log({f"train/step": global_step,
                                                f"train/timestep_{ts_key}_loss": mean_loss})
                                # Reset the key of the specific timestep
                                ts_loss_dict[ts_key] = []
                        
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()
                        
                    # TODO: #5
                    # Gather losses across all processes for logging (if distributed training is used)
                    avg_loss = accelerator.gather(loss.repeat(BATCH_SIZE)).mean()
                    # Update step and epoch loss
                    train_step_loss += avg_loss.item() / self.gradient_accumulation_steps
                    epoch_loss += train_step_loss
                    
                    # Backpropagate the computed loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(pipe.img_project_model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                # Checks if the accelerator has performed an optimization step
                if accelerator.sync_gradients:
                    
                    progress_bar.update(1)
                    global_step += 1
                    # Log current step's loss on Wandb
                    accelerator.log({f"train/step": global_step, f"train/step_loss": train_step_loss})
                                    
                    # Reset step loss
                    train_step_loss = 0.0
                    
                    if global_step % self.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            print(f"Storing checkpoint!\n=========================")
                            # Check if this save would set us over the `checkpoints_total_limit`
                            if self.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.checkpoint_output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # Before saving the new checkpoint, we need remove some of the stored ones
                                if len(checkpoints) >= self.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.checkpoint_output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            # Store checkpoint
                            save_path = os.path.join(self.checkpoint_output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                    
                # Update train progress bar
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                # Check if the training loop needs to be stopped
                if global_step >= self.max_train_steps:
                    break
                elif accelerator.sync_gradients and global_step > 0 and (global_step % no_steps_per_epoch) == 0:
                    break
           
            # Compute epoch's loss and log it on Wandb
            if not skip_train:
                epoch_loss = epoch_loss / len(train_dataloader)
                accelerator.log({"train/epoch": epoch, "train/epoch_loss": epoch_loss, 
                                 "train/step": global_step})
            
                # Store epoch checkpoint
                save_path = os.path.join(self.checkpoint_output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                
                print(f"***************************************************\n"
                    f"Training epoch {epoch} completed! Starting validation....\n"
                    f"***************************************************\n"
                    )      
            else:
                print(f"***************************************************\n"
                      f"Skipped first epoch! Starting validation....\n"
                      f"***************************************************\n"
                      )
                skip_train = False        
                

            """
            Run validation after each training epoch.
            """
            # Total epoch loss
            val_fad_score = 0.0
            val_imgbind_score_am = 0.0
            val_imgbind_score_mm = 0.0
            val_kl_div = 0.0
            
            if accelerator.is_main_process:
                if val_dataloader is not None and validation_done == False:
                    
                    total_instances = val_dataloader.__len__()
                    instance_completed = 0
                    noise_scheduler.set_timesteps(self.num_inference_steps)

                    for step, batch in enumerate(val_dataloader):
                        print(f"Currently: {instance_completed}/{total_instances} instances")
                        with torch.no_grad():
                            
                            image_emb, audio_path = batch
                            audio_path = audio_path[0]
                            
                            # Retrieve image path based on image embedding (needed to log Wandb artifact)
                            img_path = dataset.__get_image_name_from_emb__(image_emb.cpu().detach())
                            gt_aud_emb = dataset.__get_aud_emb_from_path__(audio_path)
                            
                            """ --- Inference - Audio Generation --- """
                            gen_music = pipe(
                                image_embeds=image_emb,
                                negative_prompt=NEGATIVE_PROMPT,
                                num_inference_steps=self.num_inference_steps,
                                audio_length_in_s=self.audio_duration_in_seconds,
                                num_waveforms_per_prompt=self.no_waveforms_per_prompt,
                                generator=generator,
                                guidance_scale=self.guidance_scale,
                            ).audios

                            # Empty folder after 500 music files have been stored [avoid storing too many files at once]
                            if self.eval_audios == 500:
                                tu.empty_folder(VAL_AUDIO_DIR)
                                self.eval_audios = 0

                            if self.eval_audios < self.max_eval_audios:
                                generated_audio_path = VAL_AUDIO_DIR + f"val_audio_{self.eval_audios}.wav"
                                scipy.io.wavfile.write(generated_audio_path, rate=16000, data=gen_music[0])
                                self.eval_audios += 1
                            
                                # Create Wandb artifact with artwork and generated audio, and log it
                                wandb_artifact = create_artifact(img_path, generated_audio_path, completed_val_steps)
                                accelerator.log(wandb_artifact)
                                # accelerator.log(wandb_artifact, step=completed_val_steps)
                            
                            """ --- Metrics Computation ---"""
                            # KL Divergence
                            kl_div = tu.compute_kl_div(audio_path, generated_audio_path)
                            
                            # ImageBind Score
                            imgbind_score_am, imgbind_score_mm = tu.compute_imagebind_score(image_embedding=image_emb,
                                                                                            gt_audio_emb=gt_aud_emb, 
                                                                                            generated_audio=gen_music,
                                                                                            imagebind_model=imagebind, 
                                                                                            tmp_gen_audio_dir=TMP_DIR_GT)
                            
                            # FAD Score
                            if using_cuda:
                                
                                # Copy files before computing FAD score
                                shutil.copy(audio_path, TMP_DIR_GT)
                                shutil.copy(generated_audio_path, TMP_DIR_GEN)
                                
                                try:
                                    fad_score = tu.calculate_fad(ground_truth_dir_path=TMP_DIR_GT,
                                                                 generated_audio_dir_path=TMP_DIR_GEN,
                                                                 load_from_local=True)
                                    val_fad_score += fad_score
                                except Exception as _:
                                    fad_score=None
                                
                                # Remove files after computing FAD score
                                tu.empty_folder(TMP_DIR_GT)
                                tu.empty_folder(TMP_DIR_GEN)
                            
                            else:
                                fad_score = None
                            
                            val_imgbind_score_am += imgbind_score_am
                            val_imgbind_score_mm += imgbind_score_mm
                            val_kl_div += kl_div
                            
                            logger.info("***** Validation Metrics *****")
                            logger.info(f"KL-Divergence = {kl_div}")
                            logger.info(f"ImageBind Score Artwork-Music = {imgbind_score_am}")
                            logger.info(f"ImageBind Score Music-Music = {imgbind_score_mm}")
                            if using_cuda:
                                logger.info(f"FAD Score = {fad_score}")
                            
                            # Log validation metrics on Wandb
                            accelerator.log({"val/step": completed_val_steps, 
                                             "val/kl_div": kl_div, 
                                             "val/fad_score": fad_score, 
                                             "val/imagebind_score_am": imgbind_score_am,
                                             "val/imagebind_score_mm": imgbind_score_mm})
                            
                            instance_completed += 1
                            completed_val_steps += 1
                            
                    # Validation completed
                    validation_done = True
                    print(f"***************************************************\n"
                          f"Validation for epoch {epoch} completed!\n"
                          f"***************************************************\n"
                          )
                    
                    # Log average validation metrics after each validation
                    # If no fad score was computed, we have val_fad_score == 0.0
                    val_fad_score = val_fad_score / len(val_dataloader)
                    val_imgbind_score_am = val_imgbind_score_am / len(val_dataloader)
                    val_imgbind_score_mm = val_imgbind_score_mm / len(val_dataloader)
                    val_kl_div = val_kl_div / len(val_dataloader)
                    
                    accelerator.log({"val/step": completed_val_steps, 
                                     "val/avg_fad_score": val_fad_score,
                                     "val/avg_imgbind_score_am": val_imgbind_score_am,
                                     "val/avg_imgbind_score_mm": val_imgbind_score_mm,
                                     "val/avg_kl_div": val_kl_div})
                    
                    # Reset these metrics values after logging them
                    val_fad_score = 0.0
                    val_imgbind_score_am = 0.0
                    val_imgbind_score_mm = 0.0
                    val_kl_div = 0.0
        
        print('the data artifact is still: %s' % self.my_var)
        self.next(self.end)

    @step
    def train(self):
        print('Training model...')
        print('dataset is:', self.dataset)
        self.next(self.end)

    @step
    def end(self):
        print("Done training!")


if __name__ == "__main__":
    Art2MusTrainFlow()
