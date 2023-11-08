import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
import utils.ptp_utils as ptp_utils
from diffusers import DDIMInverseScheduler,DPMSolverMultistepInverseScheduler
class Inversion:

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        #print(f"next_step:timestep {timestep} next {next_timestep}")
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context,cond=True,both=False):
        added_cond_id=1 if cond else 0
            
        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance=False
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        if both is False:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds[added_cond_id].unsqueeze(0), "time_ids": self.add_time_ids[added_cond_id].unsqueeze(0)}
        else:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=context,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]#1,4,128,128
        #old:noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * self.model.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def init_prompt(
        self,
        prompt: str,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None
    ):
        original_size = original_size or (1024, 1024)
        target_size = target_size or (1024, 1024)
        # 3. Encode input prompt
        do_classifier_free_guidance=True
        (
            prompt_embeds,# 2,77,2048
            negative_prompt_embeds,# 2,77,2048
            pooled_prompt_embeds,# 2,1280
            negative_pooled_prompt_embeds,# 2,1280
        ) = self.model.encode_prompt_not_zero_uncond(
            prompt,
            self.model.device,
            1,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )
        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.model._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        self.add_text_embeds = add_text_embeds.to(self.device)
        self.add_time_ids = add_time_ids.to(self.device).repeat(1 * 1, 1)

        self.prompt_embeds=prompt_embeds
        self.negative_prompt_embeds=negative_prompt_embeds
        self.pooled_prompt_embeds=pooled_prompt_embeds
        self.negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
        self.prompt = prompt
        self.context=prompt_embeds

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(self.generator, self.eta)
        if isinstance(self.inverse_scheduler,DDIMInverseScheduler):
            extra_step_kwargs.pop("generator")
        for i in range(self.num_ddim_steps):
            #print(i)
            use_inv_sc=True
            if use_inv_sc:
                t = self.inverse_scheduler.timesteps[i]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings,cond=True)


                #latent = self.next_step(noise_pred, t, latent)
                latent = self.inverse_scheduler.step(noise_pred, t, latent, **extra_step_kwargs, return_dict=False)[0]
            else:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings,cond=True)
                latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image) #img:# 512,512,3          latent:1,4,64,64
        image_rec = self.latent2image(latent) # 512,512,3
        ddim_latents = self.ddim_loop(latent.to(self.model.unet.dtype)) # [1,4,64,64]*steps
        return image_rec, ddim_latents

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=10, early_stop_epsilon=1e-5,
               verbose=True,train_free=True,all_latents=True):
        self.init_prompt(prompt)
        #ptp_utils.register_attention_control(self.model, None)
        image_gt = load_1024(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Done.")
        if train_free is False:
            uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
            return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        else:
            if all_latents:
                return (image_gt, image_rec), ddim_latents[-1], ddim_latents,self.prompt_embeds[1].unsqueeze(0),self.pooled_prompt_embeds
            else:
                return (image_gt, image_rec), ddim_latents[-1], self.prompt_embeds[1].unsqueeze(0),self.pooled_prompt_embeds


    def __init__(self, model,num_ddim_steps,generator=None,scheduler_type="DDIM"):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.num_ddim_steps=num_ddim_steps
        if scheduler_type == "DDIM":
            self.inverse_scheduler=DDIMInverseScheduler.from_config(self.model.scheduler.config)
            self.inverse_scheduler.set_timesteps(num_ddim_steps)
        elif scheduler_type=="DPMSolver":
            self.inverse_scheduler=DPMSolverMultistepInverseScheduler.from_config(self.model.scheduler.config)
            self.inverse_scheduler.set_timesteps(num_ddim_steps)
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.model.vae.to(dtype=torch.float32)
        self.prompt = None
        self.context = None
        self.device=self.model.unet.device
        self.generator=generator
        self.eta=0.0

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def load_1024_mask(image_path, left=0, right=0, top=0, bottom=0,target_H=128,target_W=128):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, np.newaxis]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image=image.squeeze()
    image = np.array(Image.fromarray(image).resize((target_H, target_W)))
    return image

def load_1024(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((1024, 1024)))
    return image