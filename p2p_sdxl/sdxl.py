
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import torch.nn.functional as F

from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin 
from diffusers.models.attention_processor import ( AttnProcessor2_0, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor, XFormersAttnProcessor, ) 
from diffusers.utils import ( is_accelerate_available, is_accelerate_version, logging, randn_tensor, replace_example_docstring, ) 
from diffusers.pipelines.pipeline_utils import DiffusionPipeline 
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput 


from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class sdxl(StableDiffusionXLPipeline): 
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,same_init=False):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if same_init is True:
            if latents is None:
                latents = randn_tensor((1,*shape[1:]), generator=generator, device=device, dtype=dtype).expand(shape).to(device)
            else:
                if batch_size>1 and latents.shape[0]==1:
                    latents=latents.expand(shape).to(device)
                else:
                    latents = latents.to(device)
        else: 
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    @torch.no_grad()
    def __call__(
        self,
        controller=None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        p2p=None,
        same_init=False,
        null_inversion=False,
        **kwargs
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                TODO
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                TODO
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                TODO

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # -1. Controller
        if controller is not None:
            ptp_utils.register_attention_control(self, controller)
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        if null_inversion is False:
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
        else:
            negative_pooled_prompt_embeds=[]
            for negative_prompt_embeds_per_stage in negative_prompt_embeds:
                negative_pooled_prompt_embeds.append(negative_prompt_embeds_per_stage[:,0,768:])
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds[0],
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,# 2,77,2048
            negative_prompt_embeds,# 2,77,2048
            pooled_prompt_embeds,# 2,1280
            negative_pooled_prompt_embeds,# 2,1280
        ) = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(#2,4,64,64
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            same_init=same_init
        )
        latent_store=latents[0].clone().detach()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            if isinstance(negative_prompt_embeds,List) is False:
                all_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                all_add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                all_prompt_embeds = all_prompt_embeds.to(device)
                all_add_text_embeds = all_add_text_embeds.to(device)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)


        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if isinstance(negative_prompt_embeds,List):
                    if do_classifier_free_guidance:
                        all_prompt_embeds = torch.cat([negative_prompt_embeds[i].repeat(2,1,1), prompt_embeds], dim=0)
                        all_prompt_embeds = all_prompt_embeds.to(device)
                        all_add_text_embeds = torch.cat([negative_pooled_prompt_embeds[i].repeat(2,1), add_text_embeds], dim=0)
                        all_add_text_embeds = all_add_text_embeds.to(device)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": all_add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=all_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                kwargs.update(extra_step_kwargs)
                latents=exec_classifier_free_guidance(self,
                                                      latents,
                                                      controller,
                                                      t,
                                                      guidance_scale,
                                                      do_classifier_free_guidance,
                                                      noise_pred,
                                                      guidance_rescale,
                                                      i=i,
                                                      **kwargs)


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(latents.dtype)
            self.vae.decoder.conv_in.to(latents.dtype)
            self.vae.decoder.mid_block.to(latents.dtype)
        else:
            latents = latents.float()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if p2p is True:
            return image,latent_store
        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
    
    def encode_prompt(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            
            if isinstance(negative_prompt, List):
                all_negative_prompt=[]
                for negative_prompt_ in negative_prompt:
                    negative_prompt_embeds_list = []
                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                        # textual inversion: procecss multi-vector tokens if necessary
                        if isinstance(self, TextualInversionLoaderMixin):
                            uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                        max_length = prompt_embeds.shape[1]
                        uncond_input = tokenizer(
                            uncond_tokens,
                            padding="max_length",
                            max_length=max_length,
                            truncation=True,
                            return_tensors="pt",
                        )

                        negative_prompt_embeds = text_encoder(
                            uncond_input.input_ids.to(device),
                            output_hidden_states=True,
                        )
                        # We are only ALWAYS interested in the pooled output of the final text encoder
                        negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                        negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                        if do_classifier_free_guidance:
                            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                            seq_len = negative_prompt_embeds.shape[1]

                            negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                            negative_prompt_embeds = negative_prompt_embeds.view(
                                batch_size * num_images_per_prompt, seq_len, -1
                            )

                            # For classifier free guidance, we need to do two forward passes.
                            # Here we concatenate the unconditional and text embeddings into a single batch
                            # to avoid doing two forward passes

                        negative_prompt_embeds_list.append(negative_prompt_embeds)

                    negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
                    all_negative_prompt.append(negative_prompt_embeds)
                negative_prompt_embeds=all_negative_prompt
            else:
                negative_prompt_embeds_list = []
                for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                    # textual inversion: procecss multi-vector tokens if necessary
                    if isinstance(self, TextualInversionLoaderMixin):
                        uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    negative_prompt_embeds = text_encoder(
                        uncond_input.input_ids.to(device),
                        output_hidden_states=True,
                    )
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                    if do_classifier_free_guidance:
                        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                        seq_len = negative_prompt_embeds.shape[1]

                        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                        negative_prompt_embeds = negative_prompt_embeds.view(
                            batch_size * num_images_per_prompt, seq_len, -1
                        )

                        # For classifier free guidance, we need to do two forward passes.
                        # Here we concatenate the unconditional and text embeddings into a single batch
                        # to avoid doing two forward passes

                    negative_prompt_embeds_list.append(negative_prompt_embeds)

                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        if num_images_per_prompt>1:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
        if negative_pooled_prompt_embeds.shape[0]==1 and bs_embed!=1:
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.repeat(bs_embed,1)
        if negative_prompt_embeds.shape[0]==1 and bs_embed!=1:
            negative_prompt_embeds=negative_prompt_embeds.repeat(bs_embed,1,1)
          
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def encode_prompt_not_zero_uncond(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_images_per_prompt, seq_len, -1
                    )

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def slerp(val, low, high):
    """ taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)


def dilate(image, kernel_size, stride=1, padding=0):
    """
    Perform dilation on a binary image using a square kernel.
    """
    # Ensure the image is binary
    assert image.max() <= 1 and image.min() >= 0
    
    # Get the maximum value in each neighborhood
    dilated_image = F.max_pool2d(image, kernel_size, stride, padding)
    
    return dilated_image

def exec_classifier_free_guidance(model,latents,controller,t,guidance_scale,
                                  do_classifier_free_guidance,noise_pred,guidance_rescale,
                                  prox=None, quantile=0.75,image_enc=None, recon_lr=0.1, recon_t=400,recon_end_t=0,
                                  inversion_guidance=False, reconstruction_guidance=False,x_stars=None, i=0,
                                    use_localblend_mask=False,
                                  save_heatmap=True,**kwargs):
    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if prox is None and inversion_guidance is True:
            prox = 'l1'
        step_kwargs = {
            'ref_image': None,
            'recon_lr': 0,
            'recon_mask': None,
        }
        mask_edit = None
        if prox is not None:
            if prox == 'l1':
                score_delta = (noise_pred_text - noise_pred_uncond).float()
                if quantile > 0:
                    threshold = score_delta.abs().quantile(quantile)
                else:
                    threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                score_delta -= score_delta.clamp(-threshold, threshold)
                score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
                score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                    step_kwargs['ref_image'] = image_enc
                    step_kwargs['recon_lr'] = recon_lr
                    score_delta_norm=score_delta.abs()
                    score_delta_norm=(score_delta_norm - score_delta_norm.min ()) / (score_delta_norm.max () - score_delta_norm.min ())
                    import seaborn as sns
                    import matplotlib.pyplot as plt

                    mask_edit = (score_delta.abs() > threshold).float()
                    if save_heatmap and i%10==0:
                        for kk in range(4):
                            # sns.heatmap(score_delta_norm[1][kk].clone().cpu(), cmap='coolwarm')
                            # plt.savefig(f'./vis/heatmap1_inversion_{i}_{kk}.png')
                            # plt.clf()
                            # sns.heatmap(score_delta.abs()[1][kk].clone().cpu(), cmap='coolwarm')
                            # plt.savefig(f'./vis/heatmap1_inversion_old_{i}_{kk}.png')
                            # plt.clf()
                            sns.heatmap(mask_edit[1][kk].clone().cpu(), cmap='coolwarm')
                            plt.savefig(f'./vis/prox_inv/heatmap1_mask_{i}_{kk}.png')
                            plt.clf()
                    if kwargs.get('dilate_mask', 2) > 0:
                        radius = int(kwargs.get('dilate_mask', 2))
                        mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                    if save_heatmap and i%10==0:
                        for kk in range(4):
                            sns.heatmap(mask_edit[1][kk].clone().cpu(), cmap='coolwarm')
                            plt.savefig(f'./vis/prox_inv/heatmap1_mask_dilate_{i}_{kk}.png')
                            plt.clf()
                    step_kwargs['recon_mask'] = 1 - mask_edit
            elif prox == 'l0':
                score_delta = (noise_pred_text - noise_pred_uncond).float()
                if quantile > 0:
                    threshold = score_delta.abs().quantile(quantile)
                else:
                    threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                score_delta -= score_delta.clamp(-threshold, threshold)
                if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                    step_kwargs['ref_image'] = image_enc
                    step_kwargs['recon_lr'] = recon_lr
                    mask_edit = (score_delta.abs() > threshold).float()
                    if kwargs.get('dilate_mask', 2) > 0:
                        radius = int(kwargs.get('dilate_mask', 2))
                        mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                    step_kwargs['recon_mask'] = 1 - mask_edit
            else:
                raise NotImplementedError
            noise_pred = (noise_pred_uncond + guidance_scale * score_delta).to(model.unet.dtype)
        else:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
    if do_classifier_free_guidance and guidance_rescale > 0.0:
    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
    if reconstruction_guidance:
        kwargs.update(step_kwargs)
    latents = model.scheduler.step(noise_pred, t, latents, **kwargs, return_dict=False)[0]
    if mask_edit is not None and inversion_guidance and (recon_t > recon_end_t and t < recon_t) or (recon_t < recon_end_t and t > -recon_t):
        if use_localblend_mask:
            assert hasattr(controller,"local_blend")
            if save_heatmap and i%10==0:
                sns.heatmap(controller.local_blend.mask[0][0].clone().cpu(), cmap='coolwarm')
                plt.savefig(f'./vis/prox_inv/heatmap0_localblendmask_{i}.png')
                plt.clf()
                sns.heatmap(controller.local_blend.mask[1][0].clone().cpu(), cmap='coolwarm')
                plt.savefig(f'./vis/prox_inv/heatmap1_localblendmask_{i}.png')
                plt.clf()
            local_blend_mask=controller.local_blend.mask.float()
            local_blend_mask[0]=local_blend_mask[1]
            recon_mask=1-local_blend_mask.expand_as(latents)
        else:
            recon_mask = 1 - mask_edit
        latents = latents - recon_lr * (latents - x_stars[len(x_stars)-i-2].expand_as(latents)) * recon_mask

    # controller
    if controller is not None:
        latents = controller.step_callback(latents)
    return latents.to(model.unet.dtype)