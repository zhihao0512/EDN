import json
import os

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import *
from torch import nn

from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel
from euler_discrete_inverse import EulerDiscreteInverseScheduler

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
# Copied from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion.StableVideoDiffusionPipeline
class StableVideo3DDiffusionPipeline(StableVideoDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,

        )
        self.inv_scheduler = EulerDiscreteInverseScheduler(
         beta_end=0.012,
         beta_schedule="scaled_linear",
         beta_start=0.00085,
         # clip_sample=False,
         final_sigmas_type="zero",
         interpolation_type="linear",
         num_train_timesteps=1000,
         prediction_type="v_prediction",
         rescale_betas_zero_snr=False,
         # set_alpha_to_one=False,
         sigma_max=700.0,
         sigma_min=0.002,
         # skip_prk_steps=True,
         steps_offset=1,
         timestep_spacing="leading",
         timestep_type="continuous",
         trained_betas=None,
         use_karras_sigmas=True
)
        self.next_scheduler = EulerDiscreteScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            # clip_sample=False,
            final_sigmas_type="zero",
            interpolation_type="linear",
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            rescale_betas_zero_snr=False,
            # set_alpha_to_one=False,
            sigma_max=700.0,
            sigma_min=0.002,
            # skip_prk_steps=True,
            steps_offset=1,
            timestep_spacing="leading",
            timestep_type="continuous",
            trained_betas=None,
            use_karras_sigmas=True
)
    def _get_add_time_ids(
        self,
        noise_aug_strength: float,
        polars_rad: List[float],
        azimuths_rad: List[float],
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        cond_aug = torch.tensor([noise_aug_strength]*len(polars_rad), dtype=dtype).repeat(batch_size * num_videos_per_prompt, 1)
        polars_rad = torch.tensor(polars_rad, dtype=dtype).repeat(batch_size * num_videos_per_prompt, 1)
        azimuths_rad = torch.tensor(azimuths_rad, dtype=dtype).repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            cond_aug = torch.cat([cond_aug, cond_aug])
            polars_rad = torch.cat([polars_rad, polars_rad])
            azimuths_rad = torch.cat([azimuths_rad, azimuths_rad])

        add_time_ids = [cond_aug, polars_rad, azimuths_rad]

        return add_time_ids

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],

        polars_rad: List[float],
        azimuths_rad: List[float],
        triangle_cfg_scaling: bool = True,

        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 2.5,
        noise_aug_strength: float = 1e-5,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        last_folder = None
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        image_need = image
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents_need = self._encode_vae_image(
            image_need,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents_need = image_latents_need.to(image_embeddings.dtype)
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        image_latents_need = image_latents_need.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        _, image_latenets_without_noise = torch.split(image_latents_need, 1, dim=0)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            noise_aug_strength,
            polars_rad,
            azimuths_rad,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = [a.to(device) for a in added_time_ids]

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)
        inv_timesteps, inv_num_inference_steps = retrieve_timesteps(self.inv_scheduler, num_inference_steps, device, None, sigmas)
        next_timesteps, next_num_inference_steps = retrieve_timesteps(self.next_scheduler, num_inference_steps, device, None, sigmas)
        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        org_latents = latents
        latent_shape = latents.shape
        latents_dtype = latents.dtype
        # 8. Prepare guidance scale
        if triangle_cfg_scaling:
            # Triangle CFG scaling; the last view is input condition
            guidance_scale = torch.cat([
                torch.linspace(min_guidance_scale, max_guidance_scale, num_frames//2 + 1)[1:].unsqueeze(0),
                torch.linspace(max_guidance_scale, min_guidance_scale, num_frames - num_frames//2 + 1)[1:].unsqueeze(0)
            ], dim=-1)
        else:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        inv_guidance_scale = torch.linspace(0.0, 0.0, num_frames).unsqueeze(0)
        inv_guidance_scale = inv_guidance_scale.to(device, latents.dtype)
        inv_guidance_scale = inv_guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        inv_guidance_scale = _append_dims(inv_guidance_scale, latents.ndim)

        inf_guidance_scale = torch.linspace(6.0, 2.5, num_frames).unsqueeze(0)
        inf_guidance_scale = inf_guidance_scale.to(device, latents.dtype)
        inf_guidance_scale = inf_guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        inf_guidance_scale = _append_dims(inf_guidance_scale, latents.ndim)

        # 9. Denoising loop
        target_mean_list = []
        target_std_list = []
        target_mean_list_1 = []
        target_std_list_1 = []
        target_mean_list_2 = []
        target_std_list_2 = []
        target_mean_list_3 = []
        target_std_list_3 = []
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < 16:
                    target_mean = latents.mean().item()
                    target_std = latents.std().item()
                    target_mean_list_1.append(target_mean)
                    target_std_list_1.append(target_std)
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    target_mean = latent_model_input.mean().item()
                    target_std = latent_model_input.std().item()
                    target_mean_list.append(target_mean)
                    target_std_list.append(target_std)
                    # Concatenate image_latents over channels dimension
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        target_mean = noise_pred_uncond.mean().item()
                        target_std = noise_pred_uncond.std().item()
                        target_mean_list_2.append(target_mean)
                        target_std_list_2.append(target_std)
                        target_mean = noise_pred_cond.mean().item()
                        target_std = noise_pred_cond.std().item()
                        target_mean_list_3.append(target_mean)
                        target_std_list_3.append(target_std)
                        noise_pred = noise_pred_uncond + inf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < 16:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.inv_scheduler.scale_model_input(latent_model_input, t)
                    old_mean = latent_model_input.mean().item()
                    old_std = latent_model_input.std().item()
                    latent_model_input = (latent_model_input - old_mean) * (target_std_list[15 - i] / old_std) + target_mean_list[15 - i]
                    # Concatenate image_latents over channels dimension
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        timesteps[15 - i],
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        old_mean = noise_pred_uncond.mean().item()
                        old_std = noise_pred_uncond.std().item()
                        noise_pred_uncond = (noise_pred_uncond - old_mean) * (target_std_list_2[15 - i] / old_std) + \
                                             target_mean_list_2[15 - i]
                        old_mean = noise_pred_cond.mean().item()
                        old_std = noise_pred_cond.std().item()
                        noise_pred_cond = (noise_pred_cond - old_mean) * (target_std_list_3[15 - i] / old_std) + \
                                            target_mean_list_3[15 - i]
                        noise_pred = noise_pred_uncond + inv_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
                    old_mean = latents.mean().item()
                    old_std = latents.std().item()
                    latents = (latents - old_mean) * (target_std_list_1[15 - i] / old_std) + target_mean_list_1[15 - i]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        gt_latents = latents

        loss_fn = nn.MSELoss(reduction="mean").to(device)
        loss = loss_fn(org_latents, latents)
        print(loss)


        self.maybe_free_model_hooks()

        return org_latents,gt_latents,image_latenets_without_noise
