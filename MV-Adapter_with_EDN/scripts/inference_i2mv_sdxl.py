import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)
from golden_noise_net import EDN
def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterI2MVSDXLPipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_i2mv_sdxl.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe


def remove_bg(image, net, transform, device):
    image_size = image.size
    input_images = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    # if H > W:
    #     W = int(W * (height * 0.9) / H)
    #     H = int(height * 0.9)
    # else:
    #     H = int(H * (width * 0.9) / W)
    #     W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def run_pipeline(
    pipe,
    num_views,
    text,
    image,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    folder_name,
    remove_bg_fn=None,
    reference_conditioning_scale=1.0,
    negative_prompt="",
    lora_scale=1.0,
    device="cuda",
    golden_noise_net=None,
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 0, 0],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, width
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    # Prepare image
    reference_image = Image.open(image) if isinstance(image, str) else image
    if reference_image.mode == "RGBA":
        reference_image = preprocess_image(reference_image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=reference_image,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        folder_name=folder_name,
        **pipe_kwargs,
        golden_noise_net=golden_noise_net,
    )

    return images, reference_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
    )
    parser.add_argument("--output", type=str, default="")
    # Extra
    parser.add_argument("--remove_bg", default=True, help="Remove background")
    parser.add_argument('--pretrained_path', default="", type=str)#Place the trained EDN weights here
    args = parser.parse_args()

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )
    golden_noise_net = EDN(args.pretrained_path)
    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(args.device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, args.device)
    else:
        remove_bg_fn = None
    seed = args.seed
    for folder_name in os.listdir(args.image_folder):
        output_folder_path = os.path.join(args.output, folder_name)
        if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
            print(f"输出文件夹{output_folder_path}存在！")
        else:
            print(folder_name)
            folder_path = os.path.join(args.image_folder, folder_name)
            image_path = os.path.join(folder_path, "000.png")
            images, reference_image = run_pipeline(
                pipe,
                num_views=args.num_views,
                text="",
                image=image_path,
                height=768,
                width=768,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=seed,
                lora_scale=args.lora_scale,
                reference_conditioning_scale=args.reference_conditioning_scale,
                negative_prompt=args.negative_prompt,
                device=args.device,
                remove_bg_fn=remove_bg_fn,
                folder_name=folder_name,
                golden_noise_net=golden_noise_net,
            )
            output_folder_path = os.path.join(args.output, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            # 遍历图像列表并保存
            for i, image in enumerate(images):
                # 生成文件名，格式为 000, 001, 002, ...
                file_name = f"{i:03d}.png"  # :03d 表示用 3 位数字，不足补零
                file_path = os.path.join(output_folder_path, file_name)

                # 保存图像
                image.save(file_path)
            rf_file_path = os.path.join(output_folder_path, "reference.png")
            reference_image.save(rf_file_path)
