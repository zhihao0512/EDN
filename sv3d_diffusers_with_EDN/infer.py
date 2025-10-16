import os
import argparse

from PIL import Image
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils import export_to_gif

from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline
from golden_noise_net import EDN
SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs", type=str, help="Output filepath")
    parser.add_argument("--input_paths", default="", type=str, help="Image filepath")
    parser.add_argument("--elevation", default=0, type=float, help="Camera elevation of the input image")
    parser.add_argument("--half_precision", default=True, help="Use fp16 half precision")
    parser.add_argument("--seed", default=23, type=int, help="Random seed")
    parser.add_argument('--pretrained_path', default="", type=str)#Place the trained EDN weights here
    args = parser.parse_args()

    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(SV3D_DIFFUSERS, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(SV3D_DIFFUSERS, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(SV3D_DIFFUSERS, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SV3D_DIFFUSERS, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(SV3D_DIFFUSERS, subfolder="feature_extractor")

    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, 
        unet=unet, vae=vae,
        scheduler=scheduler,
    )
    golden_noise_net = EDN(args.pretrained_path)
    num_frames, sv3d_res = 21, 576
    elevation = args.elevation
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    pipeline = pipeline.to("cuda")
    new_addresses = []
    seed = args.seed
    for root, dirs, files in os.walk(args.input_paths):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            new_address = os.path.join(dir_path, "020.png")
            new_addresses.append(new_address)
    for input_path in new_addresses:
        # 去掉文件名部分
        dir_part = os.path.dirname(input_path)
        # 提取最后一个文件夹名称
        last_folder = os.path.basename(dir_part)
        folder_dir = os.path.join(args.output_dir,last_folder)
        if os.path.exists(folder_dir) and os.path.isdir(folder_dir):
            print(f"输出文件夹{last_folder}存在！")
        else:
            print(f"随机数种子为{seed}")
            print(f"方位角为{elevation}")
            elevations_deg = [elevation] * num_frames
            polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16 if args.half_precision else torch.float32,
                                    enabled=True):
                    image = Image.open(input_path)
                    image.load()  # required for `.split()`
                    if len(image.split()) == 4:  # RGBA
                        input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
                        input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
                    else:
                        input_image = image
                    input_image = input_image.resize((sv3d_res, sv3d_res))
                    video_frames = pipeline(
                        input_image,
                        height=sv3d_res,
                        width=sv3d_res,
                        num_frames=num_frames,
                        decode_chunk_size=8,  # smaller to save memory
                        polars_rad=polars_rad,
                        azimuths_rad=azimuths_rad,
                        generator=torch.manual_seed(seed) if seed >= 0 else None,
                        golden_noise_net=golden_noise_net,
                    )
            vid = video_frames[0]
            os.makedirs(args.output_dir, exist_ok=True)
            image_folder = os.path.join(args.output_dir, last_folder)
            os.makedirs(image_folder, exist_ok=True)
            for i in range(len(vid)):
                vid[i].save(
                    f"{args.output_dir}/{last_folder}/{str(i).zfill(3)}.png")
            rf_file_path = os.path.join(image_folder, "reference.png")
            input_image.save(rf_file_path)

if __name__ == "__main__":
    main()
