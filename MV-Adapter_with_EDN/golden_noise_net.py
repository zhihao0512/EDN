import einops
import torch
from torch import nn
from model import NoiseTransformer, SVDNoiseUnet
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.unets.unet_spatio_temporal_condition import *
from attention import MySelfAttention,MyMutualAttention
from math import prod
from resnet import ResnetEncoder
from normnet_decoder import NormDecoder
class EDN(nn.Module):
    def __init__(self, pretrained_path=None, device='cuda') -> None:
        super(NPNet, self).__init__()
        self.device = device
        self.pretrained_path = pretrained_path
        self.layernorm4 = nn.LayerNorm(4,device=device,dtype=torch.float32)
        self.layernorm144 = nn.LayerNorm(144,device=device,dtype=torch.float32)

        (self.encoder,
         self.decoder,
         )= self.get_model()

    def get_model(self):
        encoder = ResnetEncoder(
            num_layers=18, pretrained=None, num_input_images=1
        ).to(self.device).to(torch.float32)
        decoder = NormDecoder(encoder.num_ch_enc, num_output_channels=4).to(self.device).to(torch.float32)
        if self.pretrained_path is not None:
            # 加载保存的参数字典
            state_dict = torch.load(self.pretrained_path, map_location=self.device)

            # 加载各个组件的参数
            encoder.load_state_dict(state_dict['encoder'])
            decoder.load_state_dict(state_dict['decoder'])

            print("Load Pretrained Weights Successfully!")
        else:
            assert ("No Pretrained Weights Found!")
        return encoder,decoder


    def forward(self, org_latents, image_latents):

        batch_size = org_latents.shape[0]
        org_latents = einops.rearrange(org_latents, "b t c h w ->(b t)  c h w").to(torch.float32)
        image_latents = einops.rearrange(image_latents, "b t c h w ->(b t) c h w").to(torch.float32)
        encoder_hidden_states=torch.cat((org_latents, image_latents), dim=1)
        features = self.encoder(encoder_hidden_states)
        encoder_hidden_states = self.decoder(features)
        encoder_hidden_states =encoder_hidden_states[("norm", 0)]
        golden_noise =   encoder_hidden_states
        golden_noise = einops.rearrange(golden_noise, " (b t) c h w->b t c h w",b=batch_size).to(torch.float32)

        return golden_noise