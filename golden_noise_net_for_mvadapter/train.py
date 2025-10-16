import argparse
import math
import os
import time
from math import prod

import numpy as np
import torch
from torch import nn, optim


from read_data import LatentsData
from torch.utils.data import DataLoader
from golden_noise_net import EDN


device = torch.device("cuda")


def save_model( model, path):
    """保存NPNet模型的参数"""
    # 创建需要保存的参数字典
    state_dict = {
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
    }

    # 保存参数字典到文件
    torch.save(state_dict, path)
    print(f"模型参数已保存到 {path}")
def calculate_average(lst):
    if len(lst) == 0:
        return 0  # 如果列表为空，返回0或抛出异常
    total_sum = sum(lst)
    average = total_sum / len(lst)
    return average

parser = argparse.ArgumentParser()
parser.add_argument('--train_root_dir', default="D:\\MV-Adapter_golden_noise\\MV-Adapter-main\\MV-Adapter-main\\inversion_mvadapter_objaverse1765_elevation0_outputs_select", type=str)

parser.add_argument('--test_root_dir', default='', type=str)

parser.add_argument('--pretrained_path', default="", type=str)
parser.add_argument("--epoch", default=600, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--num_frames", default=6, type=int)
args = parser.parse_args()

golden_noise_net = EDN(args.pretrained_path)
train_datasets =LatentsData(root_dir=args.train_root_dir,)
train_dataloder = DataLoader(train_datasets, batch_size=args.batch_size, shuffle= True, num_workers=0,drop_last=False )
test_datasets =LatentsData(root_dir=args.test_root_dir, )
test_dataloder = DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=0,drop_last=False )
loss_fn = nn.SmoothL1Loss(reduction="mean").to(device)
optimizer = optim.Adam(golden_noise_net.parameters(), lr=3e-4,)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200,gamma=0.8, last_epoch=-1)

for i in range(args.epoch):
    print(f"----------第{i + 1}轮训练开始----------")
    print(f"Epoch {i + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}")
    batch = 0
    inf= 0
    nan = 0
    golden_noise_net.train()
    for data in train_dataloder:
                    gt_latents, org_latents, image_latents = data
                    gt_latents = gt_latents/36.4351
                    org_latents = org_latents/36.4351
                    image_latents = image_latents.repeat(1, 6, 1, 1, 1)
                    org_latents = org_latents.to(device)
                    gt_latents = gt_latents.to(device).to(torch.float32)
                    gt_noise = gt_latents - org_latents
                    golden_noise = golden_noise_net(org_latents, image_latents)
                    optimizer.zero_grad()
                    loss = loss_fn(gt_noise, golden_noise)
                    batch = batch + 1
                    loss.backward()
                    optimizer.step()
    golden_noise_net.eval()
    with torch.no_grad():
        total_test_loss =[]
        for data in test_dataloder:
            gt_latents, org_latents, image_latents= data
            gt_latents = gt_latents/36.4351
            org_latents = org_latents/36.4351

            image_latents = image_latents.repeat(1, 6, 1, 1, 1)

            org_latents = org_latents.to(device)

            gt_latents = gt_latents.to(device).to(torch.float32)

            golden_noise = golden_noise_net(org_latents, image_latents)
            golden_latents = (golden_noise+org_latents)

            loss = loss_fn(gt_latents, golden_latents)
            if not (math.isinf(loss) or math.isnan(loss)):
               total_test_loss.append(loss)
        test_loss = calculate_average(total_test_loss)
        print(f"epoch_{i+1}对应test_loss={test_loss*1}")
        print(f"inf={inf}")
        print(f"nan={nan}")
        if i == 0:
            best_test_loss = test_loss
            save_model(golden_noise_net, f"./best.pth")
        else:
            if test_loss < best_test_loss:
                best_test_loss =test_loss
                save_model(golden_noise_net, f"./best.pth")
    scheduler.step()

# writer.close()
print("训练完成")
print(best_test_loss)







