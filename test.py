import numpy as np
from evaluator import evaluation_model
from dataloader import iclevrLoader
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import csv
import random
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from diffusers import UNet2DConditionModel, UNet2DModel
from torch.utils.data import SubsetRandomSampler
import os
from accelerate import Accelerator
from diffusers import DDPMScheduler
#from dataloader_b import encoding_dict
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DiffusionPipeline
import torchvision
import time

time_step = 1200
beta = torch.linspace(1e-4, 2e-3, time_step) # generate num of {time_step} points
alpha = 1 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0) # product of all alphas
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod) 
sqrt_oneminus_alpha_cumprod = torch.sqrt(1-alpha_cumprod)   

# do not minus last element in alpha_cumprod
one_minus_alpha_cumprod_t_minus_1 = torch.cat(((torch.tensor(1).unsqueeze(0)), (1-alpha_cumprod)[:-1])) 
one_minus_alpha_cumprod = (1 - alpha_cumprod)
sqrt_variance = torch.sqrt((beta*(one_minus_alpha_cumprod_t_minus_1/one_minus_alpha_cumprod)))

# train is to add noise, compute xt from xt-1
def compute_xt(args, data, rand_t, noise):
    coef_x0 = []
    coef_noise = []
    for i in range(data.shape[0]): # data.shape[0] is batch size
        coef_x0.append(sqrt_alpha_cumprod[rand_t[i]-1]) # choose time_step randomly agument model
        coef_noise.append(sqrt_oneminus_alpha_cumprod[rand_t[i]-1])
    coef_x0 = torch.tensor(coef_x0)
    coef_noise = torch.tensor(coef_noise) 
    # why ?  
    coef_x0 = coef_x0[:, None, None, None]
    coef_noise = coef_noise[:, None, None, None]
    
    return coef_x0.to(args.device) * data.to(args.device) + coef_x0.to(args.device) * noise.to(args.device)
    
# use xt to compute xt-1, use to denoising, generate image 
def compute_prev_x(xt, t, pred_noise, args, batch_size):
    coef = 1/torch.sqrt(alpha[t-1])
    noise_coef = beta[t-1] / sqrt_oneminus_alpha_cumprod[t-1]
    if t <= 1:
        z = 0
    else:
        z = torch.randn(pred_noise.shape[0], 3, 64, 64) # z follows N(0, I) distribution
    sqrt_var = sqrt_variance[t-1]
    mean = coef * (xt[:pred_noise.shape[0]] - noise_coef * pred_noise)
    prev_x = mean.to('cpu') + sqrt_var.to('cpu') * z
    
    return prev_x
    
def train(args, model, device, train_loader, test_loader, optimizer, lr_scheduler, accelerator):
    model.train()
    noise_scheduler = DDPMScheduler(num_train_timesteps=time_step)
    for epoch in range(1, args.epochs+1):
        for batch_idx, (data, condition) in enumerate(train_loader):
            data, condition = data.to(device,dtype=torch.float32), condition.to(device)
            condition = condition.squeeze() # reduce dimension
            optimizer.zero_grad() 
            # select t, produce 1, 2, ....time_step, include end
            # but when processing, we minus 1 on t, the final range is 0, 1, ....., time_step-1 
            rand_t = torch.tensor([random.randint(1, time_step) for i in range(data.shape[0])])
            # select noise
            noise = torch.randn(data.shape[0], 3, 64, 64) # why this shape?
            xt = compute_xt(args, data, rand_t, noise)
            noisy_image = noise_scheduler.add_noise(xt.to(args.device), noise.to(args.device), 
                random.choice(rand_t)-1)
            output = model(sample=noisy_image.to(args.device), timestep=rand_t.to(args.device), 
                class_labels=condition.to(torch.float32))
            loss_function = nn.MSELoss()
            loss = loss_function(output.sample.to(args.device), noise.to(args.device))
            accelerator.backward(loss)
            lr_scheduler.step()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        ((100 * batch_idx) / len(train_loader)), loss.item()))
            sample(model, device, test_loader, args, "test_"+str(epoch))
        if args.save_model and (epoch % args.save_interval)==0:
            path = os.path.join(args.ckpt, 'train_epoch='+str(epoch)+'.pt')
            torch.save(model.state_dict(), path)
        print("==test.json==")
        # sample(model, device, test_loader, args, "test_"+str(epoch))

def save_images(images, filename, args):
    grid = torchvision.utils.make_grid(images.detach().cpu().clip(-1, 1), nrow=8)
    if not os.path.exists(os.path.join(args.save_root, args.mode)):
        os.makedirs(os.path.join(args.save_root, args.mode))
    save_image(grid, fp=os.path.join(args.save_root, args.mode, filename+'.png'), normalize=True)
    
def sample(model, device, test_loader, args, filename):
    model.eval()
    with torch.no_grad():
        # img here is a dummy value
        best_acc = 0
        best_xt = None
        best_xt_name = None
        best_acc_file = None
        best_acc_file_path = None
        for batch_idx, (img, condition) in enumerate(test_loader):
            xt = torch.randn(args.test_batch, 3, 64, 64)
            if args.mode == 'test':
                filename = 'test_' + str(batch_idx)
            condition = condition.to(device, dtype=torch.float32).squeeze()
            # reconstruct image
            for t in range(time_step, 0, -1):
                # let batch number of xt the same as condition
                output = model(sample=xt[:condition.shape[0]].to(device, dtype=torch.float32), 
                    timestep=t, class_labels=condition)
                print(f'Shpae of output : {output.shape}')
                print(f'output : {output}')
                xt = compute_prev_x(xt.to(device, dtype=torch.float32), t, output.sample.to(device), 
                    args, args.test_batch)
            # evaluate
            evaluate = evaluation_model()
            txt_path = os.path.join(args.save_root, args.mode, 'txt_file', f'{filename}.txt')
            acc = evaluate.eval(xt.to(device), condition, txt_path)
            # img_path = os.path.join(args.ckpt, f'{filename}.png')
            if acc > best_acc:
                best_xt = xt
            print(f'The result: {acc}')
            if not os.path.exists(txt_path):
                with open(txt_path, "w") as file:
                    pass
            with open(txt_path, 'a') as test_record:
                test_record.write((f'Accuracy : {acc}\n'))
                
        save_images(best_xt, filename, args)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--train_batch', type=int, default=20)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4 * 0.5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--save_root', default='./save_root')
    parser.add_argument('--dataset', default='./iclevr')
    parser.add_argument('--ckpt', default='./ckpt')
    parser.add_argument('--load_model', default='./ckpt/test_22.pt')
    args = parser.parse_args()
    device = torch.device(args.device)
    train_kwargs = {'batch_size': args.train_batch}
    test_kwargs = {'batch_size': args.test_batch}
        
    # dataloader
    train_loader = torch.utils.data.DataLoader(iclevrLoader(args=args, data_path=args.dataset, mode="train"),**train_kwargs,shuffle=True)
    test_loader = torch.utils.data.DataLoader(iclevrLoader(args=args, data_path=args.dataset, mode="test"),**test_kwargs,shuffle=True)
    test_loader_new = torch.utils.data.DataLoader(iclevrLoader(args=args, data_path=args.dataset, mode="new_test"),**test_kwargs,shuffle=False)
    # create model
    model = UNet2DModel(
        sample_size = 64,
        in_channels = 3,
        out_channels = 3,
        layers_per_block = 2,
        class_embed_type = None, 
        block_out_channels = (128, 128, 256, 256, 512, 512),
        down_block_types = (
            "DownBlock2D",  
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  
            "DownBlock2D",
        ), 
        up_block_types=(
            "UpBlock2D",  
            "AttnUpBlock2D",  
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ))
    # overwrite class_embedding because there are 24 different objects 
    model.class_embedding = nn.Linear(24, 512)
    if args.test_only:
        # load model 
        # model = UNet2DModel.from_pretrained(pretrained_model_name_or_path = 'local-unetepoch_20', 
        #     variant = 'non_ema', from_tf=True, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        # model.class_embedding = nn.Linear(24, 512)
        # state_dict = torch.load('./ckpt/test_22.pt')
        # # key[16:] is used to remove 'class_embedding.' just keep 'weight' or 'bias'
        # filter_state_dict = {k[16:]:v for k, v in state_dict.items() if k == 'class_embedding.weight' or k == 'class_embedding.bias'}
        # model.class_embedding.load_state_dict(filter_state_dict)
        
        model.load_state_dict(torch.load(args.load_model))
    else:
        
        # optimizer 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps = 0, 
            num_training_steps = len(train_loader)*500)
    
    model = model.to(args.device)
    
    # Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    if not args.test_only:
        train(args, model, device, train_loader, test_loader, optimizer, lr_scheduler, accelerator)
    else:
        sample(model, device, test_loader_new, args, None)
if __name__ == '__main__':
    main()