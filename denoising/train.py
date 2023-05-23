import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import DPDNN
from Dataset import Train_Data
# from config import opt
from torch.utils.data import DataLoader
from visdom import Visdom
from PIL import Image
from torchvision import transforms as T
from utils import PSNR, add_noise
import os
from argparse import ArgumentParser
from torch import autograd


def train(opt):

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda', torch.cuda.current_device()) #cuda:0/1

    train_data = Train_Data(opt.data_root)
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    net = DPDNN()
    criterion = nn.MSELoss()
    net = net.to(device)
    net = nn.DataParallel(net)
    criterion = criterion.to(device)

    # initialize weights by Xavizer
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    # # Save the original model
    # torch.save(net.state_dict(), opt.save_model_path)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    num_batch = 0
    num_show = 0

    # visdom
    # vis = Visdom()

    print('start training ...')

    for epoch in range(opt.max_epoch):
        for idx, label in enumerate(train_loader):
            # data = data.to(device) #noisy data
            label = label.to(device) #ground truth
            
            #smt-grad
            if opt.smt: 
                path = f'smtgrad_stp{opt.stp}_std{opt.std}_smp{opt.smp}'
                times = opt.stp
                in_times = opt.smp
                for j in range(1,times+1):
                    optimizer.zero_grad()
                    for _ in range(in_times):
                        Soutput = net(add_noise(add_noise(label, (opt.std/times)*j, device), opt.noise_level, device))
                        smt_loss = criterion(Soutput, label) / (times * in_times)
                        smt_loss.backward(retain_graph=True)
                    optimizer.step()
                print(f'smtgrad epoch {epoch+1} batch {idx+1} done.')

            #adv
            elif opt.atk:
                adv_data = label.clone().detach().to(device).requires_grad_()
                adv_opt = torch.optim.SGD([adv_data], lr=opt.alp)
                for _ in range(opt.itr):
                    #smt-adv
                    if opt.smp:
                        path = f'smtadv_itr{opt.itr}_alp{opt.alp}_eps{opt.eps}_smp{opt.smp}_std{opt.std}'
                        for _ in range(opt.smp):
                            # if j==0:
                            #     Aoutput = torch.clamp(net(add_noise(add_noise(adv_data, opt.std, device), opt.noise_level, device)), min=0, max=1) / opt.smp
                            # else:
                            #     Aoutput += torch.clamp(net(add_noise(add_noise(adv_data, opt.std, device), opt.noise_level, device)), min=0, max=1) / opt.smp
                            Aoutput = torch.clamp(net(add_noise(add_noise(adv_data, opt.std, device), opt.noise_level, device)), min=0, max=1)
                            adv_loss = -1. * criterion(Aoutput, label) / opt.smp #adv_loss = criterion(Aoutput, label) / opt.smp
                            adv_opt.zero_grad()
                            adv_loss.backward(retain_graph=True)
                            adv_opt.step()
                            del Aoutput
                            del adv_loss
                    else:
                        path = f'adv_itr{opt.itr}_alp{opt.alp}_eps{opt.eps}'
                        Aoutput = torch.clamp(net(add_noise(adv_data, opt.noise_level, device)), min=0, max=1) 
                        adv_loss = -1. * criterion(Aoutput, label) #adv_loss = criterion(Aoutput, label)
                        adv_opt.zero_grad()
                        adv_loss.backward(retain_graph=True)
                        adv_opt.step() #adv_data.data = adv_data.data + opt.alp * adv_data.grad
                        del Aoutput
                        del adv_loss
                    with torch.no_grad():
                        diff_ori = adv_data - label
                        norm = torch.norm(diff_ori) #l2 norm
                        div = norm/opt.eps if norm>opt.eps else 1.
                        adv_data.data = diff_ori/div + label
                ptb_data = adv_data.clone().detach().to(device).requires_grad_(False)
                del adv_data
                optimizer.zero_grad()
                output = net(add_noise(ptb_data, opt.noise_level, device)) 
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                print(f'adv/smtadv epoch {epoch+1} batch {idx+1} done.')

            #jcb
            elif opt.jcb:
                def random_u(b,c): 
                    v = torch.normal(0, 1, size=(b,c)) 
                    vnorm = torch.norm(v, 2, 1, True) 
                    nomrlised_v = torch.div(v, vnorm) #(w,h)
                    return nomrlised_v
                path = 'jcb'
                noisy_input = add_noise(label, opt.noise_level, device).requires_grad_()
                output = net(noisy_input) #[32, 1, 128, 128]
                O_loss = criterion(output, label)
                
                z = random_u(opt.batch_size, 128*128).view(opt.batch_size, 1, 128, 128).to(device) #[32, 1, 128, 128]
                autograd.backward(output, z, retain_graph=True)
                J = noisy_input.grad #[32, 1, 128, 128]
                J_norm = 128*128 * torch.mean(torch.norm(J, dim=(-2,-1)))**2 / (1*opt.batch_size)
                beta_J = torch.pow( 10, torch.floor(torch.log(O_loss/J_norm)) ) / opt.gma
                J_loss = beta_J * J_norm
                
                loss = O_loss + J_loss
                noisy_input.requires_grad_(False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f'jcb epoch {epoch+1} batch {idx+1} done.')

                
            #ord
            else:
                path = 'ord'
                output = net(add_noise(label, opt.noise_level, device))
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            #log
            if (idx+1) % 20 == 0:
                print(f'epoch {epoch+1} batch {idx+1} done.')
        
        #save
        if (epoch+1) % 20 == 0:
            torch.save(net.state_dict(), f'{opt.save_model_dir}/{path}_sigma{opt.noise_level}_data{opt.num_data}_epo{epoch+1}.pth') 

        #learning rate decay
        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay #TODO

    print('Finished Training')


# This function is for checking the training effect, not the test code
def test(epoch, i):
    net1 = DPDNN()
    net1 = net1.to(device)
    net1 = nn.DataParallel(net1)
    net1.load_state_dict(torch.load(opt.load_model_path))

    noise = Image.open('./data/test_data/sigma%d/test.png'%opt.noise_level)
    label = Image.open('./data/test_data/sigma%d/testgt.png'%opt.noise_level)

    img_H = noise.size[0]
    img_W = noise.size[1]

    transform = T.ToTensor()
    transform1 = T.ToPILImage()
    noise = transform(noise)
    noise = noise.resize_(1, 1, img_H, img_W)
    noise = noise.to(device)
    label = np.array(label).astype(np.float32)

    output = net1(noise)     # dim=4
    output = torch.clamp(output, min=0.0, max=1.0)
    output = torch.tensor(output)
    output = output.resize(img_H, img_W).cpu()
    output_img = transform1(output)

    # every 500 batches save test output
    if i%500 == 0:
        save_index = str(int(epoch*(opt.num_data/opt.batch_size/500) + (i+1)/500))
        output_img.save('./data/test_data/sigma%d/test_output/'%opt.noise_level+save_index+'.png')

    output = np.array(output_img)
    mse, psnr = PSNR(output, label)
    return mse, psnr


if __name__ == '__main__':
    
    parser = ArgumentParser(description='dpdnn denoiser training')

    parser.add_argument('--data_root', type=str, default='./data/DIV2K_gray', help='training data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--max_epoch', type=int, default=40, help='max_epoch') #Apr18
    parser.add_argument('--save_model_dir', type=str, default='./checkpoints', help='load_model_path')
    parser.add_argument('--num_data', type=int, default=3200, help='num of croped patches') #need align with config.py
    parser.add_argument('--noise_level', type=int, default=15, help='noise_level') #need align with config.py

    parser.add_argument('--atk', type=int, default=0, help='attack indicator')
    parser.add_argument('--alp', type=float, default=0., help='attack step size')
    parser.add_argument('--eps', type=float, default=0., help='attack norm')
    parser.add_argument('--itr', type=int, default=0, help='attack iterations')

    parser.add_argument('--jcb', type=int, default=0, help='jacobian indicator')
    # parser.add_argument('--spc', type=int, default=0, help='spectral norm iterations')
    parser.add_argument('--gma', type=int, default=20, help='spectral norm iterations')

    parser.add_argument('--smt', type=int, default=0, help='smoothing indicator')
    parser.add_argument('--smp', type=int, default=0, help='smoothing samples')
    parser.add_argument('--std', type=float, default=0., help='smoothing std')
    parser.add_argument('--stp', type=int, default=0, help='smoothing training stpes')

    parser.add_argument('--gpu', type=str, default='0', help='gpu device')

    opt = parser.parse_args()

    train(opt)

    # a = torch.ones((2,1,3,3))
    # b = torch.norm(a, dim=(-2,-1))
    # c = torch.mean(b)
    # print(b)
    # print(c)






