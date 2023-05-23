
import torch
from PIL import Image 
import numpy as np
from torchvision import transforms as T
from utils import add_noise
from utils import PSNR
import math
from model import DPDNN
import torch.nn as nn
# from config import opt
import os
import scipy.stats as stats
import math
import pickle as pkl 
import pandas as pd
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt


#args
parser = ArgumentParser(description='dpdnn denoiser')

parser.add_argument('--load_model_path', type=str, default='./org_checkpoints/DPDNN_denoise_sigma15.pth', help='load_model_path')
parser.add_argument('--noi', type=int, default=15, help='noise_level')

parser.add_argument('--atk', type=int, default=1, help='attack indicator')
parser.add_argument('--alp', type=float, default=40000., help='attack step size')
parser.add_argument('--eps', type=float, default=3., help='attack norm')
parser.add_argument('--itr', type=int, default=6, help='attack iterations')

parser.add_argument('--smt', type=int, default=1, help='smoothing indicator')
parser.add_argument('--smp', type=int, default=100, help='smoothing samples')
parser.add_argument('--std', type=float, default=3., help='smoothing std')

parser.add_argument('--vis', type=int, default=0, help='visualize indicator')
parser.add_argument('--pkl', type=str, default='test_ord', help='pickle file')
parser.add_argument('--gpu', type=str, default='0', help='gpu device')

parser.add_argument('--sgm', type=int, default=None, help='smoothing sigma used for training')

opt = parser.parse_args()


#helper
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda', torch.cuda.current_device()) #cuda:0/1
transform1 = T.ToTensor()
transform2 = T.ToPILImage()
noise_level = opt.noi
test_alpha = opt.alp
test_attack_iter = opt.itr
test_epsilon = opt.eps
test_smooth = opt.smt
test_std = opt.std
test_sample = opt.smp
pickle_file_path = opt.pkl + '.pkl'
visual = opt.vis
load_model_path = opt.load_model_path 

def estimate_ql_qu(eps, sample_count, sigma, conf_thres=.99):
    theo_perc_l = stats.norm.cdf(-eps/sigma)
    theo_perc_u = stats.norm.cdf(eps/sigma)

    q_u_u = sample_count -1
    q_u_l = math.ceil(theo_perc_u*sample_count)
    q_l_u = math.floor(theo_perc_l*sample_count)
    q_l_l = 0
    
    q_u_final = q_u_u
    count = q_u_l
    for q_u in range(q_u_l, q_u_u):
        conf = stats.binom.cdf(q_u-1, sample_count, theo_perc_u)
        count += 1
        if conf > conf_thres:
            q_u_final = q_u
            break

    if count-1 == q_u_u:
        raise Exception("Null")
    else:
        return q_u_final 

def img_crop(image, crop_size):
    img_H = image.size[1]
    img_W = image.size[0]

    if img_H <= img_W:
        left = (img_W - img_H)/2
        top = 0
        right = (img_W + img_H)/2
        bottom = img_H
    elif img_H > img_W:
        left = 0
        top = (img_H - img_W)/2
        right = img_W
        bottom = (img_H + img_W)/2
    crop_box = (left, top, right, bottom)
    image = image.crop(crop_box)
    image = image.resize((crop_size,crop_size), Image.ANTIALIAS)
    return image

#model
model_type = load_model_path.split('/')[-1].split('_')[0] #e.g., "./checkpoints/ord_sigma15_data3200_epo100.pth"
if opt.sgm:
    model_type += f'_{opt.sgm}'
net = DPDNN()
net = nn.DataParallel(net)
net.load_state_dict(torch.load(load_model_path))
net = net.to(device)

#for log
num_data = 68 
num_restart = 1
cropsize = 256 
PSNR_All = np.zeros([4, num_data*num_restart], dtype=np.float32)
DIS_All = np.zeros([4, num_data*num_restart], dtype=np.float32)
ql = -1
qu = -1
dis_debug_mean, psnr_debug_mean = [], []
ddis_debug_mean1, dpsnr_debug_mean1 = [], []
ddis_debug_mean2, dpsnr_debug_mean2 = [], []
dis_debug, psnr_debug = [], []
ddis_debug, dpsnr_debug = [], []
psnr_atk = []

for img_idx in range(num_data): 
                
    #data
    img_idx_ = '00' + str(img_idx+1) if img_idx < 9 else '0' + str(img_idx+1)
    img = Image.open(f'./BSD68/test{img_idx_}.png')
    img = img_crop(img, cropsize)
    if visual:
        model_dir = f'./BSD68_output/{img_idx+1}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        img.save(model_dir + f'/original_{img_idx+1}.png')
    label = np.array(img).astype(np.float32)  
    img_H = img.size[0]
    img_W = img.size[1]
    img = transform1(img).resize_(1, 1, img_H, img_W).to(device)

    #attack
    adv_img = img.clone().detach().type(torch.FloatTensor).to(device).requires_grad_()
    optimizer = torch.optim.SGD([adv_img], lr=test_alpha)
    for i in range(test_attack_iter):
        x_output = torch.clamp(net(add_noise(adv_img, noise_level, device)), min=0, max=1) #Ax + b
        loss = -1 * torch.mean(torch.pow(x_output - img, 2))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        with torch.no_grad():
            diff_ori = adv_img - img
            norm = torch.norm(diff_ori) #l2 norm
            div = norm/test_epsilon if norm>test_epsilon else 1.
            adv_img.data = diff_ori/div + img
            # print(f'PGD info: diff_norm:{norm}. epsilon:{test_epsilon}. norm/eps:(>1? 1=no touch bound) {div}')
    ptb_img = adv_img.clone().detach().type(torch.FloatTensor).to(device).requires_grad_(False) #x+\xi
    if visual:
        transform2(ptb_img.clone().detach().type(torch.FloatTensor).requires_grad_(False).resize_(img_H, img_W)).save('%s/noisy_nom%.1f.png'%(model_dir, test_epsilon))
    ptb_img = add_noise(ptb_img, noise_level, device) #A(x+\xi)+b
    if visual:
        transform2(ptb_img.clone().detach().type(torch.FloatTensor).resize_(img_H, img_W)).save('%s/noisy_nom%.1f_noi%.1f.png'%(model_dir, test_epsilon, noise_level))

    #denoising
    with torch.no_grad():
        Ooutput = torch.clamp(net(img).cpu().resize_(img_H, img_W), min=0, max=1)
        Noutput = torch.clamp(net(ptb_img).cpu().resize_(img_H, img_W), min=0, max=1)
        Ndis, Npsnr = PSNR(np.array(Noutput)*255., np.array(Noutput)*255., label) 
        if visual:
            transform2(Noutput).save('%s/noisy_nom%.1f_noi%.1f_%s_Orec%.2f.png'%(model_dir, test_epsilon, noise_level, model_type, Npsnr))

        if test_smooth:
            ql, qu = estimate_ql_qu(test_epsilon, test_sample, test_std, conf_thres=.99)
            samples_dis = []
            samples_PSNR = []

            for j in range(test_sample):
                ptb_img_noise = add_noise(ptb_img, test_std, device) #A(x+\xi)+b+\delta 
                Soutput = torch.clamp(net(ptb_img_noise).cpu().resize_(img_H, img_W), min=0, max=1)
                Sdis, Spsnr = PSNR(np.array(Soutput)*255., np.array(Noutput)*255., label)          
                samples_dis.append(Sdis)
                samples_PSNR.append(Spsnr)
                if visual:
                    transform2(Soutput).save('%s/noisy_nom%.1f_noi%.1f_%s_Srec_%d_%.2f.png'%(model_dir, test_epsilon, noise_level, model_type, j, Spsnr))
            
    psnr_atk.append(Npsnr)
    dis_debug.append(samples_dis) #(img,test_sample)
    psnr_debug.append(samples_PSNR) #(img,test_sample)

    print(f'image {img_idx+1}/{num_data} done.')

#log
dis = np.mean(np.array(dis_debug), axis=0).squeeze() #(100,)
psnr = np.mean(np.array(psnr_debug), axis=0).squeeze() #(100,)
atk_psnr = np.mean(np.array(psnr_atk)).squeeze() #(68,) to (1,)

dis_std = np.std(np.array(dis_debug), axis=0).squeeze() #(100,)
psnr_std = np.std(np.array(psnr_debug), axis=0).squeeze() #(100,)
atk_psnr_std = np.std(np.array(psnr_atk)).squeeze() #(68,) to (1,)

dis_idx_list = sorted(range(len(dis)), key=lambda k: dis[k])
lb_idx = dis_idx_list[ql]
me_idx = dis_idx_list[int(test_sample/2)]
ub_idx = dis_idx_list[qu]

if not os.path.exists(pickle_file_path):
    d = {'model_type':[model_type], 'atk_eps':[test_epsilon], 'add_noise':[noise_level],
        'smooth':[test_smooth], 'smooth_std':[test_std], 'smooth_sample':[test_sample], 'ql':[ql], 'qu':[qu], 
        'atk_psnr':[atk_psnr], 'lb_psnr':[psnr[lb_idx]], 'sth_psnr':[psnr[me_idx]], 'ub_psnr':[psnr[ub_idx]],
        'atk_dis':[0.], 'lb_dis':[dis[lb_idx]], 'sth_dis':[dis[me_idx]], 'ub_dis':[dis[ub_idx]],
        'atk_psnr_std':[atk_psnr_std], 'lb_psnr_std':[psnr_std[lb_idx]], 'sth_psnr_std':[psnr_std[me_idx]], 'ub_psnr_std':[psnr_std[ub_idx]],
        'atk_dis_std':[0.], 'lb_dis_std':[dis_std[lb_idx]], 'sth_dis_std':[dis_std[me_idx]], 'ub_dis_std':[dis_std[ub_idx]]
        }
    df = pd.DataFrame(data=d)
    df.to_pickle(pickle_file_path)
else:
    d = {'model_type':model_type, 'atk_eps':test_epsilon, 'add_noise':noise_level,
        'smooth':test_smooth, 'smooth_std':test_std, 'smooth_sample':test_sample, 'ql':ql, 'qu':qu, 
        'atk_psnr':atk_psnr, 'lb_psnr':psnr[lb_idx], 'sth_psnr':psnr[me_idx], 'ub_psnr':psnr[ub_idx],
        'atk_dis':0., 'lb_dis':dis[lb_idx], 'sth_dis':dis[me_idx], 'ub_dis':dis[ub_idx],
        'atk_psnr_std':atk_psnr_std, 'lb_psnr_std':psnr_std[lb_idx], 'sth_psnr_std':psnr_std[me_idx], 'ub_psnr_std':psnr_std[ub_idx],
        'atk_dis_std':0., 'lb_dis_std':dis_std[lb_idx], 'sth_dis_std':dis_std[me_idx], 'ub_dis_std':dis_std[ub_idx]
        }
    df = pd.read_pickle(pickle_file_path)
    df = df.append(d, ignore_index=True)
    df.to_pickle(pickle_file_path)

with open(pickle_file_path, "rb") as f:
    object = pkl.load(f)
df = pd.DataFrame(object)
df.to_csv(f'{opt.pkl}.csv', index=False)













