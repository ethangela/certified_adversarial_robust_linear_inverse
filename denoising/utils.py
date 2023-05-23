
import torch
import numpy as np
import math

# input type:tensor ; output type:tensor


def add_noise(input_img, noise_sigma, device, debug=False):
    noise_sigma = noise_sigma / 255
    noise = noise_sigma * torch.randn_like(input_img).to(device)
    noise_img = torch.clamp(input_img + noise, 0.0, 1.0)
    if debug:
        debug_noise = torch.clamp(input_img - noise, 0.0, 1.0)
        return noise_img, debug_noise
    else:
        return noise_img


def PSNR(opt1, opt2, tgt, color=False):
    dis = np.mean((opt1 / 255. - opt2 / 255.) ** 2)
    mse = np.mean((opt1 / 255. - tgt / 255.) ** 2)
    PIXEL_MAX = 1
    if mse < 1.0e-10:
        psnr = 100
    else:
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return dis, psnr



if __name__ == '__main__':
    a = {0:1.4, 1:1.5, 2:0.6, 3: 0.1}
    b = dict(sorted(a.items(), key=lambda item: item[1]))
    print(b)
    print(list(b.keys())[0])
















