from tryon_utils import image_preparation
from numpy.lib.type_check import imag
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import cv2
import base64

def tryon_process(cloth_image_str, person_image_str):
    images_dict = image_preparation(cloth_image_str, person_image_str)
    torch_images = torch.utils.data.DataLoader(
            images_dict,
            batch_size=1,
            shuffle = False,
            num_workers=int(1))
    gen_model, warp_model = init_model() 
    model_process(gen_model, warp_model, torch_images.dataset)


def init_model():
    warp_model = AFWM(input_nc=3)
    warp_model.eval()
    warp_model.cuda()
    load_checkpoint(warp_model, 'checkpoints/PFAFN/warp_model_final.pth')

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    gen_model.eval()
    gen_model.cuda()
    load_checkpoint(gen_model, 'checkpoints/PFAFN/gen_model_final.pth')

    return gen_model, warp_model

def model_process(gen_model, warp_model, data):
        real_image = data['image']
        clothes = data['clothes']

        real_image = real_image.unsqueeze(0)
        clothes = clothes.unsqueeze(0)

        edge = data['edge']
        edge = edge.unsqueeze(0)

        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge        

        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                        mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        c = p_tryon
        
        combine = torch.cat([c[0]], 2).squeeze()
        cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb=(cv_img*255).astype(np.uint8)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        revtal, buffer = cv2.imencode('.jpg', bgr)
        text = base64.b64encode(buffer)
        print(text)
