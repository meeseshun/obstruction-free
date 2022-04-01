import completionnet
import argparse
from os import path
import torch
from torch import nn
from torch.nn import Sequential
import cv2
import numpy as np
import torchvision.utils as vutils
from utils import *


def inpaint(input_image_path, mask_image_path, model_path='completionnet_places2_freeform.pth', gpu=False, postproc=False):
    # load Completion Network
    model = completionnet.completionnet
    model.load_state_dict(torch.load(model_path))
    model.eval()
    datamean = torch.FloatTensor([0.4560, 0.4472, 0.4155])

    # load data
    input_img = cv2.imread(input_image_path)
    I = torch.from_numpy(cvimg2tensor(input_img)).float()

    input_mask = cv2.imread(mask_image_path)
    M = torch.from_numpy(
        cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY) / 255).float()
    M[M <= 0.2] = 0.0
    M[M > 0.2] = 1.0
    M = M.view(1, M.size(0), M.size(1))
    assert I.size(1) == M.size(1) and I.size(2) == M.size(2)

    for i in range(3):
        I[i, :, :] = I[i, :, :] - datamean[i]

    # make mask_3ch
    M_3ch = torch.cat((M, M, M), 0)

    Im = I * (M_3ch*(-1)+1)

    # set up input
    input = torch.cat((Im, M), 0)
    input = input.view(1, input.size(0), input.size(1), input.size(2)).float()

    if gpu:
        print('using GPU...')
        model.cuda()
        input = input.cuda()

    # evaluate
    res = model.forward(input)[0].cpu()

    # make out
    for i in range(3):
        I[i, :, :] = I[i, :, :] + datamean[i]

    out = res.float()*M_3ch.float() + I.float()*(M_3ch*(-1)+1).float()

    # post-processing
    # if postproc:
    #     from poissonblending import blend
    #     print('post-processing...')
    #     target = input_img    # background
    #     source = tensor2cvimg(out.detach().numpy())    # foreground
    #     mask = input_mask
    #     out = blend(target, source, mask, offset=(0, 0))

    #     out = torch.from_numpy(cvimg2tensor(out))

    # save images
    print('saving images...')
    vutils.save_image(Im, 'masked_input.png', normalize=True)
    vutils.save_image(out, 'out.png', normalize=True)
    # vutils.save_image(M_3ch, 'mask.png', normalize=True)
    # vutils.save_image(res, 'res.png', normalize=True)
    print('Done')
