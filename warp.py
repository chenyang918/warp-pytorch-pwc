import os
import cv2
import torch
import numpy as np

import pwcnet

def create_pwcnet(modelpath):
    # Initialize the network
    flownet = pwcnet.PWCNet().eval()
    # Load a pre-trained network
    data = torch.load(modelpath)
    if 'state_dict' in data.keys():
        flownet.load_state_dict(data['state_dict'])
    else:
        flownet.load_state_dict(data)
    print('PWCNet is loaded!')
    # It does not gradient
    for param in flownet.parameters():
        param.requires_grad = False
    return flownet

def read_img(imgpath):
    img = cv2.imread(imgpath)                                       # H, W, C
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      # H, W, C
    img = (img / 255.0).astype(np.float32)                          # H, W, C; PWC-Net needs range of [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)       # 1, C, H, W 
    return img
    
def visualize_img(img):
    img = img.data.squeeze(0).permute(1, 2, 0).cpu().numpy()        # H, W, C
    img = (img * 255.0).astype(np.uint8)                            # H, W, C
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      # H, W, C
    return img

if __name__ == '__main__':
    # Define PWC-Net
    modelpath = './pwcNet-default.pytorch'
    flownet = create_pwcnet(modelpath)
    flownet = flownet.cuda()

    # Read images
    img1 = read_img('./00015.jpg').cuda()
    img2 = read_img('./00020.jpg').cuda()
    print(img1.shape)
    print(img2.shape)
    
    # For loop training
    opt_1_to_2 = pwcnet.PWCEstimate(flownet, img2, img1, drange = True, reshape = True)
    print(opt_1_to_2.shape)
    warped_img = pwcnet.PWCNetBackward(img1, opt_1_to_2)            # range: [0, 1]
    print(warped_img.shape)
    
    # Visualize images
    warped_img = visualize_img(warped_img)
    cv2.imwrite('warped_img.png', warped_img)
