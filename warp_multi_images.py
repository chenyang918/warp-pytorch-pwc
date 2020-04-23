import os
import cv2
import torch
import numpy as np

import pwcnet

def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # Read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

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

def warp(imgpath1, imgpath2):
    # Read images
    img1 = read_img(imgpath1).cuda()
    img2 = read_img(imgpath2).cuda()
    # For loop training
    opt_1_to_2 = pwcnet.PWCEstimate(flownet, img2, img1, drange = True, reshape = True)
    warped_img = pwcnet.PWCNetBackward(img1, opt_1_to_2)            # range: [0, 1]
    # Visualize images
    warped_img = visualize_img(warped_img)
    return warped_img

if __name__ == '__main__':

    # Define PWC-Net
    modelpath = './pwcNet-default.pytorch'
    flownet = create_pwcnet(modelpath)
    flownet = flownet.cuda()

    # Warp
    single_folder = True
    if single_folder:
        # Define imglist
        folderpath = ''
        imglist = get_files(folderpath)
        # Warp images
        for i in range(len(imglist) - 1):
            imgpath1 = imglist[i]
            imgpath2 = imglist[i + 1]
            warped_img = warp(imgpath1, imgpath2)
    else:
        # Define imglist
        folderpath1 = ''
        folderpath2 = ''
        imglist1 = get_files(folderpath1)
        imglist2 = get_files(folderpath2)
        # Warp images
        for i in range(len(imglist)):
            imgpath1 = imglist1[i]
            imgpath2 = imglist2[i]
            warped_img = warp(imgpath1, imgpath2)
