#!/usr/bin/python3 env
print("setup the envirement ...")

#!git clone https://github.com/MED-YAHYAOUI/style-transfer-pytorch

#!pip install -e ./style-transfer-pytorch

#!cd style-transfer-pytorch/

#import the necessary package and class

from style_transfer import StyleTransfer

from style_transfer import srgb_profile

import argparse
import atexit
from dataclasses import asdict
import io
import json
from pathlib import Path
import platform
import sys
import webbrowser

import numpy as np
from PIL import Image, ImageCms
from tifffile import TIFF, TiffWriter
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

#define the functions

def print_error(err):
    print('\033[31m{}:\033[0m {}'.format(type(err).__name__, err), file=sys.stderr)


def prof_to_prof(image, src_prof, dst_prof, **kwargs):
    src_prof = io.BytesIO(src_prof)
    dst_prof = io.BytesIO(dst_prof)
    return ImageCms.profileToProfile(image, src_prof, dst_prof, **kwargs)


def load_image(path, proof_prof=None):
    src_prof = dst_prof = srgb_profile
    try:
        image = Image.open(path)
        if 'icc_profile' in image.info:
            src_prof = image.info['icc_profile']
        else:
            image = image.convert('RGB')
        if proof_prof is None:
            if src_prof == dst_prof:
                return image.convert('RGB')
            return prof_to_prof(image, src_prof, dst_prof, outputMode='RGB')
        proof_prof = Path(proof_prof).read_bytes()
        cmyk = prof_to_prof(image, src_prof, proof_prof, outputMode='CMYK')
        return prof_to_prof(cmyk, proof_prof, dst_prof, outputMode='RGB')
    except OSError as err:
        print_error(err)
        sys.exit(1)


devices = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')]


devices

def setup_exceptions():
    try:
        from IPython.core.ultratb import FormattedTB
        sys.excepthook = FormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


def fix_start_method():
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')


setup_exceptions()
fix_start_method()

def save_pil(path, image):
    try:
        kwargs = {'icc_profile': srgb_profile}
        if path.suffix.lower() in {'.jpg', '.jpeg'}:
            kwargs['quality'] = 95
            kwargs['subsampling'] = 0
        elif path.suffix.lower() == '.webp':
            kwargs['quality'] = 95
        image.save(path, **kwargs)
    except (OSError, ValueError) as err:
        print_error(err)
        sys.exit(1)



def save_image(path, image):
    path = Path(path)
    tqdm.write(f'Writing image to {path}.')
    if isinstance(image, Image.Image):
        save_pil(path, image)
    elif isinstance(image, np.ndarray) and path.suffix.lower() in {'.tif', '.tiff'}:
        save_tiff(path, image)
    else:
        raise ValueError('Unsupported combination of image type and extension')



#predict_sign is a function that load and style a list of images and styles , 
#it takes a list of path to contents and styles and save img in return ,
def predict_sign(contents,styles):
#     iterations = input("Enter your value for iteration default is 500: ")
#     print('iterations : '+str(iterations))
#     initial_iterations = input("Enter your value for initial_iterations default is 1000: ")
#     print('initial_iterations : '+str(initial_iterations))
#     style_scale_fac = input("Enter your value for style_scale_fac default is 1. : ")
#     print('style_scale_fac : '+str(style_scale_fac))
#     step_size = input("Enter your value for step_size default is 0.02 : ")
#     print('step_size : '+str(step_size))
#     content_weight = input("Enter your value for content_weight default is 0.015: ")
#     print('content_weight : '+str(content_weight))
#     tv_weight = input("Enter your value for tv_weight default is 2. : ")
#     print('tv_weight : '+str(tv_weight))
#     end_scale = input("Enter your value for end_scale default is 512 : ")
#     print('end_scale : '+str(end_scale))
    g=0

    
    for content,style in zip(contents,styles):
        g=g+1
        print("loading img from paths...")
        content = load_image(content)
        style = load_image(style)
        
        print("loading img done.")
        
        
        if devices[0].type == 'cuda':
            
            for i, device in enumerate(devices):
                props = torch.cuda.get_device_properties(device)
                print(f'GPU {i} type: {props.name} (compute {props.major}.{props.minor})')
                print(f'GPU {i} RAM:', round(props.total_memory / 1024 / 1024), 'MB')
        for device in devices:
                  torch.tensor(0).to(device)
                  torch.manual_seed(0)
        print("...... styling signature  "+str(g))
        st = StyleTransfer(devices=devices, pooling='max')
            
        try:
            st.stylize(content_image=content,style_images=[style]
#                        ,iterations=int(iterations),
#                        initial_iterations=int(initial_iterations),
#                        style_scale_fac=float(style_scale_fac),
#                        step_size=float(step_size),
#                        content_weight=float(content_weight),
#                        tv_weight=float(tv_weight),
#                        end_scale=int(end_scale)
                      )
        except KeyboardInterrupt:
            pass
        image_type = 'pil'
        output_image = st.get_image(image_type)
        if output_image is not None:
            save_image(str(g)+"_output.png", output_image)


    
#predict_sign_multistyle is a procedure that take one img and multi style(u can modify it to one style by cmnting the ling 194 and uncoment 195)
# u can modify the parametre of styling at stylize function
# it takes a loaded img and content by the method load img and save generated img in return
# here is and exemple of how u can use it
#content_img = load_image(contents_list, None)
#style_imgs = [load_image(img, None) for img in styles_list]
#styles_list & contents_list lists that have the path to imgs


def predict_sign_multistyle(content_img,style_imgs):
    from datetime import datetime
    
    
    if devices[0].type == 'cuda':
            
        for i, device in enumerate(devices):
                props = torch.cuda.get_device_properties(device)
                print(f'GPU {i} type: {props.name} (compute {props.major}.{props.minor})')
                print(f'GPU {i} RAM:', round(props.total_memory / 1024 / 1024), 'MB')
        for device in devices:
                  torch.tensor(0).to(device)
                  torch.manual_seed(0)
        print("...... styling signature  ")

    now = datetime.now()
    
    
    st = StyleTransfer(devices=devices, pooling='max')
    try:
        st.stylize(content_image=content_img,style_images=style_imgs
        #st.stylize(content_image=content_img,style_images=[style_imgs] #for one style
                     ,iterations=int(300),
                     initial_iterations=int(500),
                     #style_scale_fac=float(2.03),
#                     step_size=float(step_size),
                     #content_weight=float(0.015),
#                     tv_weight=float(tv_weight),
                     end_scale=int(256)
                  )
    except KeyboardInterrupt:
        pass
    image_type = 'pil'
    output_image = st.get_image(image_type)
    if output_image is not None:
        save_image("_test_output.png", output_image)
    now2 = datetime.now()
    now3=now2-now
    # current_time = now3.strftime("%H:%M:%S")
    print("execution Time =", now3)
    

