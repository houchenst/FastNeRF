import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import json
import random
import time
import pprint

import matplotlib.pyplot as plt

import run_nerf_fast

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

# profiling
import cProfile
import pstats
from pstats import SortKey

# CELL 1

basedir = './logs'
expname = 'fern_example'
# expname = 'fern_test'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf_fast.config_parser()

weights_name = 'model_200000.npy'
# weights_name = 'model_000700.npy'
args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, weights_name)))
print('loaded args')

images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, 
                                                          recenter=True, bd_factor=.75, 
                                                          spherify=args.spherify)
H, W, focal = poses[0,:3,-1].astype(np.float32)

H = int(H)
W = int(W)
hwf = [H, W, focal]

images = images.astype(np.float32)
poses = poses.astype(np.float32)

if args.no_ndc:
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
else:
    near = 0.
    far = 1.


# CELL 2

# Create nerf model

def new_render():
    _, render_kwargs_test, start, grad_vars, models = run_nerf_fast.create_nerf(args)

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs_test.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs_test)


    down = 4
    render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
    render_kwargs_fast['N_importance'] = 128

    c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
    test = run_nerf_fast.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)
    img = np.clip(test[0],0,1)
    disp = test[1]
    disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
    acc_alpha = test[2]
    return img,disp,acc_alpha


# profile rendering
# cProfile.run('img,disp, acc_alpha = new_render()', 'render_stats')
img, disp,acc_alpha = new_render()

# show results
plt.imshow(img)
plt.show()

# profiling results
p = pstats.Stats('render_stats')
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

# CELL 3

# down = 8 # trade off resolution+aliasing for render speed to make this video faster
# frames = []
# for i, c2w in enumerate(render_poses):
#     if i%8==0: print(i)
#     test = run_nerf_fast.render(H//down, W//down, focal/down, c2w=c2w[:3,:4], **render_kwargs_fast)
#     frames.append((255*np.clip(test[0],0,1)).astype(np.uint8))
    
# print('done, saving')
# f = 'logs/fern_example/video.mp4'
# imageio.mimwrite(f, frames, fps=30, quality=8)

# from IPython.display import Video
# Video(f, height=320)

# Cell 4

# from ipywidgets import interactive, widgets
# import matplotlib.pyplot as plt
# import numpy as np


# def f(x, y, z):
    
#     c2w = tf.convert_to_tensor([
#         [1,0,0,x],
#         [0,1,0,y],
#         [0,0,1,z],
#         [0,0,0,1],
#     ], dtype=tf.float32)
    
#     test = run_nerf_fast.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_fast)
#     img = np.clip(test[0],0,1)
    
#     plt.figure(2, figsize=(20,6))
#     plt.imshow(img)
#     plt.show()
    

# sldr = lambda : widgets.FloatSlider(
#     value=0.,
#     min=-1.,
#     max=1.,
#     step=.01,
# )

# names = ['x', 'y', 'z']
    
# interactive_plot = interactive(f, **{n : sldr() for n in names})
# interactive_plot