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
import argparse

import matplotlib.pyplot as plt

import run_nerf
import run_nerf_fast
import point_cloud

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
parser = run_nerf.config_parser()

weights_name = 'model_200000.npy'
# weights_name = 'model_000700.npy'
args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, weights_name)))
print('loaded args')

images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, 
                                                          recenter=True, bd_factor=.75, 
                                                          spherify=args.spherify)
H, W, focal = poses[0,:3,-1].astype(np.float32)
poses = poses[:, :3, :4]
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

def new_render(pc=False,sparse=False, down=16):
    _, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs_test.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs_test)


    render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
    render_kwargs_fast['N_importance'] = 128

    # c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
    if not sparse:
        test = run_nerf.render(H//down, W//down, focal/down, c2w=poses[0], pc=pc, cloudsize=16, **render_kwargs_fast)
    else:
        test = run_nerf_fast.render(H//down, W//down, focal/down, c2w=poses[0], **render_kwargs_fast)
    img = np.clip(test[0],0,1)
    disp = test[1]
    disp = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
    return img,disp


# # profile rendering
# cProfile.run('img,disp = new_render()', 'render_stats')

# # show results
# plt.imshow(img)
# plt.show()

# # profiling results
# p = pstats.Stats('render_stats')
# p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

if __name__ == "__main__":
    parser2 = argparse.ArgumentParser(description='Show FastNeRF rendering in action')
    parser2.add_argument('--vanilla',  action='store_true', help='renders using vanilla nerf')
    parser2.add_argument('--cloud', action='store_true', help='renders by limiting the depth range using a point cloud')
    parser2.add_argument('--interpolate', action='store_true', help='renders by sparsely (1/9) sampling the first pass and interpolating pdfs')
    parser2.add_argument('--cloud_viz', action='store_true', help='Visualizes the point cloud and shows depth lower and upper bounds')
    parser2.add_argument('--down', type=int, default=12, help='Downsampling factor for the novel view resolution. 16 recommended unless on GPU')
    args2 = parser2.parse_args()

    down = 16
    if args2.down is not None:
        down = args2.down

    if args2.vanilla:
        img, disp = new_render(down=down)
        plt.imshow(img)
        plt.show()
    if args2.cloud:
        img, disp = new_render(pc=True, down=down)
        plt.imshow(img)
        plt.show()
    if args2.interpolate:
        img, disp = new_render(sparse=True, down=down)
        plt.imshow(img)
        plt.show()

    if args2.cloud_viz:
        point_cloud.depth_bounds([H//down,W//down,focal/down], poses[0], cloudsize=16, bounds=True, show_view=True)
        