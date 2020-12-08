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
import run_nerf_helpers_fast
import run_nerf

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
poses = poses[:, :3, :4]
print(f"NUM IMAGES: {poses.shape[0]}")

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

# get the test and train sets
if not isinstance(i_test, list):
    i_test = [i_test]

if args.llffhold > 0:
    print('Auto LLFF holdout,', args.llffhold)
    i_test = np.arange(images.shape[0])[::args.llffhold]

i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                (i not in i_test)])

down = 2

gt = images[i_test[0]]
gt_down = np.zeros((H//down, W//down, 3))
for i in range(gt_down.shape[0]):
    for j in range(gt_down.shape[1]):
        patch = gt[down*i:down*(i+1), (down*j):down*(j+1)]
        patch = np.reshape(patch, (-1,3))
        gt_down[i,j] = np.mean(patch, axis=0)

# Create nerf model

def new_render(img_dir, fast=False,r2=128,d=3):
    _, render_kwargs_test, start, grad_vars, models = run_nerf_fast.create_nerf(args)

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs_test.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs_test)


    
    render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
    render_kwargs_fast['N_importance'] = r2

    # c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
    c2w = poses[0]
    start_time = time.time()
    if fast:
        test = run_nerf_fast.render(H//down, W//down, focal/down, c2w=c2w, d_factor=d, **render_kwargs_fast)
    else:
        test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, pc=False, **render_kwargs_fast)
    end_time = time.time()
    net = end_time - start_time

    img = np.clip(test[0],0,1)
    if not fast:
        plt.imsave(os.path.join(img_dir, f"NeRF_render.png"), images[i_test[1]])
    else:
        plt.imsave(os.path.join(img_dir, f"FastNeRF_sparse_{d}x.png"), images[i_test[1]])
    
    mse = run_nerf_helpers_fast.img2mse(tf.cast(gt_down, tf.float32), img)
    psnr = run_nerf_helpers_fast.mse2psnr(mse)
    mse = float(mse)
    psnr = float(psnr)

    return net, mse, psnr


res_dir = "./fast_results"
img_dir = os.path.join(res_dir, "imgs")

plt.imsave(os.path.join(img_dir, f"GT0.png"), gt_down)

down_x = [1,2,3,6,9,18]
psnr = []
mse = []
ts = []
for x in down_x:
    print(f"Running with {x}x reduced sampling")
    if x == 1:
        t, m, p = new_render(img_dir, fast=False,r2=128,d=3)
    else:
        t, m, p = new_render(img_dir, fast=True, r2=192, d=x)
    psnr.append(psnr)
    mse.append(mse)
    ts.append(t)
res = {}
res['down_x'] = [x**2 for x in down_x]
res['psnr'] = psnr
res['mse'] = mse
res['time'] = ts
print(res)

with open(os.path.join(res_dir, 'results.txt'), 'w') as outfile:
        json.dump(res,outfile)
        
fig, ax = plt.subplots(1,1)
fig.suptitle('Accuracy vs Sampling Rate')
ax.set_xlabel('Sampling Rate (1/x sampled)')
ax.set_ylabel('PSNR')
plt.xscale('log')
ax.plot(res['down_x'][1:],res['psnr'][1:], label="Fast NeRF")
ax.scatter([res['down_x'][0]], res['psnr'][0], c="red", label="NeRF")
ax.legend()
plt.savefig(os.path.join(res_dir, 'sampling_rate_accuracy.png'))

fig, ax = plt.subplots(1,1)
fig.suptitle('Accuracy vs Time')
ax.set_xlabel('Running Time (seconds)')
ax.set_ylabel('PSNR')
plt.xscale('log')
ax.plot(res['time'][1:],res['psnr'][1:], label="Fast NeRF")
ax.scatter([res['time'][0]], res['psnr'][0], c="red", label="NeRF")
ax.legend()
plt.savefix(os.path.join(res_dir, 'time_accuracy.png'))



