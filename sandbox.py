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

import matplotlib
# matplotlib.use('Qt4Agg')


import run_nerf
from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import random


points_3d = np.load("./newpointcloud.npy")
centers = np.load("./camcenters.npy")

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# reduce points for visualization
red_factor = 1
inds = random.sample(range(points_3d.shape[0]), int(points_3d.shape[0]/ red_factor))
points_3d = np.take(points_3d, inds, axis=0)
ax.scatter(points_3d[:, 1], points_3d[:, 0], points_3d[:, 2], c="C0", alpha=0.8)
ax.scatter(centers[:, 1], centers[:,0], centers[:,2], c="red", alpha=0.5)
xs = [-1,-1,1,1,-1]
ys = [-1,1,1,-1,-1]
zs = [-1,-1,-1,-1,-1]
ax.plot(xs,ys,zs)
ax.autoscale_view(tight=None, scalex=False, scaley=False, scalez=True)
plt.show()
