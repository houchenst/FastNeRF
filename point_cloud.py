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

import run_nerf
from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def load_nerf(args):

    _, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

    # The point cloud functionality should only be used for ndc image sets
    near = 0.
    far = 1.

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs_test.update(bds_dict)

    render_kwargs_fast = {k : render_kwargs_test[k] for k in render_kwargs_test}
    render_kwargs_fast['N_importance'] = 128
    return render_kwargs_fast


def make_point_cloud(hwf, poses, i_train, args, render_kwargs, down=8):
    '''
    Makes 3D point cloud using estimated depth data from images

    '''
    near = 1.
    H, W, focal = hwf
    # use only the training images
    all_points = []
    centers = []
    for i in range(1):
        print(f'Working on image #{i+1}')
        c2w = poses[i_train[i]]
        centers.append(c2w[np.newaxis,:3, -1])
        rays_o, rays_d = get_rays(H//down, W//down, focal, c2w)

        res = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs)
        t_prime = res[3]['depth_map']
        # plt.imshow(t_prime)
        # plt.show()

        # convert to numpy
        rays_o, rays_d, t_prime = rays_o.numpy(), rays_d.numpy(), t_prime.numpy()
        oz, dz = rays_o[...,2], rays_d[...,2]

        # accounting for shifted origin before ndc conversion
        tn = -(near + oz) / dz
        # plt.imshow(tn)
        # plt.show()
        # print(tn)
        on = rays_o + tn[..., np.newaxis] * rays_d
        # print("RAYO")
        # print(rays_o)
        # print("RAY_D")
        # print(rays_d)
        # print("ON")
        # print(on)
        oz = on[...,2]
        # plt.imshow(oz)
        # plt.show()
        # solve for t given t prime using equation 15 from the paper
        # t_prime should be the ndc ray depth, while t is the real world ray depth

        t = (-1. * t_prime) + 1.
        # plt.imshow(t)
        # plt.show()
        t = 1. / t
        # plt.imshow(t)
        # plt.show()
        t = oz * t
        # plt.imshow(t)
        # plt.show()
        t = t - oz
        # plt.imshow(t)
        # plt.show()
        t = t / dz
        # plt.imshow(t)
        # plt.show()

        # get point cloud
        points_3d = on + t[..., np.newaxis] * rays_d

        points_3d = points_3d.reshape((-1,3))

        all_points.append(points_3d)

    all_points = np.concatenate(all_points, axis=0)
    centers = np.concatenate(centers, axis=0)
    np.save("./newpointcloud.npy", all_points)
    np.save("./camcenters.npy", centers)


    # # plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2])

    # plt.show()
        
        




if __name__ == "__main__":
    basedir = './logs'
    # NOTE: select what to make the point cloud for
    expname = 'fern_example'
    config = os.path.join(basedir, expname, 'config.txt')
    parser = run_nerf.config_parser()

    weights_name = 'model_200000.npy'
    args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, weights_name)))

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

    # get the test and train sets
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test)])

    render_kwargs = load_nerf(args)

    make_point_cloud(hwf, poses, i_train, args, render_kwargs)

