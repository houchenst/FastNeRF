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
from run_nerf_helpers_fast import *

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

def depth_bounds(hwf, c2w):
    epsilon = 0.10

    near = 1.
    H, W, focal = hwf
    world_points = np.load("./newpointcloud.npy")

    # find directional rays from camera center towards world points
    points_o = np.broadcast_to(c2w[:3, -1], np.shape(world_points))
    camera_points = np.sum(world_points[..., np.newaxis, :] * np.linalg.inv(c2w[:3, :3]), -1)
    camera_points = camera_points - points_o
    cam_normed = camera_points / (-camera_points[:,2:])

    # get rays from directions
    _,cam_rays = get_rays_np(H, W, focal, c2w, override_dirs=cam_normed)

    # get ndc points and set z equal to disparity
    depth = (world_points / cam_rays)[:,2]
    _, ndc_cloud = ndc_rays(H, W, focal, near, points_o, cam_rays)
    ndc_cloud = ndc_cloud.numpy()
    disp = 1. / depth
    ndc_cloud[:,2] = disp

    # find range of view in ndc coords
    gen_os, generated_rays = get_rays_np(H, W, focal, c2w)
    print(generated_rays)
    _, generated_ndc = ndc_rays(H, W, focal, near, gen_os, generated_rays)
    generated_ndc = generated_ndc.numpy()
    print(generated_ndc)

    low_x = generated_ndc[0,0,0]
    high_x = generated_ndc[0,-1,0]
    low_y = generated_ndc[0,0,1]
    high_y = generated_ndc[-1,0,1]

    # cloud_low_x = np.min(ndc_cloud[...,0])
    # cloud_high_x = np.max(ndc_cloud[...,0])
    # cloud_low_y = np.min(ndc_cloud[...,1])
    # cloud_high_y = np.max(ndc_cloud[...,1])

    # print("generated")
    # print(f'x- {low_x},{high_x}')
    # print(f'y- {low_y}{high_y}')
    # print("cloud")
    # print(f'x- {cloud_low_x}{cloud_high_x}')
    # print(f'y- {cloud_low_y}{cloud_high_y}')

    # find pixel size in ndc space
    near_bound = np.ones((H, W)) *-1.
    far_bound = np.ones((H, W)) * 2.
    pix_height = (high_y - low_y) / (H)
    pix_width = (high_x - low_x) / (W)
    print(f'h: {pix_height}')
    print(f'w: {pix_width}')
    
    # rays are center of pixels, adjust bounds accordingly
    low_x = low_x - (pix_width * 0.5)
    high_x = high_x + (pix_width * 0.5)
    low_y = low_y - (pix_height * 0.5)
    high_y = high_y + (pix_height * 0.5)

    # check every point in the cloud to update the bounds
    for a in range(ndc_cloud.shape[0]):
        p = ndc_cloud[a]
        # find image coords
        i = int((p[0] - low_x) // pix_width)
        j = int((p[1] - low_y) // pix_height)
        # check if image coords are valid
        if i < W and i >= 0 and j < H and j >= 0:
            # compare to near bound
            if p[2] > near_bound[j,i]:
                near_bound[j,i] = p[2]
            # compare to far_bound
            if p[2] < far_bound[j,i]:
                far_bound[j,i] = p[2]
    
    # max bounds if they haven't been updated
    near_bound[near_bound < 0.] = 1.
    far_bound[far_bound > 1.0] = 0.
    # invert since sampling is done 0-1
    near_bound = 1. - near_bound
    far_bound = 1. - far_bound

    near_bound = np.clip(near_bound - epsilon, 0., 1.)
    far_bound = np.clip(far_bound + epsilon, 0., 1.)

    # max and min local region
    translated_near = []
    translated_far = []
    for i in range(-1,2):
        for j in range(-1,2):
            # i = i-1
            # j = j-1
            near_base = np.zeros((H,W))
            far_base = np.ones((H,W))
            if i < 0:
                if j < 0:
                    near_base[:-1,:-1] = near_bound[1:,1:]
                    far_base[:-1,:-1] = far_bound[1:,1:]
                if j == 0:
                    near_base[:-1,:] = near_bound[1:,:]
                    far_base[:-1,:] = far_bound[1:,:]
                if j > 0:
                    near_base[:-1,1:] = near_bound[1:,:-1]
                    far_base[:-1,1:] = far_bound[1:,:-1]
            if i > 0:
                if j < 0:
                    near_base[1:,:-1] = near_bound[:-1,1:]
                    far_base[1:,:-1] = far_bound[:-1,1:]
                if j == 0:
                    near_base[1:,:] = near_bound[:-1,:]
                    far_base[1:,:] = far_bound[:-1,:]
                if j > 0:
                    near_base[1:,1:] = near_bound[:-1,:-1]
                    far_base[1:,1:] = far_bound[:-1,:-1]
            if i == 0:
                if j < 0:
                    near_base[:,:-1] = near_bound[:,1:]
                    far_base[:,:-1] = far_bound[:,1:]
                if j == 0:
                    near_base[:,:] = near_bound[:,:]
                    far_base[:,:] = far_bound[:,:]
                if j > 0:
                    near_base[:,1:] = near_bound[:,:-1]
                    far_base[:,1:] = far_bound[:,:-1]
            translated_near.append(near_base)
            translated_far.append(far_base)
    near_stack = np.stack(translated_near, axis=-1)
    far_stack = np.stack(translated_far, axis=-1)
    near_bound = np.amin(near_stack, axis=-1)
    far_bound = np.amax(far_stack, axis=-1)


    # Show near and far bounds
    # plt.imshow(near_bound, vmin=0, vmax=1)
    # plt.show()

    # plt.imshow(far_bound, vmin=0, vmax=1)
    # plt.show()

    # NOTE: Use to show world points relative to camera view
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plot_scene(ax, world_points, [c2w], [hwf])
    # plt.show()
    return (near_bound, far_bound)


def make_point_cloud(hwf, poses, i_train, args, render_kwargs, down=32):
    '''
    Makes 3D point cloud using estimated depth data from images

    '''
    near = 1.
    H, W, focal = hwf
    # use only the training images
    all_points = []
    centers = []
    for i in range(len(i_train)):
        print(f'Working on image #{i+1}')
        # c2w = poses[i_train[i]]
        c2w = poses[i_train[i]]
        print(f"Using {c2w}")
        print(c2w)
        centers.append(c2w[np.newaxis,:3, -1])
        rays_o, rays_d = get_rays(H//down, W//down, focal/down, c2w)

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
    np.save(f"./pointcloud_down{down}.npy", all_points)
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

    make_point_cloud(hwf, poses, i_train, args, render_kwargs, down=1)
    # down = 4
    # hwf = [H//64, W//64, focal//64]
    # depth_bounds(hwf, poses[i_test[0]])

