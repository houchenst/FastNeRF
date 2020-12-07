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

def new_views(hwf, poses, i_test, args, render_kwargs, down=32):

    near = 1.
    H, W, focal = hwf
    world_points = np.load("./newpointcloud.npy")

    

    # pose 2
    c2w = poses[0]
    print(f'recovering with {c2w}')
    points_o = np.broadcast_to(c2w[:3, -1], np.shape(world_points))
    camera_points = np.sum(world_points[..., np.newaxis, :] * np.linalg.inv(c2w[:3, :3]), -1)
    camera_points = camera_points - points_o
    cam_normed = camera_points / (-camera_points[:,2:])

    _,cam_rays = get_rays_np(H//down, W//down, focal/down, c2w, override_dirs=cam_normed)

    # temp = np.copy(cam_rays[:,1])
    # cam_rays[:,1] = cam_rays[:,0]
    # cam_rays[:,0] = temp

    print("DEPTH")
    depth = (world_points / cam_rays)[:,2]
    # print(depth[0])
    # print(depth[100])
    # cam_rays[:,2] = -depth

    _, ndc_cloud = ndc_rays(H//down, W//down, focal/down, near, points_o, cam_rays)
    ndc_cloud = ndc_cloud.numpy()
    disp = 1. / depth
    ndc_cloud[:,2] = disp

    # for i in range(3):
    #         print(f'Original: {cam_rays[i*50]} \t NDC: {ndc_cloud[i*50]}')

    gen_os, generated_rays = get_rays_np(H//down, W//down, focal/down, c2w)
    _, generated_ndc = ndc_rays(H//down, W//down, focal/down, near, gen_os, generated_rays)
    generated_ndc = generated_ndc.numpy()
    low_x = np.min(generated_ndc[...,0])
    high_x = np.max(generated_ndc[...,0])
    low_y = np.min(generated_ndc[...,1])
    high_y = np.max(generated_ndc[...,1])

    cloud_low_x = np.min(ndc_cloud[...,0])
    cloud_high_x = np.max(ndc_cloud[...,0])
    cloud_low_y = np.min(ndc_cloud[...,1])
    cloud_high_y = np.max(ndc_cloud[...,1])

    print("generated")
    print(f'x- {low_x},{high_x}')
    print(f'y- {low_y}{high_y}')
    print("cloud")
    print(f'x- {cloud_low_x}{cloud_high_x}')
    print(f'y- {cloud_low_y}{cloud_high_y}')

    near_bound = np.ones((H//down, W//down)) *-1.
    far_bound = np.ones((H//down, W//down)) * 2.

    pix_height = (high_y - low_y) / (H//down)
    pix_width = (high_x - low_x) / (W//down)
    
    # rays are center of pixels, adjust bounds accordingly
    low_x = low_x - (pix_width * 0.5)
    high_x = high_x + (pix_width * 0.5)
    low_y = low_y - (pix_height * 0.5)
    high_y = high_y + (pix_height * 0.5)

    # check every point in the cloud to update the bounds
    for a in range(ndc_cloud.shape[0]):
        p = ndc_cloud[a]
        # print(p)
        # ndc x and y ranges are -1 to 1
        # coords are x,y,z
        i = int(W//down - (p[0] - low_x) // pix_width)
        j = int((p[1] - low_y) / pix_height)
        # print(f'i: {i}/{H//down}')
        # print(f'j: {j}/{W//down}')
        if i < W//down and i >= 0 and j < H//down and j >= 0:
            # compare to near bound
            if p[2] > near_bound[j,i]:
                near_bound[j,i] = p[2]
            # compare to far_bound
            if p[2] < far_bound[j,i]:
                far_bound[j,i] = p[2]
    
    # max bounds if they haven't been updated
    near_bound[near_bound < -0.1] = 1.
    far_bound[far_bound > 1.1] = 0.

    near_bound = 1. - near_bound
    far_bound = 1. - far_bound

    plt.imshow(far_bound)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot_scene(ax, world_points, [poses[x] for x in range(poses.shape[0])], [hwf for x in range(poses.shape[0])])
    plot_scene(ax, cam_rays, [poses[1]], [hwf])

    # for i in range(3):
    #         print(f'Original: {generated_rays[i*30]} \t NDC: {generated_ndc[i*30]}')
    # ax.scatter(generated_rays[...,0], generated_rays[...,1], generated_rays[...,2], c='orange', alpha=1.0)
    
    plt.show()

    for i in i_test:
        break
        c2w = poses[i]

        # recenter points around camera center
        points_o = np.broadcast_to(c2w[:3, -1], np.shape(world_points))

        points_d = np.copy(world_points) #- points_o
        dz = points_d[:,2:]


        points_d = points_d / (-dz)
        # points_d[:,0] = points_d[:,0] / ((W * 0.5)/focal)
        # points_d[:,1] = points_d[:,1] / ((H * 0.5)/focal)

        # dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
        # print(dirs)
        # points_d = tf.reduce_sum(points_d[..., np.newaxis, :] * c2w[:3, :3], -1)
        # print(rays_d)
        # rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))

        # disp = 1 / (-dz)
        # points_d[:,2:] = disp
        # norm_d = points_d / dz

        # print(norm_d)
        
        # get ndc coords
        _, ndc_cloud = ndc_rays(H, W, focal, near, points_o, points_d)
        ndc_cloud = ndc_cloud.numpy()
        # ndc_cloud[:,2:] = disp
        
        # make near and far depth maps for each pixel in the new view
        # valid range for ndc depth is 0->1
        # we will intialize the values outside of this range so that we 
        # can tell which ones have been assigned
        near_bound = np.ones((H//down, W//down)) * 2.
        far_bound = np.ones((H//down, W//down)) * -1.

        pix_height = 2. / (H//down)
        pix_width = 2. / (W//down)

        for i in range(3):
            print(f'Original: {points_d[i*50]} \t NDC: {ndc_cloud[i*50]}')

        if False:
            # check every point in the cloud to update the bounds
            for a in range(ndc_cloud.shape[0]):
                p = ndc_cloud[a]
                print(p)
                # ndc x and y ranges are -1 to 1
                # coords are x,y,z
                i = int((p[1] + 1.) // pix_height)
                j = int((p[0] + 1.) // pix_width)
                print(f'i: {i}/{H//down}')
                print(f'j: {j}/{W//down}')
                # compare to near bound
                if p[2] < near_bound[i,j]:
                    near_bound[i,j] = p[2]
                # compare to far_bound
                if p[2] > far_bound[i,j]:
                    far_bound[i,j] = p[2]
            
            # max bounds if they haven't been updated
            near_bound[near_bound > 1.1] = 0.
            far_bound[far_bound < -0.1] = 1.

            plt.imshow(near_bound* 255)
            plt.show()


def make_point_cloud(hwf, poses, i_train, args, render_kwargs, down=32):
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
        # c2w = poses[i_train[i]]
        c2w = poses[i]
        print(f"Using {c2w}")
        print(c2w)
        centers.append(c2w[np.newaxis,:3, -1])
        rays_o, rays_d = get_rays(H//down, W//down, focal/down, c2w)

        res = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs)
        t_prime = res[3]['depth_map']
        plt.imshow(t_prime)
        plt.show()

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

    new_views(hwf, poses, i_train, args, render_kwargs)

