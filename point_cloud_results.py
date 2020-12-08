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

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

# profiling
import cProfile
import pstats
from pstats import SortKey

# CELL 1
def get_data():
    basedir = './logs'
    expname = 'fern_example'

    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())
    parser = run_nerf.config_parser()

    weights_name = 'model_200000.npy'
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

    near = 0.
    far = 1.

    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    _, render_kwargs, start, grad_vars, models = run_nerf.create_nerf(args)

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs)

    results = {}
    results['pc'] = {}
    results['no_pc'] = {}

    # NOTE: Where to output results!
    result_directory = "./fern_pc_results"
    img_dir = os.path.join(result_directory, "imgs")

    down = 1

    plt.imsave(os.path.join(img_dir, f"GT{i_test[0]}.png"), images[i_test[0]])
    plt.imsave(os.path.join(img_dir, f"GT{i_test[1]}.png"), images[i_test[1]])

    for num_samps in [4,8,16,32,64]:
        print(f'Running {num_samps} sample test')
        for pc in [True, False]:
            print(f'{"not " if not pc else ""}using pc')
            results['pc' if pc else 'no_pc'][num_samps] = {}
            render_kwargs['N_samples'] = num_samps
            render_kwargs['N_importance'] = 2*num_samps

            total_time = 0
            total_mse = 0
            total_psnr = 0
            for i in [i_test[0], i_test[1]]:
                gt = images[i]
                start_time = time.time()
                ret_vals = run_nerf.render(H//down, W//down, focal/down, c2w=poses[i], pc=pc, cloudsize=16, **render_kwargs)
                end_time = time.time()
                
                # add to cum time
                total_time += (end_time - start_time)
                
                # add to accuracy
                img = np.clip(ret_vals[0],0,1)
                # TODO: make sure this is commented out for real results (just used to test that it runs)
                # mse = run_nerf.img2mse(np.zeros((H//down, W//down,3), dtype=np.float32), img)
                mse = run_nerf.img2mse(gt, img)
                psnr = run_nerf.mse2psnr(mse)
                total_mse += float(mse)
                total_psnr += float(psnr)

                plt.imsave(os.path.join(img_dir, f'IMG{i}_{"pc" if pc else "no_pc"}_{num_samps}samples.png'), img)

            total_time /= 2.
            total_mse /= 2.
            total_psnr /= 2.
            results['pc' if pc else 'no_pc'][num_samps]['time'] = total_time
            results['pc' if pc else 'no_pc'][num_samps]['mse'] = total_mse
            results['pc' if pc else 'no_pc'][num_samps]['psnr'] = total_psnr

    with open(os.path.join(result_directory, 'results.txt'), 'w') as outfile:
        json.dump(results,outfile)

def plot_data():
    '''
    Plot the data from results.txt in the given results directory
    '''
    
    res_dir = "./cloud_size_test"

    with open(os.path.join(res_dir, 'results.txt')) as json_file:
        res = json.load(json_file)

        pc_time = []
        nopc_time = []
        pc_psnr = []
        nopc_psnr = []
        for num_samps in [4,8,16,32,64]:
            pc_time.append(res['pc'][num_samps]['time'])
            nopc_time.append(res['no_pc'][num_samps]['time'])
            pc_psnr.append(res['pc'][num_samps]['psnr'])
            pc_psnr.append(res['no_pc'][num_samps]['psnr'])

        fig, ax = plt.subplots(1,1)
        fig.suptitle('PSNR vs Point Cloud Size')
        ax.set_xlabel('Cloud Size')
        ax.set_ylabel('PSNR')
        plt.xscale('log')
        ax.plot(res['cloud_size'],res['psnr'])
        plt.savefig(os.path.join(res_dir, 'cs_psnr.png'))

        fig, ax = plt.subplots(1,1)
        fig.suptitle('PSNR vs Running Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('PSNR')
        plt.xscale('log')
        ax.plot(res['time'],res['psnr'])
        plt.savefig(os.path.join(res_dir, 'time_psnr.png'))




def cloud_size_vs_performance():
    basedir = './logs'
    expname = 'fern_example'

    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())
    parser = run_nerf.config_parser()

    weights_name = 'model_200000.npy'
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

    near = 0.
    far = 1.

    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    _, render_kwargs, start, grad_vars, models = run_nerf.create_nerf(args)
    to_use = i_test[0]

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs)

    res_dir = "./cloud_size_test"

    res = {}
    res['cloud_size'] = []
    res['mse'] = []
    res['psnr'] = []
    res['time'] = []

    for i in [1,2,4,8,16,32]:
        print(f'Running with cloud downsampled {i}x')
        start_time = time.time()
        ret_vals = run_nerf.render(H, W, focal, c2w=poses[to_use], pc=True, cloudsize=i, **render_kwargs)
        end_time = time.time()
        img = np.clip(ret_vals[0],0,1)
        mse = run_nerf.img2mse(images[to_use], img)
        psnr = run_nerf.mse2psnr(mse)
        res['cloud_size'].append((17 * H * W) // (i * i))
        res['mse'].append(float(mse))
        res['psnr'].append(float(psnr))
        res['time'].append(end_time - start_time)

    # a = [1,2,4,8,16,32]
    # b = [1/x for x in a]

    # make plots
    # cs vs psnr
    fig, ax = plt.subplots(1,1)
    fig.suptitle('PSNR vs Point Cloud Size')
    ax.set_xlabel('Cloud Size')
    ax.set_ylabel('PSNR')
    plt.xscale('log')
    ax.plot(res['cloud_size'],res['psnr'])
    plt.savefig(os.path.join(res_dir, 'cs_psnr.png'))

    fig, ax = plt.subplots(1,1)
    fig.suptitle('PSNR vs Running Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('PSNR')
    plt.xscale('log')
    ax.plot(res['time'],res['psnr'])
    plt.savefig(os.path.join(res_dir, 'time_psnr.png'))

    fig, ax = plt.subplots(1,1)
    fig.suptitle('Running Time vs Cloud Size')
    ax.set_xlabel('Cloud Size')
    ax.set_ylabel('Running Time')
    plt.xscale('log')
    plt.yscale('log')
    ax.plot(res['cloud_size'],res['time'])
    plt.savefig(os.path.join(res_dir, 'cs_time.png'))
    
    with open(os.path.join(res_dir, 'results.txt'), 'w') as outfile:
        json.dump(res,outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the point cloud method vs vanilla NeRF')
    parser.add_argument('--generate',  action='store_true', help='generate data')
    parser.add_argument('--cloudsize', action='store_true', help='compare gains from larger cloud size')
    parser.add_argument('--plot', action='store_true', help='plot existing data')
    args = parser.parse_args()
    
    if args.generate:
        get_data()

    if args.cloudsize:
        cloud_size_vs_performance()

    if args.plot:
        plot_data()