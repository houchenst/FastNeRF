import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Model architecture

def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# For scene vizualization
def plot_scene(ax, points, poses, hwfs):
    '''
    Plots camera centers, boxes showing their view directions, and 
    3D world points on a provided matplotlib axis.

    Arguments-
    ax     - a matplotlib axis
    points - 3D world points
    poses  - c2w camera matrices
    hwfs   - height, width, focal length of camera views 
    '''

    ax.scatter(points[:,0], points[:,1], points[:,2], c='C0', alpha=0.8)
    for i in range(len(poses)):
        c2w = poses[i]
        H, W, focal = hwfs[i]
        os, ds = get_rays_np(H,W,focal, c2w)
        
        # plot center
        center = os[0,0]
        ax.scatter([center[0]], [center[1]], [center[2]], c='red', alpha=0.5)

        # plot view bounds
        xs = [ds[0,0,0], ds[0,-1,0],ds[-1,-1,0],ds[-1,0,0],ds[0,0,0]]
        ys = [ds[0,0,1], ds[0,-1,1],ds[-1,-1,1],ds[-1,0,1],ds[0,0,1]]
        zs = [ds[0,0,2], ds[0,-1,2],ds[-1,-1,2],ds[-1,0,2],ds[0,0,2]]
        ax.plot(xs, ys, zs, c='red')

    #fix the scaling 
    ax.autoscale_view(tight=None, scalex=False, scaley=False, scalez=True)

    x_bound = ax.get_xlim()
    y_bound = ax.get_ylim()
    new_bound = (min(x_bound[0], y_bound[0]), max(x_bound[1], y_bound[1]))
    ax.set_xlim(left=new_bound[0], right=new_bound[1])
    ax.set_ylim(bottom=new_bound[0], top=new_bound[1])

# Ray helpers

def get_rays(H, W, focal, c2w, override_dirs=None):
    """Get ray origins, directions from a pinhole camera."""
    print(f"HW: ({H},{W})")
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    if override_dirs is not None:
        dirs = override_dirs
    print(dirs)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    print(rays_d)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    print(f"origins: {rays_o.shape}")
    print(f"dirs:    {rays_d.shape}")
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w, override_dirs=None):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    if override_dirs is not None:
        dirs = override_dirs
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    print("RAYS")
    # print(rays_d[1])
    # print(rays_d[50])
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)
    print(rays_d[...,2])
    return rays_o, rays_d

def ndc_points(H, W, focal, near, points):
    '''
    Convert 3d world points to normalized device coordinates
    '''
    # Projection
    o0 = -1./(W/(2.*focal)) * points[..., 0] / points[..., 2]
    o1 = -1./(H/(2.*focal)) * points[..., 1] / points[..., 2]
    o2 = 1. + 2. * near / points[..., 2]

    points = tf.stack([o0, o1, o2], -1)
    return points


def mat_bilinear_interpolation(known, H, W, x_offset, y_offset, gap):
    '''
    Performs bilear interpolation for Z-2 dimensional points, given a set of Z dimensional vectors
    where the first two dimensions are x and y positions
    Arguments:
        known    - the known points arranged in a grid with a distance of size <gap> between
                   points in the x and y directions. Should start at (0,0)
        H        - x upper bound for interpolated points
        W        - y upper bound for interpolated points
        x_offset - the x index of the first desired point (subsequent points will be spaced by gap)
        y_offset - the y index of the first desired point (subsequent points will be spaced by gap)
        gap      - The x and y gap size between points for the known grid and the desired grid of points

    Returns:
        A grid of points, gap distance apart in x and y, interpolated from the known points

    NOTE: No longer used
    '''

    num_y = H // gap + (H%gap > y_offset)
    num_x = W // gap + (W%gap > x_offset)
    new_shape = (num_y, num_x) + tuple(known.shape[2:])
    interpolated = np.zeros(new_shape)

    # interpolate in the x direction
    x_blend = (1 - x_offset/gap) * known[:, :-1, ...] + (x_offset/gap) * known[:, 1:, ...]

    # now blend in the y direction
    full_blend = (1 - y_offset/gap) * x_blend[:-1, ...] + (y_offset/gap) * x_blend[1:, ...]

    interpolated[:full_blend.shape[0], :full_blend.shape[1], ...] = full_blend

    # handle points that aren't between four other points

    # bottom row points
    if num_y > full_blend.shape[0]:
        bottom_row = (1 - x_offset/gap) * known[-1:, :-1, ...] + (x_offset/gap) * known[-1:, 1:, ...]
        interpolated[-1:, :bottom_row.shape[1], ...] = bottom_row
    
    # far right points
    if num_x > full_blend.shape[1]:
        right_column = (1 - y_offset/gap) * known[:-1, -1:, ...] + (y_offset/gap) * known[1:, -1:, ...]
        interpolated[:right_column.shape[0], -1:, ...] = right_column

    # bottom right corner
    if num_y > full_blend.shape[0] and num_x > full_blend.shape[1]:
        interpolated[-1,-1,...] = known[-1, -1, ...]

    return interpolated


def weighted_sampling_interpolation(known_z_vals, weights, H, W, x_offset, y_offset, gap, samples, det=False):
    '''
    Produces samples for a new ray by drawing from the pdfs of neighboring rays with known
    pdfs. Number of samples drawn from neighboring pdfs depends on their distance.
    Arguments:
        known_z_vals - an array with the z_values of the known pdfs
        weights      - an array with the weights of the known pdfs
        H            - height of the image being reconstructed (this method is a subroutine of render())
        W            - width of the image being reconstructed
        x_offset     - the x coordinate of the first point being interpolated
        y_offset     - the y coordinate of the first point being interpolated
        gap          - the x and y gap between points to interpolate (and the known points)
        samples      - the total number of samples to draw
    '''

    num_y = H // gap + (H%gap > y_offset)
    num_x = W // gap + (W%gap > x_offset)
    new_shape = (num_y, num_x) + (samples,)
    interpolated = np.zeros(new_shape)

    vertical = (x_offset == 0)
    horizontal = (y_offset == 0)

    # duplicate the last row/column of the values if necessary 
    # (we may want to interpolate somewhere where there aren't four surrounding points)
    far_bottom = (known_z_vals.shape[0] - 1 < num_y)
    far_right = (known_z_vals.shape[1] - 1 < num_x)

    # size of the known grid
    known_y = known_z_vals.shape[0]
    known_x = known_z_vals.shape[1]

    new_z_shape = (known_y + far_bottom, known_x + far_right) + tuple(known_z_vals.shape[2:])
    new_weights_shape = (known_y + far_bottom, known_x + far_bottom) + tuple(weights.shape[2:])
    new_zs = np.zeros(new_z_shape)
    new_weights = np.zeros(new_weights_shape)

    # fill the new, larger, arrays
    new_zs[:known_y, :known_x, ...] = known_z_vals
    new_weights[:known_y, :known_x, ...] = weights

    if far_bottom:
        new_zs[-1:, :known_x, ...] = known_z_vals[-1:, :, ...]
        new_weights[-1:, :known_x, ...] = weights[-1:, :, ...]
    
    if far_right:
        new_zs[:known_y, -1:, ...] = known_z_vals[:,-1:, ...]
        new_weights[:known_y, -1:, ...] = weights[:,-1:, ...]

    if far_bottom and far_right:
        new_zs[-1:, -1:, ...] = known_z_vals[-1:, -1:, ...]
        new_weights[-1:, -1:, ...] = weights[-1:, -1:, ...]

    known_z_vals = tf.convert_to_tensor(new_zs, dtype=tf.float32)
    weights = tf.convert_to_tensor(new_weights, dtype=tf.float32)


    # if directly between two points vertically
    if vertical:
        top_left = round((1 - y_offset/gap) * samples)
        bottom_left = round((y_offset/gap) * samples)
    elif horizontal:
        top_left = round((1 - x_offset/gap) * samples)
        top_right = round((x_offset/gap) * samples)
    else:
        top_left = int((1 - x_offset/gap) * (1 - y_offset/gap) * samples)
        top_right = int((x_offset/gap) * (1 - y_offset/gap) * samples)
        bottom_left = int((1 - x_offset/gap) * (y_offset/gap) * samples)
        bottom_right = int((x_offset/gap) * (y_offset/gap) * samples)

        # randomly assign remainders
        for i in range(samples - top_left - top_right - bottom_left - bottom_right):
            ray = random.randint(0,3)
            if ray == 0:
                top_left += 1
            elif ray == 1:
                top_right += 1
            elif ray == 2:
                bottom_left += 1
            elif ray == 3:
                bottom_right += 1

    # sample the four distributions
    top_left_samples = sample_pdf(known_z_vals[:-1, :-1, ...], weights[:-1,:-1,...], top_left, det=det)
    if not vertical:
        top_right_samples = sample_pdf(known_z_vals[:-1,1:, ...], weights[:-1,1:,...], top_right, det=det)
    if not horizontal:
        bottom_left_samples = sample_pdf(known_z_vals[1:, :-1, ...], weights[1:,:-1,...], bottom_left, det=det)
    if not (vertical or horizontal):
        bottom_right_samples = sample_pdf(known_z_vals[1:, 1:, ...], weights[1:,1:,...], bottom_right, det=det)

    # combine the samples
    if vertical:
        all_points = [top_left_samples, bottom_left_samples]
    elif horizontal:
        all_points = [top_left_samples, top_right_samples]
    else:
        all_points = [top_left_samples, top_right_samples, bottom_left_samples, bottom_right_samples]
    combined_samples = tf.sort(tf.concat(all_points, -1), -1)
    interpolated[:combined_samples.shape[0], :combined_samples.shape[1], ...] = combined_samples

    #This should be taken care of at the beginning now 

    # # if we want to interpolate below the known points
    # far_bottom = combined_samples.shape[0] < num_y
    # if far_bottom:
    #     left = round((1-x_offset/gap) * samples)
    #     right = round((x_offset/gap) * samples)
    #     left_samples = sample_pdf(known_z_vals[-1:,:-1, ...], weights[-1:,:-1,...], left, det=det)
    #     right_samples = sample_pdf(known_z_vals[-1:,1:,...], weights[-1:,1:,...], right, det=det)
    #     combined_samples = tf.sort(tf.concat([left_samples, right_samples], -1), -1)
    #     interpolated[-1:, :combined_samples.shape[1], ...] = combined_samples

    # # if we want to interpolate to the right of the known points
    # far_right = combined_samples.shape[1] < num_x
    # if far_right:
    #     top = round((1-y_offset/gap) * samples)
    #     bottom = round((y_offset/gap) * samples)
    #     top_samples = sample_pdf(known_z_vals[:-1,-1:, ...], weights[:-1,-1:,...], top, det=det)
    #     bottom_samples = sample_pdf(known_z_vals[1:,-1:,...], weights[1:,-1:,...], bottom, det=det)
    #     combined_samples = tf.sort(tf.concat([top_samples, bottom_samples], -1), -1)
    #     interpolated[:combined_samples.shape[0],-1:,...] = combined_samples

    # # handle far bottom right point if necessary
    # if far_bottom and far_right:
    #     interpolated[-1:,-1:,...] = sample_pdf(known_z_vals[-1:,-1:, ...], weights[-1:,-1:, ...], samples, det=det)

    return tf.cast(interpolated, dtype=tf.float32)
    


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):\
    # NOTE: had to cast a bunch of stuff to tf.double for some reason

    weights = tf.cast(weights, dtype=tf.float64)
    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)
    cdf = tf.cast(cdf, dtype=tf.double)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    u = tf.cast(u, dtype=tf.double)

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    t = tf.cast(t, dtype=tf.double)
    bins_g = tf.cast(bins_g, dtype=tf.double)
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    # back to float
    samples = tf.cast(samples, dtype=tf.float32)
    return samples
