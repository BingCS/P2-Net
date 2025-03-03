import torch
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists

from utils.ply_tools import read_ply, write_ply


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def rnd_sample(inputs, n_sample):  # inputs = [pos1[i], pos2[i]]
    cur_size = inputs[0].size(1)  # inputs[0] = pos1[i] = [[iiiii],[jjjjjj]]
    rnd_idx = torch.randperm(cur_size)[0:n_sample]
    outputs = [a[:, rnd_idx] for a in inputs]
    return outputs


def swap_axis(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def grid_positions(h, w, device):
    rows = torch.arange(0, h, device=device).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(0, w, device=device).view(1, -1).float().repeat(h, 1)
    return torch.cat([rows.view(1, -1), columns.view(1, -1)], dim=0)


def interpolate_depth(pos, depth, device):  # used in get_warp
    ids = torch.arange(0, pos.size(1), device=device)
    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return interpolated_depth, pos, ids


def interpolate(pos, inputs):
    _, h, w = inputs.size()

    i = pos[0, :]
    j = pos[1, :]

    i_top_left = torch.clamp(torch.floor(i).long(), 0, h - 1)
    j_top_left = torch.clamp(torch.floor(j).long(), 0, w - 1)

    i_top_right = torch.clamp(torch.floor(i).long(), 0, h - 1)
    j_top_right = torch.clamp(torch.ceil(j).long(), 0, w - 1)

    i_bottom_left = torch.clamp(torch.ceil(i).long(), 0, h - 1)
    j_bottom_left = torch.clamp(torch.floor(j).long(), 0, w - 1)

    i_bottom_right = torch.clamp(torch.ceil(i).long(), 0, h - 1)
    j_bottom_right = torch.clamp(torch.ceil(j).long(), 0, w - 1)

    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolate_val = (
            w_top_left * inputs[:, i_top_left, j_top_left] +
            w_top_right * inputs[:, i_top_right, j_top_right] +
            w_bottom_left * inputs[:, i_bottom_left, j_bottom_left] +
            w_bottom_right * inputs[:, i_bottom_right, j_bottom_right]
    )

    return interpolate_val  # size: (128, 512) / (channels / num_corr)


def warp(pos, depth1, K1, depth2, K2, rel_pose, batch_size, num_corr, device):
    all_pos1 = []
    all_points2 = []
    for i in torch.arange(batch_size):  # new pos1:(2, 512)(C, N)
        z1, new_pos1, _ = interpolate_depth(pos, depth1[i], device)
        # K1 : (3,3) uv1_homo : (3, N)
        uv1_homo = torch.cat([swap_axis(new_pos1), torch.ones((1, new_pos1.size(1)), device=device)], dim=0)
        xy1_homo = torch.matmul(torch.inverse(K1[i]), uv1_homo)   # (3, N')
        xyz1_homo = torch.cat([torch.unsqueeze(z1, dim=0) * xy1_homo,  # z1 = [N]
                               torch.ones((1, new_pos1.size(1)), device=device)], dim=0)

        xyz2_homo = torch.matmul(rel_pose[i], xyz1_homo)
        xy2_homo = xyz2_homo / torch.unsqueeze(xyz2_homo[-1, :], dim=0)  # (4, N')
        uv2 = torch.matmul(K2[i], xy2_homo)[0:2, :]

        annotated_depth, new_pos2, new_ids = interpolate_depth(swap_axis(uv2), depth2[i], device)
        estimated_depth = xyz2_homo[:, new_ids][-1, :]
        inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

        new_pos1 = new_pos1[:, new_ids]
        new_pos1 = new_pos1[:, inlier_mask]
        new_pos2 = torch.round(new_pos2[:, inlier_mask])
        new_pos1, new_pos2 = rnd_sample([new_pos1, new_pos2], num_corr)

        _, ptcld_pos2, _ = interpolate_depth(pos, depth2[i], device)
        inds = ((new_pos2.transpose(1, 0)[:, None, :] == ptcld_pos2.transpose(1, 0)[None, ...]).all(dim=2)).nonzero()

        all_pos1.append(new_pos1[:, inds[:, 0]])
        all_points2.append(inds[:, 1])

    return all_pos1, all_points2


def get_dist_mat(feat1, feat2, dist_type):
    cos_dist_mat = torch.matmul(feat1.transpose(-1, -2), feat2)
    if dist_type == 'cosine_dist':
        dist_mat = torch.clamp(cos_dist_mat, -1, 1)
    elif dist_type == 'euclidean_dist':
        dist_mat = torch.sqrt(torch.clamp_min(-cos_dist_mat * 2 + 2, min=0))
    elif dist_type == 'euclidean_dist_no_norm':
        norm1 = torch.sum(feat1 * feat1, dim=0, keepdim=True)
        norm2 = torch.sum(feat2 * feat2, dim=0, keepdim=True)
        dist_mat = torch.sqrt(torch.clamp_min(norm1.transpose(-1, -2) - 2 * cos_dist_mat + norm2, min=0))
    else:
        raise NotImplementedError()

    return dist_mat


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))


def spherical_Lloyd(radius, num_cells, dimension=3, fixed='center', approximation='monte-carlo',
                    approx_n=5000, max_iter=500, momentum=0.9, verbose=0):
    """
    Creation of kernel point via Lloyd algorithm. We use an approximation of the algorithm, and compute the Voronoi
    cell centers with discretization  of space. The exact formula is not trivial with part of the sphere as sides.
    :param radius: Radius of the kernels
    :param num_cells: Number of cell (kernel points) in the Voronoi diagram.
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
    :param approx_n: Number of point used for approximation.
    :param max_iter: Maximum nu;ber of iteration for the algorithm.
    :param momentum: Momentum of the low pass filter smoothing kernel point positions
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points (Uniform distribution in a sphere)
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[np.logical_and(d2 < radius0 ** 2, (0.9 * radius0) ** 2 < d2), :]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))

    # Optional fixing
    if fixed == 'center':
        kernel_points[0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    ##############################
    # Approximation initialization
    ##############################

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

    # Initialize discretization in this method is chosen
    if approximation == 'discretization':
        side_n = int(np.floor(approx_n ** (1. / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl/2, radius0, dl)
        if dimension == 2:
            x, y = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            x, y, z = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            x, y, z, t = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError('Unsupported dimension (max is 4)')
    elif approximation == 'monte-carlo':
        X = np.zeros((0, dimension))
    else:
        raise ValueError('Wrong approximation method chosen: "{:s}"'.format(approximation))

    # Only points inside the sphere are used
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]

    #####################
    # Kernel optimization
    #####################

    # Warning if at least one kernel point has no cell
    warning = False

    # moving vectors of kernel points saved to detect convergence
    max_moves = np.zeros((0,))

    for iter in range(max_iter):

        # In the case of monte-carlo, renew the sampled points
        if approximation == 'monte-carlo':
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]

        # Get the distances matrix [n_approx, K, dim]
        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)

        # Compute cell centers
        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = (cell_inds == c)
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                warning = True
                centers.append(kernel_points[c])

        # Update kernel points with low pass filter to smooth mote carlo
        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        # Check moves for convergence
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))

        # Optional fixing
        if fixed == 'center':
            kernel_points[0, :] *= 0
        if fixed == 'verticals':
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

        if verbose:
            print('iter {:5d} / max move = {:f}'.format(iter, np.max(np.linalg.norm(moves, axis=1))))
            if warning:
                print('{:}WARNING: at least one point has no cell{:}'.format(bcolors.WARNING, bcolors.ENDC))
        if verbose > 1:
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0,
                        marker='.', cmap=plt.get_cmap('tab20'))
            # plt.scatter(kernel_points[:, 0], kernel_points[:, 1], c=np.arange(num_cells), s=100.0,
            #            marker='+', cmap=plt.get_cmap('tab20'))
            plt.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_ylim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)

    ###################
    # User verification
    ###################

    # Show the convergence to ask user if this kernel is correct
    if verbose:
        if dimension == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10.4, 4.8])
            ax1.plot(max_moves)
            ax2.scatter(X[:, 0], X[:, 1], c=cell_inds, s=20.0,
                        marker='.', cmap=plt.get_cmap('tab20'))
            # plt.scatter(kernel_points[:, 0], kernel_points[:, 1], c=np.arange(num_cells), s=100.0,
            #            marker='+', cmap=plt.get_cmap('tab20'))
            ax2.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            ax2.add_artist(circle)
            ax2.set_xlim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_ylim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_aspect('equal')
            plt.title('Check if kernel is correct.')
            plt.draw()
            plt.show()

        if dimension > 2:
            plt.figure()
            plt.plot(max_moves)
            plt.title('Check if kernel is correct.')
            plt.show()

    # Rescale kernels with real radius
    return kernel_points * radius


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3,
                                    fixed='center', ratio=0.66, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3/2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10*kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
            fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):

    # Kernel directory
    kernel_dir = 'networks/libs/kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # To many points switch to Lloyds
    if num_kpoints > 30:
        lloyd = True

    # Kernel_file
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D.ply'.format(num_kpoints, fixed, dimension))

    # Check if already done
    if not exists(kernel_file):
        if lloyd:
            # Create kernels
            kernel_points = spherical_Lloyd(1.0,
                                            num_kpoints,
                                            dimension=dimension,
                                            fixed=fixed,
                                            verbose=0)

        else:
            # Create kernels
            kernel_points, grad_norms = kernel_point_optimization_debug(1.0,
                                                                        num_kpoints,
                                                                        num_kernels=100,
                                                                        dimension=dimension,
                                                                        fixed=fixed,
                                                                        verbose=0)

            # Find best candidate
            best_k = np.argmin(grad_norms[-1, :])

            # Save points
            kernel_points = kernel_points[best_k, :, :]

        write_ply(kernel_file, kernel_points, ['x', 'y', 'z'])

    else:
        data = read_ply(kernel_file)
        kernel_points = np.vstack((data['x'], data['y'], data['z'])).T

    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

    elif dimension == 3:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        else:
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)

    # Scale kernels
    kernel_points = radius * kernel_points

    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)
