import numpy as np
import torch
from torch.utils.data import Dataset
from networks.libs.base_lib import create_3D_rotations

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B) the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/


class PointCloudDataset(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, opt):
        """
        Initialize parameters of the dataset here.
        """
        self.config = opt
        self.neighborhood_limits = {'train': [51, 42, 35, 31, 30],
                                    'val': [51, 42, 35, 31, 30],
                                    'test': [51, 42, 35, 31, 30]
                                    }

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return 0

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """

        return 0

    def big_neighborhood_filter(self, neighbors, layer, mode):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits[mode]) > 0:
            return neighbors[:, :self.neighborhood_limits[mode][layer]]
        else:
            return neighbors

    def descriptor_inputs(self, stacked_points, stacked_features, stack_lengths, mode):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            # deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    # deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    # deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points), mode)
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points), mode)
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points)+1, mode)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples
        li += [stacked_features]

        return li


class ThreeDMatchCustomBatch:
    """Custom batch definition with memory pinning"""

    def __init__(self, input_list):
        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.stack_lengths = torch.from_numpy(input_list[ind])
        ind += 1
        self.anc_keypts_inds = torch.squeeze(torch.from_numpy(input_list[ind]))
        ind += 1
        self.pos_keypts_inds = torch.squeeze(torch.from_numpy(input_list[ind]))
        ind += 1
        self.backup_points = torch.from_numpy(input_list[ind])
        ind += 1
        self.anc_id = input_list[ind][0]
        self.pos_id = input_list[ind][1]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.features = self.features.pin_memory()
        self.stack_lengths = self.stack_lengths.pin_memory()
        self.anc_keypts_inds = self.anc_keypts_inds.pin_memory()
        self.pos_keypts_inds = self.pos_keypts_inds.pin_memory()
        self.backup_points = self.backup_points.pin_memory()

        return self

    def to(self, device):
        self.points = [in_tensor.to(device, non_blocking=True) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device, non_blocking=True) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device, non_blocking=True) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device, non_blocking=True) for in_tensor in self.upsamples]
        self.features = self.features.to(device, non_blocking=True)
        self.stack_lengths = self.stack_lengths.to(device, non_blocking=True)
        self.anc_keypts_inds = self.anc_keypts_inds.to(device, non_blocking=True)
        self.pos_keypts_inds = self.pos_keypts_inds.to(device, non_blocking=True)
        self.backup_points = self.backup_points.to(device, non_blocking=True)

        return self


class P2NETCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):
        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.images = torch.from_numpy(input_list[ind])
        ind += 1
        self.valid_depth_mask = [torch.from_numpy(nparray) for nparray in input_list[ind]]
        ind += 1
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.stack_lengths = torch.from_numpy(input_list[ind])
        ind += 1
        self.anc_keypts_inds = [torch.from_numpy(nparray) for nparray in input_list[ind]]
        ind += 1
        self.pos_keypts_inds = [torch.from_numpy(nparray) for nparray in input_list[ind]]
        ind += 1
        self.backup_points = torch.from_numpy(input_list[ind])
        ind += 1

        self.anc_id = input_list[ind][:, 0]
        self.pos_id = input_list[ind][:, 1]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.images = self.images.pin_memory()
        self.valid_depth_mask = [in_tensor.pin_memory() for in_tensor in self.valid_depth_mask]
        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.features = self.features.pin_memory()
        self.stack_lengths = self.stack_lengths.pin_memory()
        self.anc_keypts_inds = [in_tensor.pin_memory() for in_tensor in self.anc_keypts_inds]
        self.pos_keypts_inds = [in_tensor.pin_memory() for in_tensor in self.pos_keypts_inds]
        self.backup_points = self.backup_points.pin_memory()

        return self

    def to(self, device):
        self.images = self.images.to(device, non_blocking=True)
        self.valid_depth_mask = [in_tensor.to(device, non_blocking=True) for in_tensor in self.valid_depth_mask]
        self.points = [in_tensor.to(device, non_blocking=True) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device, non_blocking=True) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device, non_blocking=True) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device, non_blocking=True) for in_tensor in self.upsamples]
        self.features = self.features.to(device, non_blocking=True)
        self.stack_lengths = self.stack_lengths.to(device, non_blocking=True)
        self.anc_keypts_inds = [in_tensor.to(device, non_blocking=True) for in_tensor in self.anc_keypts_inds]
        self.pos_keypts_inds = [in_tensor.to(device, non_blocking=True) for in_tensor in self.pos_keypts_inds]
        self.backup_points = self.backup_points.to(device, non_blocking=True)

        return self


def ThreeDMatchCollate(batch_data):
    return ThreeDMatchCustomBatch(batch_data)


def P2NETCollate(batch_data):
    return P2NETCustomBatch(batch_data)
