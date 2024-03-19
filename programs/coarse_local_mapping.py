import argparse
import copy
import time

import numpy as np
import ot
import point_cloud_utils as pcu
import torch
import torch.nn as nn
from fml.nn import SinkhornLoss, pairwise_distances
from scipy.spatial import cKDTree

from programs import utils_overlapping


class MLP(nn.Module):
    """
    A simple fully connected network mapping vectors in dimension in_dim to vectors in dimension out_dim
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def compute_patches(x, n, r, c, angle_thresh=95.0,  min_pts_per_patch=10, devices=('cpu',)):
    """
    Given an input point cloud, X, compute a set of patches (subsets of X) and parametric samples for those patches.
    Each patch is a cluster of points which lie in a ball of radius c * r and share a similar normal.
    The spacing between patches is roughly the radius, r. This function also returns a set of 2D parametric samples
    for each patch. These samples are used to fit a function from the samples to R^3 which agrees with the patch.

    :param x: A 3D point cloud with |x| points specified as an array of shape (|x|, 3) (each row is a point)
    :param n: Unit normals for the point cloud, x, of shape (|x|, 3) (each row is a unit normal)
    :param r: The approximate separation between patches
    :param c: Each patch will fit inside a ball of radius c * r
    :param angle_thresh: If the normal of a point in a patch differs by greater than angle_thresh degrees from the
                        normal of the point at the center of the patch, it is discarded.
    :param min_pts_per_patch: The minimum number of points allowed in a patch
    :param devices: A list of devices on which to store each patch. Patch i is stored on devices[i % len(devices)].
    :return: Two lists, idx and uv, of torch tensors, where uv[i] are the parametric samples (shape = (np, 2)) for
             the i^th patch, and idx[i] are the indexes into x of the points for the i^th patch. i.e. x[idx[i]] are the
             3D points of the i^th patch.
    """

    covered = np.zeros(x.shape[0], dtype=np.bool)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    np.set_printoptions(threshold=np.inf)
    print(x.shape)
    print(n.shape)
    print(r)
    ctr_v, ctr_n = pcu.prune_point_cloud_poisson_disk(x, n, r, best_choice_sampling=True)
    
    if len(ctr_v.shape) == 1:
        ctr_v = ctr_v.reshape([1, *ctr_v.shape])
        ctr_n = ctr_n.reshape([1, *ctr_n.shape])
    kdtree = cKDTree(x)
    ball_radius = c * r
    angle_thresh = np.cos(np.deg2rad(angle_thresh))

    patch_indexes = []
    patch_uvs = []
    patch_xs = []
    patch_transformations = []

    def make_patch(v_ctr, n_ctr):
        idx_i = np.array(kdtree.query_ball_point(v_ctr, ball_radius, p=np.inf))
        good_normals = np.squeeze(n[idx_i] @ n_ctr.reshape([3, 1]) > angle_thresh)
        idx_i = idx_i[good_normals]

        if len(idx_i) < min_pts_per_patch:
            print("Rejecting small patch with %d points" % len(idx_i))
            return

        covered_indices = idx_i[np.linalg.norm(x[idx_i] - v_ctr, axis=1) < r]
        covered[covered_indices] = True

        uv_i = pcu.lloyd_2d(len(idx_i)).astype(np.float32)
        x_i = x[idx_i].astype(np.float32)
        translate_i = -np.mean(x_i, axis=0)

        device = devices[len(patch_xs) % len(devices)]
                
        scale_i = np.array([1.0 / np.max(np.linalg.norm(x_i + translate_i, axis=1))], dtype=np.float32)
        rotate_i, _, _ = np.linalg.svd((x_i + translate_i).T, full_matrices=False)
        transform_i = (torch.from_numpy(translate_i).to(device),
                       torch.from_numpy(scale_i).to(device),
                       torch.from_numpy(rotate_i).to(device))

        x_i = torch.from_numpy((scale_i * (x_i.astype(np.float32) + translate_i)) @ rotate_i).to(device)

        patch_transformations.append(transform_i)
        patch_indexes.append(torch.from_numpy(idx_i))
        patch_uvs.append(torch.tensor(uv_i, device=device, requires_grad=True))
        patch_xs.append(x_i)
        print("Computed patch with %d points" % x_i.shape[0])
        
    for i in range(ctr_v.shape[0]):
        make_patch(ctr_v[i], ctr_n[i])

    for i in range(x.shape[0]):
        if np.sum(covered) == x.shape[0]:
            break
        if not covered[i]:
            make_patch(x[i], n[i])

    print("Found %d neighborhoods" % len(patch_indexes))
    return patch_indexes, patch_uvs, patch_xs, patch_transformations


def patch_means_thetas(patch_pis, patch_uvs, patch_idx, patch_tx, phi, x, weights, output_folder):
    """
    Given a set of charts and pointwise correspondences between charts, compute the mean of the overlapping points in
    each chart. This is used to denoise the Atlas after each chart has beeen individually fitted.
    The charts may not agree exactly on their prediction, so we compute the mean predictions of overlapping charts
    and fit each chart to that mean.

    :param patch_pis: A list of correspondences between the 2D uv samples and the points in a neighborhood
    :param patch_uvs: A list of tensors, each of shape [n_i, 2] of UV positions for the given patch
    :param patch_idx: A list of tensors each of shape [n_i] containing the indices of the points in a neighborhood into
                      the input point-cloud x (of shape [n, 3])
    :param patch_tx: A list of tuples (t_i, s_i, r_i) of transformations (t_i is a translation, s_i is a scaling, and
                     r_i is a rotation matrix) which map the points in a neighborhood to a centered and whitened point
                     set
    :param phi: A list of neural networks representing the lifting function for each chart in the atlas
    :param x: A [n, 3] tensor containing the input point cloud
    :return: A list of tensors, each of shape [n_i, 3] where each tensor is the average prediction of the overlapping
             charts a the samples
    """
    num_patches = len(patch_uvs)
    

    if isinstance(x, np.ndarray):
        mean_pts = torch.from_numpy(x).to(patch_uvs[0])
    elif torch.is_tensor(x):
        mean_pts = x.clone()
    else:
        raise ValueError("Invalid type for x")

    counts = torch.ones(x.shape[0], 1).to(mean_pts)

    for i in range(num_patches):
        translate_i, scale_i, rotate_i = patch_tx[i]
        w_i = torch.from_numpy(weights[i]).to(mean_pts)

        uv_i = patch_uvs[i]
        y_i = ((phi[i](uv_i).squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i)
        #y_i = phi[i](uv_i).squeeze()
        pi_i = patch_pis[i]
        idx_i = torch.tensor(patch_idx[i][pi_i], dtype= torch.long)


        for j in range(y_i.shape[0]):
            mean_pts[idx_i[j]] += w_i[pi_i[j]]*y_i[j].to(mean_pts)
            counts[idx_i[j], :] += w_i[pi_i[j]]


    mean_pts = mean_pts / counts

    #np.savetxt('counts.txt', counts.cpu().detach().numpy(), fmt='%f')
    pcu.write_obj(output_folder + "/" + "updated_input_points_in_second_mapping.obj", mean_pts.cpu().detach().numpy(), np.array([], dtype='int'), np.array([], dtype='float32'))

    means = []
    for i in range(num_patches):
        idx_i = torch.tensor(patch_idx[i], dtype= torch.long)
        translate_i, scale_i, rotate_i = patch_tx[i]
        device_i = translate_i.device
        m_i = scale_i * (mean_pts[idx_i].to(device_i) + translate_i) @ rotate_i
        means.append(m_i)

    return means


def patch_means(patch_pis, patch_uvs, patch_idx, phi, x, weights):
    """
    Given a set of charts and pointwise correspondences between charts, compute the mean of the overlapping points in
    each chart. This is used to denoise the Atlas after each chart has beeen individually fitted.
    The charts may not agree exactly on their prediction, so we compute the mean predictions of overlapping charts
    and fit each chart to that mean.

    :param patch_pis: A list of correspondences between the 2D uv samples and the points in a neighborhood
    :param patch_uvs: A list of tensors, each of shape [n_i, 2] of UV positions for the given patch
    :param patch_idx: A list of tensors each of shape [n_i] containing the indices of the points in a neighborhood into
                      the input point-cloud x (of shape [n, 3])
    :param patch_tx: A list of tuples (t_i, s_i, r_i) of transformations (t_i is a translation, s_i is a scaling, and
                     r_i is a rotation matrix) which map the points in a neighborhood to a centered and whitened point
                     set
    :param phi: A list of neural networks representing the lifting function for each chart in the atlas
    :param x: A [n, 3] tensor containing the input point cloud
    :return: A list of tensors, each of shape [n_i, 3] where each tensor is the average prediction of the overlapping
             charts a the samples
    """
    num_patches = len(patch_uvs)
    

    if isinstance(x, np.ndarray):
        mean_pts = torch.from_numpy(x).to(patch_uvs[0])
    elif torch.is_tensor(x):
        mean_pts = x.clone()
    else:
        raise ValueError("Invalid type for x")

    counts = torch.ones(x.shape[0], 1).to(mean_pts)

    for i in range(num_patches):
        w_i = torch.from_numpy(weights[i]).to(mean_pts)

        uv_i = patch_uvs[i]
        y_i = phi[i](uv_i).squeeze()
        pi_i = patch_pis[i]
        idx_i = torch.tensor(patch_idx[i][pi_i], dtype= torch.long)

        for j in range(y_i.shape[0]):
            mean_pts[idx_i[j]] += (1.0/w_i[j])*y_i[j].to(mean_pts)
            counts[idx_i[j], :] += (1.0/w_i[j])
        

    mean_pts = mean_pts / counts

    means = []
    for i in range(num_patches):
        idx_i = torch.tensor(patch_idx[i], dtype= torch.long)
        m_i = mean_pts[idx_i].to(mean_pts)
        means.append(m_i)

    return means


def transform_pointcloud(x, device):
    translate = -np.mean(x, axis=0)
    scale = np.array([1.0 / np.max(np.linalg.norm(x + translate, axis=1))], dtype=np.float32)
    rotate, _, _ = np.linalg.svd((x + translate).T, full_matrices=False)
    transform = (torch.from_numpy(translate).to(device),
                 torch.from_numpy(scale).to(device),
                 torch.from_numpy(rotate).to(device))
    x_tx = torch.from_numpy((scale * (x.astype(np.float32) + translate)) @ rotate).to(device)

    return x_tx, transform

def transform_mesh(x, transform):
    translate, scale, rotate = transform
    translate = translate.cpu().numpy()
    scale = scale.cpu().numpy()
    rotate = rotate.cpu().numpy()

    x_tx = (scale * (x.astype(np.float32) + translate)) @ rotate
    
    return x_tx

def transform_mesh_scale(x, scale):

    x_tx = scale * (x.astype(np.float32))
    
    return x_tx


def upsample_surface(output_folder, patch_uvs, transform, patch_models, specific_scale, vertices_in_local_region, faces_in_local_region, devices, scale=1.0, num_samples=8, normal_samples=64,
                     compute_normals=True):
    vertices = []
    vertices_without_transform = []
    normals = []
    with torch.no_grad():
        for i in range(len(patch_models)):
            if (i + 1) % 10 == 0:
                print("Upsampling %d/%d" % (i+1, len(patch_models)))

            translate_i, scale_i, rotate_i = transform[i]
            uv_i = torch.from_numpy(specific_scale*vertices_in_local_region[i]).to(patch_uvs[i])
            y_i = patch_models[i](uv_i)

            vertices_without_transform.append(y_i.squeeze().cpu().numpy())

            mesh_v = ((y_i.squeeze() @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()
            mesh_v2 = (y_i.squeeze()).cpu().numpy()

            if compute_normals:
                mesh_f = faces_in_local_region.copy()
                mesh_n = pcu.per_vertex_normals(mesh_v, mesh_f)
                normals.append(mesh_n)

            vertices.append(mesh_v)

    output_vertices = vertices.copy()
    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    if compute_normals:
        normals = np.concatenate(normals, axis=0).astype(np.float32)
    else:
        print("Fixing normals...")
        normals = pcu.estimate_normals(vertices, k=normal_samples)

    return output_vertices, normals, vertices_without_transform



def save_input_points(output_folder, x_for_each_label, transform, device, name, intrplt):

    translate_i, scale_i, rotate_i = transform

    for i, x in enumerate(x_for_each_label):
        x = torch.from_numpy(x).to(device)
        x_t = ((x @ rotate_i.transpose(0, 1)) / scale_i - translate_i).cpu().numpy()
        pcu.write_obj(output_folder + "/" + "input_part_of_{}_{}.obj".format(name, i), x_t, np.array([], dtype='int'), np.array([], dtype='float32'))


def second_mapping_main(filename, s_filename, nl, ng, d, o_filename, o_folder, intrplt, specific_scale):
    args = argparse.Namespace(mesh_filename=filename, plot=False, save_every=2000, local_epochs=nl, global_epochs=ng, learning_rate=1e-3, devices=d, exact_emd=False, max_sinkhorn_iters=32, sinkhorn_epsilon=1e-3, output=o_filename, seed=-1, use_best=True, print_every=16, upsamples_per_patch=8, normal_neighborhood_size=64, interpolate=intrplt, save_pre_cc=False)

    # We'll populate this dictionary and save it as output
    output_dict = {
        "pre_cycle_consistency_model": None,
        "final_model": None,
        "patch_uvs": None,
        "patch_idx": None,
        "patch_txs": None,
        "interpolate": args.interpolate,
        "global_epochs": args.global_epochs,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "devices": args.devices,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils_overlapping.seed_everything(args.seed),
    }

    
    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils_overlapping.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=True)
    print("Computing neighborhoods...")

    anchor = 'overlapping_32vertices_theta0.55'
    dist = 'thetas'
    label_for_input = np.loadtxt(o_folder + '/input_label_{}.txt'.format(anchor))
    label_for_input = np.array(label_for_input, dtype='int')

    # Weights for integrating local regions
    thetas = np.loadtxt(o_folder + '/{}_{}.txt'.format(dist, anchor))
    thetas = np.array(thetas, dtype='float')

    input_thetas = np.loadtxt(o_folder + '/input_{}_{}.txt'.format(dist, anchor))
    input_thetas = np.array(input_thetas, dtype='float')

    f = []
    n = []
    num_of_regions = max(label_for_input[:, 1])+1
    print("Number of local regions: {}".format(num_of_regions))
    x_for_each_label, _, _, ids_table, input_theta_weights, _, _ = utils_overlapping.select_points_corresponding_to_label(x, f, n, label_for_input, num_of_regions, 'input', input_thetas)

    # Transform each local region
    transform_for_each_region = []
    for i, x_e in enumerate(x_for_each_label):
        x_e, t_e = transform_pointcloud(x_e, args.devices[0])
        transform_for_each_region.append(t_e)
        x_for_each_label[i] = x_e.cpu().numpy()

    # Obtain the vertex ID of the reference mesh corresponding to each region using the label
    v_sphere, f_sphere, n_sphere = pcu.read_obj(s_filename, dtype=np.float32)
    
    label_for_sphere = np.loadtxt(o_folder + '/label_{}.txt'.format(anchor))
    label_for_sphere = np.array(label_for_sphere, dtype='int')
    v_sphere_for_each_region, f_sphere_for_each_region, _, _, theta_weights, _, _ = utils_overlapping.select_points_corresponding_to_label(v_sphere, f_sphere, n_sphere, label_for_sphere, num_of_regions, 'sphere', thetas)

    for i, s_e in enumerate(v_sphere_for_each_region):
        v_sphere_for_each_region[i] = transform_mesh(s_e, transform_for_each_region[i])

    patch_xs = []
    patch_uvs = []
    patch_idx = []
    patch_vt = []
    init_pis = []

    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)

    for region in range(num_of_regions):
        # Generate points for training networks
        v = utils_overlapping.generate_points_for_learning(x_for_each_label[region].shape[0], v_sphere_for_each_region[region], f_sphere_for_each_region[region])
        uv = torch.tensor(v, requires_grad=True, device=args.devices[0])
        
        patch_xs.append(torch.from_numpy(x_for_each_label[region]).to(args.devices[0]))
        patch_uvs.append(uv)
        patch_idx.append(torch.from_numpy(ids_table[region]).to(args.devices[0]))

        _, p = sinkhorn_loss(patch_xs[region].unsqueeze(0), patch_uvs[region].unsqueeze(0))
        init_pi = p.squeeze().max(0)[1]

        uv = torch.tensor(transform_mesh_scale(v, specific_scale), requires_grad=True, device=args.devices[0])
        patch_vt.append(uv)
        
        init_pis.append(init_pi)
        
    patch_uvs = patch_vt.copy()


    num_patches = len(patch_uvs)
    output_dict["patch_uvs"] = patch_uvs
    output_dict["patch_idx"] = patch_idx


    # Initialize one model per patch and convert the input data to a pytorch tensor
    print("Creating models...")
    phi = nn.ModuleList([MLP(3, 3).to(args.devices[i % len(args.devices)]) for i in range(num_patches)])

    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    uv_optimizer = torch.optim.Adam(patch_uvs, lr=args.learning_rate)
    mse_loss = nn.MSELoss()

    # Fit a function, phi_i, for each patch so that phi_i(patch_uvs[i]) = x[patch_idx[i]]. i.e. so that the function
    # phi_i "agrees" with the point cloud on each patch.
    #
    # We also store the correspondences between the uvs and points which we use later for the consistency step. The
    # correspondences are stored in a list, pi where pi[i] is a vector of integers used to permute the points in
    # a patch.
    pi = [None for _ in range(num_patches)]

    # Cache model with the lowest loss if --use-best is passed
    best_models = [None for _ in range(num_patches)]
    best_losses = [np.inf for _ in range(num_patches)]

    loss_for_each_y = []

    print("Training local patches...")
    for epoch in range(args.local_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        sum_loss = torch.tensor([0.0]).to(args.devices[0])
        epoch_start_time = time.time()
        for i in range(num_patches):
            uv_i = patch_uvs[i]
            x_i = patch_xs[i]
            y_i = phi[i](uv_i)



            with torch.no_grad():
                if args.exact_emd:
                    M_i = pairwise_distances(x_i.unsqueeze(0), y_i.unsqueeze(0)).squeeze().cpu().squeeze().numpy()
                    p_i = ot.emd(np.ones(x_i.shape[0]), np.ones(y_i.shape[0]), M_i)
                    p_i = torch.from_numpy(p_i.astype(np.float32)).to(args.devices[0])
                else:
                    _, p_i = sinkhorn_loss(x_i.unsqueeze(0), y_i.unsqueeze(0))
                pi_i = init_pis[i]
                pi[i] = pi_i
            loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))
            if args.use_best and loss_i.item() < best_losses[i]:
                best_losses[i] = loss_i
                best_models[i] = copy.deepcopy(phi[i].state_dict())


            if epoch == args.local_epochs -1 :
                l_for_each_y = (x_i[pi[i]] - y_i)**2
                l_for_each_y = torch.sum(l_for_each_y, 1)
                l_for_each_y = torch.sqrt(l_for_each_y)
                loss_for_each_y.append(l_for_each_y.cpu().detach().numpy())
           
            sum_loss += loss_i.to(args.devices[0])

        sum_loss.backward()
        epoch_end_time = time.time()

        print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
              (epoch, args.local_epochs, sum_loss.item(),
               sum_loss.item() / num_patches, epoch_end_time-epoch_start_time))
        optimizer.step()
        uv_optimizer.step()


    v, n, _ = upsample_surface(o_folder, patch_uvs, transform_for_each_region, phi, specific_scale, v_sphere_for_each_region, f_sphere_for_each_region, args.devices,
                        scale=(1.0/1),
                        num_samples=args.upsamples_per_patch,
                        normal_samples=args.normal_neighborhood_size,
                        compute_normals=False)

    print("Saving integrated mesh...")

    v_sphere, f_sphere = utils_overlapping.weighted_concatenation_using_selected_two_vertices(np.array(v), v_sphere, f_sphere, label_for_sphere, thetas)
    utils_overlapping.write_ori(args.output + "_local.ori", v_sphere, f_sphere)
    

    if args.use_best:
        for i, phi_i in enumerate(phi):
            phi_i.load_state_dict(best_models[i])

    if args.save_pre_cc:
        output_dict["pre_cycle_consistency_model"] = copy.deepcopy(phi.state_dict())


    # Do a second, global, stage of fitting where we ask all patches to agree with each other on overlapping points.
    # If the user passed --interpolate, we ask that the patches agree on the original input points, otherwise we ask
    # that they agree on the average of predictions from patches overlapping a given point.
    #args.interpolate = True
    if args.interpolate:
        print("Computing patch means...")
        with torch.no_grad():
            # Updating target point clouds using distances from boundaries of local regions
            patch_xs = patch_means_thetas(pi, patch_uvs, patch_idx, transform_for_each_region, phi, x, input_theta_weights, o_folder)

    print("Training cycle consistency...")
    for epoch in range(args.global_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        sum_loss = torch.tensor([0.0]).to(args.devices[0])
        epoch_start_time = time.time()
        for i in range(num_patches):
            uv_i = patch_uvs[i]
            x_i = patch_xs[i]
            y_i = phi[i](uv_i)
            pi_i = pi[i]
            loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

            if loss_i.item() < best_losses[i]:
                best_losses[i] = loss_i
                best_models[i] = copy.deepcopy(phi[i].state_dict())

            sum_loss += loss_i.to(args.devices[0])

        sum_loss.backward()
        epoch_end_time = time.time()

        print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
              (epoch, args.global_epochs, sum_loss.item(),
               sum_loss.item() / num_patches, epoch_end_time-epoch_start_time))
        optimizer.step()
        uv_optimizer.step()

    for i, phi_i in enumerate(phi):
        phi_i.load_state_dict(best_models[i])

    output_dict["final_model"] = phi.state_dict()

    print("Generating dense point cloud...")
    v, n, _ = upsample_surface(o_folder, patch_uvs, transform_for_each_region, phi, specific_scale, v_sphere_for_each_region, f_sphere_for_each_region, args.devices,
                            scale=(1.0/1),
                            num_samples=args.upsamples_per_patch,
                            normal_samples=args.normal_neighborhood_size,
                            compute_normals=False)

    print("Saving dense point cloud...")
    v_sphere, f_sphere = utils_overlapping.weighted_concatenation_using_selected_two_vertices(np.array(v), v_sphere, f_sphere, label_for_sphere, thetas)
    n = utils_overlapping.write_ori(args.output + "_global.ori", v_sphere, f_sphere)
    pcu.write_obj(args.output + "_global.obj", v_sphere, f_sphere, n)


    print("Saving metadata...")
    torch.save(output_dict, args.output + ".pt")

