import argparse
import copy
import time

import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu

import programs.utils as utils
from fml.nn import SinkhornLoss, pairwise_distances
import ot
from point_cloud_utils import estimate_normals


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
        self.fc5 = nn.Linear(512, out_dim, bias=False)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def transform_pointcloud(x, device):
    translate = -np.mean(x, axis=0)
    scale = np.array([1.0 / np.max(np.linalg.norm(x + translate, axis=1))], dtype=np.float32)
    rotate = np.array([[1.0, .0, .0],[.0, 1.0, .0],[.0, .0, 1.0]], dtype=np.float32)
    #print("rotate: {}".format(rotate))
    transform = (torch.from_numpy(translate).to(device),
                 torch.from_numpy(scale).to(device),
                 torch.from_numpy(rotate).to(device))
    x_tx = torch.from_numpy((scale * (x.astype(np.float32) + translate)) @ rotate).to(device)

    return x_tx, transform


def upsample_surface(specific_scale, uv, transform, phi, num_samples=8, normal_samples=64,
                     compute_normals=True):
    vertices = []
    normals = []
    n = num_samples
    translate, scale, rotate = transform

    #sphere
    v, f, _ = pcu.read_obj("inputfile/common/sphere.obj", dtype=np.float32)
        
    #sphere
    uv_s = torch.from_numpy(utils.convert_from_3d_coordinate_to_2d_parameters(specific_scale, v)).to(uv)
    print(uv_s.min(), uv_s.max())
    y = phi(uv_s)

    mesh_v = ((y.squeeze() @ rotate.transpose(0, 1)) / scale - translate).cpu().detach().numpy()
    if compute_normals:
        mesh_f = f
        mesh_n = pcu.per_vertex_normals(mesh_v, mesh_f)
        normals.append(mesh_n)

    vertices.append(mesh_v)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    if compute_normals:
        normals = np.concatenate(normals, axis=0).astype(np.float32)
    else:
        print("Fixing normals...")
        normals = pcu.estimate_normals(vertices, k=normal_samples)


    # Calculate normal vectors
    base_normals = vertices + translate.cpu().numpy()
    for i in range(base_normals.shape[0]):
        length = np.sqrt(np.sum(base_normals[i, : ]**2))
        base_normals[i, :] /= length


    return vertices, normals, f, base_normals




def spherical_mapping_main(filename, nl, d, o_filename, o_folder, scale):

    args = argparse.Namespace(mesh_filename=filename, plot=False, save_every=2000, local_epochs=nl, learning_rate=1e-3, device=d, exact_emd=False, max_sinkhorn_iters=32, sinkhorn_epsilon=1e-3, output=o_filename, seed=-1, use_best=True, print_every=16, upsamples_per_patch=8, normal_neighborhood_size=64)

    # We'll populate this dictionary and save it as output
    output_dict = {
        "final_model": None,
        "uv": None,
        "x": None,
        "transform": None,
        "exact_emd": args.exact_emd,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils.seed_everything(args.seed),
    }

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=True)

    x_original = x.copy()

    # Center the point cloud about its mean and align about its principle components
    x, transform = transform_pointcloud(x, args.device)

    # Generate an initial set of UV samples in the sphere
    uv = torch.tensor(utils.fibonacci_sphere(scale, x.shape[0]), requires_grad=True, device=args.device)

    phi = MLP(3, 3).to(args.device)

    output_dict["uv"] = uv
    output_dict["x"] = x
    output_dict["transform"] = transform


    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    uv_optimizer = torch.optim.Adam([uv], lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)
 
    # Cache correspondences to plot them later
    pi = None

    # Cache model with the lowest loss if --use-best is passed
    best_model = None
    best_loss = np.inf

    epoch = 0
    while epoch < args.local_epochs:
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        epoch_start_time = time.time()
        y = phi(uv)


        # Find correspondence among vertices
        with torch.no_grad():
            if args.exact_emd:
                M = pairwise_distances(x.unsqueeze(0), y.unsqueeze(0)).squeeze().cpu().squeeze().numpy()
                p = ot.emd(np.ones(x.shape[0]), np.ones(x.shape[0]), M)
                p = torch.from_numpy(p.astype(np.float32)).to(args.device)
            else:
                if epoch > 1:
                    _, p = sinkhorn_loss(x.unsqueeze(0), y.unsqueeze(0))
                else:
                    _, p = sinkhorn_loss(x.unsqueeze(0), uv.unsqueeze(0))

            pi = p.squeeze().max(0)[1]


        tmp = ((x[pi].unsqueeze(0)- y.unsqueeze(0))**2).squeeze(0)
        
        loss = torch.mean(tmp)

        loss.backward()


        if args.use_best and loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(phi.state_dict())

        epoch_end_time = time.time()

        if epoch % args.print_every == 0:
            print("%d/%d: [Loss = %0.5f] [Time = %0.3f]" %
                  (epoch, args.local_epochs, loss.item(), epoch_end_time-epoch_start_time))

        if epoch % args.save_every == 0:
            print("Generating dense point cloud...")
            v, n, f, n_original = upsample_surface(scale, uv, transform, phi,
                            num_samples=args.upsamples_per_patch,
                            normal_samples=args.normal_neighborhood_size,
                            compute_normals=True)
            print("Saving dense point cloud...")
            pcu.write_obj("./output/"+ str(epoch) + ".obj", v, f, n)


        optimizer.step()
        uv_optimizer.step()
        epoch += 1

    if args.use_best:
        phi.load_state_dict(best_model)

    output_dict["final_model"] = copy.deepcopy(phi.state_dict())
    print(args.output)
    torch.save(output_dict, args.output)

    print("Generating dense point cloud...")
    v, n, f, n_original = upsample_surface(scale, uv, transform, phi,
                            num_samples=args.upsamples_per_patch,
                            normal_samples=args.normal_neighborhood_size,
                            compute_normals=True)

    print("Saving dense point cloud...")
    utils.write_ori(args.output + ".ori", v, f)
    pcu.write_obj(args.output + ".obj", v, f, n)
    f_original = np.empty(0)
    utils.write_ori("{}/input_points.ori".format(o_folder), x_original, f_original)
    utils.write_ori("{}/transformed_input_points.ori".format(o_folder), x.cpu().numpy(), f_original)



if __name__ == "__main__":
    spherical_mapping_main()
