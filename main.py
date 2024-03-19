# The reference implementaiton for the TVCG 2024 paper "Isomorphic mesh generation from point clouds with multilayer perceptrons"
# This code is implimented based on the reference implementaiton for the CVPR 2019 paper "Deep Geometric Prior for Surface Reconstruction" (https://github.com/fwilliams/deep-geometric-prior).
# 2024.03.15 Shoko Miyauchi
import os, sys
from programs.global_mapping import spherical_mapping_main
from programs.temporary_local_mesh_generation import spherical_division_main
from programs.make_label_for_input_points_overlapping import input_division_main
from programs.coarse_local_mapping import second_mapping_main
from programs.curve_fitting import curve_fitting_main
from programs.local_mesh_and_point_cloud_generation import correct_label_main
from programs.fine_local_mapping import third_mapping_main


args = sys.argv
i_folder = args[1]
i_filename = args[2]
o_folder = args[3]
noise_flag = bool(int(args[12]))
device = args[13]

#parameters for spherical_mapping
spherical_scale = float(args[4])
num_of_epochs = int(args[5])
o_name = "global_mapping_result{}".format(spherical_scale)

#parameters for second_mapping
second_scale=float(args[6])
num_of_local_epochs = int(args[7])
num_of_global_epoches = int(args[7])
o2_name = "coarse_local_mapping_result_e{}_s{}".format(num_of_global_epoches,second_scale)

#parameters for third_mapping
threshold_of_fitting_error = float(args[8])
third_scale=float(args[9])
num_of_local_epochs3 = int(args[10])
num_of_global_epoches3 = int(args[10])
weight = float(args[11])
o3_name = "fine_local_mapping_result_e{}_s{}".format(num_of_global_epoches3,third_scale)

if not os.path.isdir(o_folder):
        os.mkdir(o_folder)


# Global mapping
spherical_mapping_main("inputfile/{}/{}".format(i_folder, i_filename), num_of_epochs, device, "{}/{}".format(o_folder, o_name), o_folder, spherical_scale)

# Coarse local mapping
# Reference mesh is divided into 32 local meshes
spherical_division_main(o_folder)
# Input points cloud is divided into 32 local point clouds
input_division_main("{}/{}.ori".format(o_folder, o_name), o_folder)
# Deformation of each local mesh
second_mapping_main("inputfile/{}/{}".format(i_folder, i_filename), "{}/{}.obj".format(o_folder, o_name), num_of_local_epochs, num_of_global_epoches, [device], "{}/{}".format(o_folder, o2_name), o_folder, noise_flag, second_scale)

# Fine local mapping
# Division of the second reference mesh and input point cloud (Generation of a temporary local mesh)
curve_fitting_main("inputfile/{}/{}".format(i_folder, i_filename), "{}/{}_global.obj".format(o_folder, o2_name), o_folder, threshold_of_fitting_error)
# Division of the second reference mesh and input point cloud (Generation of a local mesh and local point cloud)
correct_label_main("inputfile/{}/{}".format(i_folder, i_filename), "{}/{}_global.obj".format(o_folder, o2_name), o_folder)
# Deformation of each local mesh
third_mapping_main("inputfile/{}/{}".format(i_folder, i_filename), "{}/{}_global.obj".format(o_folder, o2_name), num_of_local_epochs3, num_of_global_epoches3, [device], "{}/{}".format(o_folder, o3_name), o_folder, noise_flag, weight, third_scale)
