import numpy as np
import torch
from point_cloud_utils import sample_mesh_poisson_disk, estimate_normals, sample_mesh_lloyd, write_obj
import random

def load_srb_range_scan(file_name):
    """
    Load a range scan point cloud from the Surface Reconstruction Benchmark dataset
    :param file_name: The file containing the point cloud
    :return: A pair (v, f) of vertices and normals both with shape [n, 3]
    """
    v = []
    n = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            x, y, z, nx, ny, nz = [float(p) for p in line.split()]
            v.append((x, y, z))
            n.append((nx, ny, nz))
    return np.array(v), np.array(n)


def load_point_cloud_by_file_extension(file_name, compute_normals=False):
    import point_cloud_utils as pcu
    if file_name.endswith(".obj"):
        v, f, n = pcu.read_obj(file_name, dtype=np.float32)
    elif file_name.endswith(".off"):
        v, f, n = pcu.read_off(file_name, dtype=np.float32)
    elif file_name.endswith(".ply"):
        v, f, n, _ = pcu.read_ply(file_name, dtype=np.float32)
    elif file_name.endswith(".npts"):
        v, n = load_srb_range_scan(file_name)
        f = []
    else:
        raise ValueError("Invalid file extension must be one of .obj, .off, .ply, or .npts")

    if compute_normals and f.shape[0] > 0:
        n = pcu.per_vertex_normals(v, f)
    return v, n



def seed_everything(seed):
    """
    Seed all the RNGs that are used by the programs in this repository
    :param seed: The random seed to use. If non-positive, a seed is chosen at random
    :return: The seed used for the RNGs
    """
    if seed < 0:
        seed = np.random.randint(np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed


def faceNormal(v0, v1, v2):
    vec1 = np.array((v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]))
    vec2 = np.array((v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]))

    n = np.array((vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0]))

    len = np.linalg.norm(n, ord=2)

    return n/len

def facearea(v0, v1, v2):
    len1 = np.linalg.norm(v1-v0, ord=2)
    len2 = np.linalg.norm(v2-v1, ord=2)
    len3 = np.linalg.norm(v0-v2, ord=2)

    s = (len1 + len2 + len3) / 2.0
    area = (s*(s-len1)*(s-len2)*(s-len3)) ** 0.5

    return area


def write_ori(filename, v, f):
    n_all = []
    with open(filename, 'wt') as fo:
        fo.write(str(v.shape[0]) + '\n')
        for i in range(v.shape[0]):
            fo.write(str(v[i,0]) + ' ' + str(v[i,1]) + ' ' + str(v[i,2]) + '\n')
            
        fo.write(str(f.shape[0]) + '\n')
        for i in range(f.shape[0]):
            fo.write(str(f[i,0]) + ' ' + str(f[i,1]) + ' ' + str(f[i,2]) + '\n')
            n = faceNormal(v[f[i,0],:], v[f[i,1],:], v[f[i,2],:])
            fo.write(str(n[0]) + ' ' + str(n[1]) + ' ' + str(n[2]) + '\n')
            area = facearea(v[f[i,0],:], v[f[i,1],:], v[f[i,2],:])

            n_all.append(n)

            fo.write(str(area) + '\n')

    return np.array(n_all, dtype='float32')

        
def generate_points_for_learning(num_of_samples, v, f):

    v_poisson = sample_mesh_lloyd(v, f, num_of_samples)
    
    return np.array(v_poisson, dtype='float32')


def select_points_corresponding_to_label(v, f, n, label, num_of_regions, fname, thetas):

    vertices_for_each_region = []
    faces_for_each_region = []
    normals_for_each_region = []
    ids_tables = []
    theta_weights = []

    # Normalise thetas to 0~1
    t_min = np.min(thetas)
    t_max = np.max(thetas)
    thetas  = (thetas - t_min)/(t_max - t_min)
    thetas = np.where(thetas < 0, 0, thetas)


    for i in range(num_of_regions):
        part_of_labels = np.where(label[:, 1]==i)
        v_p = []
        n_p = []
        t_w = []
        ids_table = []
        for l_id in part_of_labels[0]:
            v_id = label[l_id, 0]
            v_p.append(v[v_id])
            ids_table.append(v_id)
            t_w.append((1.0001- thetas[l_id]))
            if len(n) != 0 :
                n_p.append(np.array(n[v_id]))

        size = int(len(f))
        f_p = []

        for j in range(size):
            tmp_p = []
            for k, p_v_id in enumerate(f[j]):
                check = np.where(label[part_of_labels[0], 0]==int(p_v_id))
                if len(check[0])==0 :
                    break
                
                else:
                    tmp_p.append(check[0])
                if k==2:
                    f_p.append(np.array([tmp_p[0], tmp_p[1], tmp_p[2]], dtype='int32'))
                    

 
        v_p = np.array(v_p, dtype='float32')
        t_w = np.array(t_w, dtype='float32')

        f_p = np.array(f_p, dtype='int')
        f_p = f_p.reshape(f_p.shape[0], 3)
        n_p = np.array(n_p, dtype='float32')

        vertices_for_each_region.append(v_p)
        faces_for_each_region.append(f_p)
        normals_for_each_region.append(n_p)
        theta_weights.append(t_w)
        ids_tables.append(np.array(ids_table, dtype='int'))

    # Check if there is a region of size 0 in faces_for_each_region
    # Loosen patch collection conditions for areas with 0
    vertices_for_each_region_temporal = vertices_for_each_region.copy()
    faces_for_each_region_temporal = faces_for_each_region.copy()
    if fname == 'sphere':
        for i, faces in enumerate(faces_for_each_region):
            if len(faces) == 0:
                v_id = []
                # Collect all vertex ids that make up the triangular patch containing the vertices in the region
                for j in range(size):
                    for k, p_v_id in enumerate(f[j]):
                        check = np.where(ids_tables[i]==int(p_v_id))
                        if len(check[0])==0 :
                            break

                        else:
                            for l in range(3):
                                v_id.append(f[j][l])
                            break
                
                # Remove duplicate v_id
                v_id = list(set(v_id))
                v_id_n = np.array(v_id, dtype='int')
                # Get a list of coordinates corresponding to v_id
                vertices_for_each_region_temporal[i] = v[v_id]
                f_p = []
                # Get patch information including v_id
                for j in range(size):
                    tmp_p = []
                    for k, p_v_id in enumerate(f[j]):
                        check = np.where(v_id_n==int(p_v_id))
                        if len(check[0])==0 :
                            break             
                        else:
                            tmp_p.append(check[0])
                        if k==2:
                            f_p.append(np.array([tmp_p[0], tmp_p[1], tmp_p[2]], dtype='int32'))
                f_p = np.array(f_p, dtype='int')
                f_p = f_p.reshape(f_p.shape[0], 3)
                faces_for_each_region_temporal[i] = f_p



    return vertices_for_each_region, faces_for_each_region, normals_for_each_region, ids_tables, theta_weights, vertices_for_each_region_temporal, faces_for_each_region_temporal



def weighted_concatenation_using_selected_two_vertices(mapping_results_v, v_in_original_sphere, f_in_original_sphere, label, thetas):
    
    num_of_labels = mapping_results_v.shape[0]
    counter = np.zeros(v_in_original_sphere.shape[0])

    # Normalise thetas values to 0~1
    t_min = np.min(thetas)
    t_max = np.max(thetas)
    thetas  = (thetas - t_min)/(t_max - t_min)
    thetas = np.where(thetas < 0, 0, thetas)

    couter_for_multi_region = [[] for i in range(v_in_original_sphere.shape[0])]
    # Count the number of overlapping areas
    for label_id in range(num_of_labels):
        part_of_labels = np.where(label[:, 1]==label_id)
        for i, l_id in enumerate(part_of_labels[0]):
            v_id = label[l_id, 0]
            couter_for_multi_region[v_id].append([thetas[l_id], label_id, i])

    for i, cmr in enumerate(couter_for_multi_region):
        cmr.sort(key=lambda x:x[0])
        v_in_original_sphere[i] = 0.0

        for c in cmr[0:3]:
            v_in_original_sphere[i] += mapping_results_v[c[1]][c[2]]*(1.0001-c[0])
            counter[i] += (1.0001-c[0])




    for i in range(counter.shape[0]):
        v_in_original_sphere[i] /= counter[i]

    return v_in_original_sphere, f_in_original_sphere

