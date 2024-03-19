from sklearn.neighbors import NearestNeighbors
import numpy as np

def calc_distance(source, target, num_of_neighbors=1):
    
    # Nearest neighbor search
    nn = NearestNeighbors(metric='euclidean')
    nn.fit(target)
    dists, result = nn.kneighbors(source, num_of_neighbors)
    
    return result, dists

def load_ori(path):
    f=open(path)
    num_of_vertices = f.readline()
    num_of_vertices = int(num_of_vertices)
    vertices = [-1] * (3*num_of_vertices)
    for i in range(num_of_vertices):
        l = f.readline()
        vertices[3*i], vertices[3*i+1], vertices[3*i+2] = l.split()

    num_of_patches = f.readline()
    num_of_patches = int(num_of_patches)
    patches = [-1]*(3*num_of_patches)
    nrmls = [-1.0]*(3*num_of_patches)
    for i in range(num_of_patches):
        l = f.readline()
        patches[3*i], patches[3*i+1], patches[3*i+2] = l.split()
        l = f.readline()
        nrmls[3*i], nrmls[3*i+1], nrmls[3*i+2] = l.split()
        l = f.readline()

    return vertices, patches, nrmls



def input_division_main(i_filename, o_folder):
    s_vertices, _, _ = load_ori(o_folder + '/input_points.ori')
    t_vertices, _, _ = load_ori(i_filename)
    label = np.loadtxt(o_folder + '/label_overlapping_32vertices_theta0.55.txt')
    label = np.array(label, dtype='int')

    thetas = np.loadtxt(o_folder + '/thetas_overlapping_32vertices_theta0.55.txt')
    thetas = np.array(thetas, dtype='float')

    label_id_all = label[:, 0]

    s_v = np.array(s_vertices, dtype='float')
    s_size = int(s_v.shape[0]/3)
    s_v = s_v.reshape([s_size, 3])
    t_v = np.array(t_vertices, dtype='float')
    t_size = int(t_v.shape[0]/3)
    t_v = t_v.reshape([t_size, 3])

    results, _ = calc_distance(s_v, t_v)

    label_ids = []
    input_thetas = []
    for i, r in enumerate(results):
        ids = np.where(label_id_all==r)
        for j in range(len(ids[0])):
            label_id = label[ids[0][j], 1]
            label_ids.append([i, label_id])
            input_thetas.append(thetas[ids[0][j]])

    with open(o_folder + "/input_label_overlapping_32vertices_theta0.55.txt", 'wt') as fd_l:
        np.savetxt(fd_l, np.array(label_ids), fmt="%d")

    with open(o_folder + "/input_thetas_overlapping_32vertices_theta0.55.txt", 'wt') as fd_d:
        np.savetxt(fd_d, np.array(input_thetas), fmt="%f")


if __name__ == "__main__":
    input_division_main()
