import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from point_cloud_utils import sample_mesh_lloyd
from programs.utils_open3d import create_bounding_boxes_for_vertices
from programs.geometric_processing import coordinate_transformation, coordinate_transformation_z, fit_func55, sampling_from_ellips_distance
import math
import collections
import pickle
import point_cloud_utils as pcu

# The minimum number of vertices in one local area is specified as a percentage of the number of vertices on the sphere
THRESHOLD_OF_VERTEX_NUM_IN_ONE_REGION = 0.003

def read_obj(filename):

    vertices, faces, nrmls = pcu.read_obj(filename, dtype=np.float32)

    return vertices, faces, nrmls

def read_label(filename, num_of_vertices):

    label = np.loadtxt(filename, dtype='int')

    # Create a table of labels based on ids of input points.
    table_based_on_id = [[-1] for i in range(num_of_vertices)]
    table_based_on_label = [[-1] for i in range(num_of_vertices)]
    for i in range(label.shape[0]):
        if table_based_on_id[label[i][0]][0] == -1:
            table_based_on_id[label[i][0]][0] = label[i][1]
        else:
            table_based_on_id[label[i][0]].append(label[i][1])

        if table_based_on_label[label[i][1]][0] == -1:
            table_based_on_label[label[i][1]][0] = label[i][0]
        else:
            table_based_on_label[label[i][1]].append(label[i][0])

    return table_based_on_id, table_based_on_label



def grouping(label_t_id, ids, rings, table_based_on_id):
    groups = []
    while len(ids) > 0:
        # Starting point for searching for vertices belonging to a contiguous region
        search_ids = [ids[0]]
        ids_in_a_group = []
        while len(search_ids) > 0:
            t_id = search_ids[0]
            neighbors = rings[t_id]

            # Check whether all labels of neighbouring vertices are the same as label_t_id
            for n in neighbors:
                n_label = np.array(table_based_on_id[n])
                if len(np.where(n_label==label_t_id)[0]) > 0:
                    if (n in ids) and (not n in search_ids):
                        search_ids.append(n)
        
            ids.remove(t_id)
            search_ids.remove(t_id)
            ids_in_a_group.append(t_id)
        
        groups.append(ids_in_a_group)
    
    return groups

def calc_distances(vertices, center):
    # Calculate the distance from the centre of gravity to each vertex on the boundary
    distances = []
    for i, v in enumerate(vertices):
        d = np.linalg.norm(v-center)
        distances.append([i, d])

    # Sort by distance
    distances.sort(key=lambda x:x[1], reverse=True)

    return distances


# Check whether there are enclaves and whether they are larger than the permitted area
def judge_enclave_and_area(faces, table_based_on_id, table_based_on_label):

    num_of_vertices = len(table_based_on_id)
    # Collect adjacency information for each vertex
    rings = [[-1] for i in range(num_of_vertices)]
    for i in range(faces.shape[0]):
        face = faces[i]

        for j in range(face.shape[0]):
            if rings[face[j]][0] == -1:
                rings[face[j]][0] = face[(j+1)%face.shape[0]]
            else:
                rings[face[j]].append(face[(j+1)%face.shape[0]])


    # Check labels of vertices adjacent to each vertex and group them
    labeling_regions_in_each_group = []
    for i, ids in enumerate(table_based_on_label):
        if ids[0] == -1:
            break

        # The label of a target vertex
        label_t_id = i

        # Grouping by contiguous areas.
        groups = grouping(label_t_id, ids, rings, table_based_on_id)

        labeling_regions_in_each_group.append(groups)
    
    labeling_regions = []
    check_ids = np.zeros(num_of_vertices)
    # If there are more than two areas comprising one group, check the number of vertices in each area
    label_counter = 0
    for i, labeling_region in enumerate(labeling_regions_in_each_group):
        if(len(labeling_region) > 1):
            for lr in labeling_region:

                #if (label_counter == 3):
                #    print(label_counter)
                # If the number of vertices is above the threshold, a new label is assigned
                if(len(lr) > num_of_vertices*THRESHOLD_OF_VERTEX_NUM_IN_ONE_REGION):
                    labeling_regions.append(lr)
                    check_ids[lr] = 1

                    if(i != label_counter):
                        # Update table_based_on_id
                        # Delete i
                        for t_lr in lr:
                            table_based_on_id[t_lr].remove(i)
                            # Add label_counter
                            table_based_on_id[t_lr].append(label_counter)

                    label_counter += 1
                
                # If less than threshold, remove the label
                else:
                    # Delete i
                    for t_lr in lr:
                        table_based_on_id[t_lr].remove(i)

        
        else:
            if(len(labeling_region[0]) > num_of_vertices*THRESHOLD_OF_VERTEX_NUM_IN_ONE_REGION):
                labeling_regions.append(labeling_region[0])

                if(i != label_counter):
                    # Update table_based_on_id
                    # Delete i
                    for t_lr in labeling_region[0]:
                        table_based_on_id[t_lr].remove(i)
                        # Add label_counter
                        table_based_on_id[t_lr].append(label_counter)

                label_counter += 1
                check_ids[labeling_region[0]] = 1
            else:
                # Delete i
                for t_lr in labeling_region[0]:
                    table_based_on_id[t_lr].remove(i)


    # Collection of vertices that do not belong to any group
    zero_ids = np.where(check_ids==0)[0]

    for zi in zero_ids:
        neighbors = rings[zi]
        labels = []
        for n in neighbors:
            n_label = np.array(table_based_on_id[n])
            labels = labels+ n_label.tolist()

        # If there are no labelled vertices in the vicinity, further search within 1-ring of the neighboura
        seen_ids = [zi] + neighbors
        while len(labels)==0:
            next_neighbors = []
            for n in neighbors:
                n_neighbors = [ri for ri in rings[n] if ri not in seen_ids and ri not in next_neighbors]
                next_neighbors = next_neighbors + n_neighbors
                for n_n in n_neighbors:
                    n_label = np.array(table_based_on_id[n_n])
                    labels = labels+ n_label.tolist()
            neighbors = next_neighbors.copy()
            seen_ids += neighbors
                    
        # Merge into the labels that are most widely distributed in the vicinity
        c = collections.Counter(labels)
        #print(c.most_common()[0][0])
        labeling_regions[c.most_common()[0][0]].append(zi)
        table_based_on_id[zi].append(c.most_common()[0][0])

    return labeling_regions, rings, table_based_on_id

def judge_centroid_position(sph_vertices, labeling_regions, rings, table_based_on_id):

    nn = NearestNeighbors(metric='euclidean')
    nn.fit(sph_vertices)

    counter = 0

    new_labeling_regions = []
    for i, labeling_region in enumerate(labeling_regions):

        center = np.mean(sph_vertices[labeling_region], axis=0)
        
        # Check whether the nearest neighbour of the centre of gravity is included as a vertex of i-th label
        _, result = nn.kneighbors([center], 1)
        if result[0][0] in labeling_region:
            new_labeling_regions.append(labeling_region)
            continue
        else:
            # re-division
            num_of_division = 3
            while True:

                points_ids_in_boxes = create_bounding_boxes_for_vertices(sph_vertices[labeling_region], num_of_division)

                while len(points_ids_in_boxes) == 1:
                    num_of_division += 1
                    points_ids_in_boxes = create_bounding_boxes_for_vertices(sph_vertices[labeling_region], num_of_division)

                # Continuity check
                # Grouping by contiguous areas.
                for k, points_ids in enumerate(points_ids_in_boxes):
                    labeling_region_a = np.array(labeling_region)
                    re_ids = labeling_region_a[points_ids]
                    groups = grouping(i, re_ids.tolist(), rings, table_based_on_id)

                    if len(groups) > 1:
                        # The group with the largest number of vertices is left untouched and all other groups are merged with another adjacent group
                        max_num = 0
                        max_id = 0
                        for l, group in enumerate(groups):
                            if len(group) > max_num:
                                max_num = len(group)
                                max_id = l

                        # Finding the nearest neighbour vertices from other groups and deciding one vertex by majority vote.
                        id_table = []
                        for l, points_ids2 in enumerate(points_ids_in_boxes):
                            if l == k:
                                continue
                            #print(labeling_region_a[points_ids2].tolist())
                            id_table = id_table + labeling_region_a[points_ids2].tolist()
                        nn2 = NearestNeighbors(metric='euclidean')
                        nn2.fit(sph_vertices[id_table])

                        for l, group in enumerate(groups):
                            if l == max_id:
                                continue

                            _, result = nn2.kneighbors(sph_vertices[group], 1)

                            # Convert id
                            re_result = []
                            for r in result:
                                re_result.append(id_table[r[0]])

                            max_num2 = 0
                            max_id2 = -1
                            for m, points_ids2 in enumerate(points_ids_in_boxes):
                                if m==k:
                                    continue
                                l1_l2_and = set(labeling_region_a[points_ids2]) & set(re_result)
                                counter = 0
                                for l1_l2 in l1_l2_and:
                                    counter += re_result.count(l1_l2)
                                if max_num2 < counter:
                                    max_num2 = counter
                                    max_id2 = m
                            
                            # Retranslation of ID
                            re_group = []
                            for g in group:
                                r_g = np.where(labeling_region_a == g)[0]
                                re_group.append(r_g[0])
                            for r_g in re_group:
                                points_ids_in_boxes[k] = np.delete(points_ids_in_boxes[k], np.where(points_ids_in_boxes[k] == r_g))
                            points_ids_in_boxes[max_id2] = np.concatenate([points_ids_in_boxes[max_id2], re_group])


                        # Continuity check for each group
                        for l, points_ids in enumerate(points_ids_in_boxes):
                            re_ids = labeling_region_a[points_ids]

                            groups = grouping(i, re_ids.tolist(), rings, table_based_on_id)
                            if len(groups) != 1:
                                print("ERROR: please check the function 'judge_centroid_position'. ")
                                #print(table_based_on_id[groups[1][0]])

                flag = 0
                for points_ids in points_ids_in_boxes:
                    center = np.mean(sph_vertices[labeling_region_a[points_ids]], axis=0)
        
                    # Check whether the nearest neighbour of the centre of gravity is included as a vertex of i-th label
                    _, result = nn.kneighbors([center], 1)
                    if not result[0][0] in labeling_region:
                        flag = 1
                        break

                #Update labeling_regions
                if flag == 0:
                    for points_ids in points_ids_in_boxes: 

                        new_labeling_regions.append(labeling_region_a[points_ids].tolist())
                        #Update table_based_on_id
                        label_counter = len(new_labeling_regions) -1
                        # Delete i
                        for t_lr in labeling_region_a[points_ids]:
                            if i in table_based_on_id[t_lr]:
                                table_based_on_id[t_lr].remove(i)
                                # Add label_counter
                                table_based_on_id[t_lr].append(label_counter)
                    break

                num_of_division += 1

                min_num = len(points_ids_in_boxes[0])
                for points_ids in points_ids_in_boxes:
                    if min_num > len(points_ids):
                        min_num = len(points_ids)
                #if min_num < sph_vertices.shape[0]*THRESHOLD_OF_VERTEX_NUM_IN_ONE_REGION:
                #    print("!!!!")
                #    print(min_num)

    return new_labeling_regions, table_based_on_id


def display(points, x):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Generate ellipses
    ellips = []
    a = 1/(math.sqrt(x[0]))
    for i in range(1000):
        e_x = i*2*a/1000 - a
        tmp = 1 - x[0]*e_x**2
        if tmp < 0:
            tmp = 0.0
        e_y = math.sqrt(tmp/x[1])
        ellips.append([e_x, e_y, 0])
        ellips.append([e_x, -1.0*e_y, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(ellips)


    p_colors = [[0, 1, 0] for i in range(len(points))]
    p_colors2 = [[1, 0, 0] for i in range(len(points))]
    all_lines_a = np.array([[0, 1], [0, 2]])
    colors = [[1, 0, 0] for i in range(len(all_lines_a))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 1], [1, 0, 0]]),
        lines=o3d.utility.Vector2iVector(all_lines_a),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    pcd.colors = o3d.utility.Vector3dVector(p_colors)

    o3d.visualization.draw_geometries([pcd, pcd2, line_set])


def calc_parameters_of_ellipsoids(sph_vertices, labeling_regions):

    sph_center = np.mean(sph_vertices, axis=0)

    # Rotation matrices for coordinate transformations
    R_rot = []
    # Parameters of the ellipse
    parameters = []



    # elliptic-column approximation
    for ids in labeling_regions:
        if ids == -1:
            continue  

        center = np.mean(sph_vertices[ids], axis=0)
        original_normal = center - sph_center
        # Coordinate transformation from the plane normal to [original_normal] to the plane normal to [0, 0, 1] (derivation of the rotation matrix)
        R = coordinate_transformation(original_normal)

        # Transform ids so that centre becomes the origin of the new coordinate system
        ids_in_new_coordinate = np.transpose(np.dot(R, np.transpose(sph_vertices[ids] - center)))
        # Projection onto the xy-plane
        ids_in_new_coordinate[:, 2] = 0.0
        distances = calc_distances(ids_in_new_coordinate, 0.0)

        # Vector with ids that is the maximum distance
        Rz = coordinate_transformation_z(ids_in_new_coordinate[distances[0][0]])

        # Coordinate transformation so that the x-axis is the major axis.
        ids_in_new_coordinate = np.transpose(np.dot(Rz, np.transpose(ids_in_new_coordinate)))

        R_rot.append(np.dot(Rz, R))

        # Margin
        margin = 1.01
        ids_abs = np.abs(ids_in_new_coordinate)
        a = margin*np.max(ids_abs[:,0], axis=0)
        b = margin*np.max(ids_abs[:,1], axis=0)

        x = [1.0/a**2, 1.0/b**2]

        # Check whether the interior and exterior of a vertex
        flag = 1
        while(flag):
            flag = 0
            for d in distances:
                t_id = d[0]    
                r = ((ids_in_new_coordinate[t_id][0])**2*x[0]) + ((ids_in_new_coordinate[t_id][1])**2*x[1])
                if  r > 1:
                    # Parameter update to pass through t_id
                    x[1] = (1.0 - ids_in_new_coordinate[t_id][0]**2*x[0])/(ids_in_new_coordinate[t_id][1]**2)/margin**2
                    flag = 1
                    break
        
        parameters.append(x)

    return R_rot, sph_center, parameters


def check_of_labeling(sph_vertices, labels_based_on_label, angles_based_on_label, rings, R_rot, parameters):
    num_of_vertices = sph_vertices.shape[0]
    labels_based_on_ids = [[-1] for i in range(num_of_vertices)]

    for i, lb in enumerate(labels_based_on_label):
        for id_in_lb in lb:
            if labels_based_on_ids[id_in_lb][0] == -1:
                labels_based_on_ids[id_in_lb][0] = i
            else:
                labels_based_on_ids[id_in_lb].append(i)

    for i, lbi in enumerate(labels_based_on_ids):
        if lbi[0] == -1:
            # Assign the same labels to vertices that do not belong anywhere as to the surrounding vertices
            labels_for_rings = []
            for r in rings[i]:
                if labels_based_on_ids[r][0] == -1:
                    continue
                else:
                    labels_for_rings = labels_for_rings + labels_based_on_ids[r]
            # If there are no labelled vertices in the vicinity, search further in the surrounding 1-ring
            neighbors = rings[i]
            while len(labels_for_rings)==0:
                next_neighbors = []
                for n in neighbors:
                    if len(labels_for_rings) > 5:
                        break
                    n_neighbors = rings[n]
                    next_neighbors = next_neighbors + n_neighbors
                    for n_n in n_neighbors:
                        if labels_based_on_ids[n_n][0] == -1:
                            continue
                        else:
                            labels_for_rings = labels_for_rings + labels_based_on_ids[n_n]
                neighbors = next_neighbors.copy()
                if len(labels_for_rings) > 0:
                    break

            # Addition of vertices to the most frequent label id
            mode_label = collections.Counter(labels_for_rings).most_common()[0][0]

            # Calc angle
            R = R_rot[mode_label]
            p = parameters[mode_label]


            # Coordinate transformation
            vertices_in_new_coordinate = np.transpose(np.dot(R, np.transpose(sph_vertices[i])))
            # Projection onto the xy-plane
            vertices_in_new_coordinate[2] = 0.0  

            t = vertices_in_new_coordinate
            a = 1.0/math.sqrt(p[0])
            b = 1.0/math.sqrt(p[1])

            # Check whether the interior and exterior of a vertex
            r = (t[0]**2*p[0]) + (t[1]**2*p[1])

            # Calculate the ratio of the distances from the centre of the ellipse to the target vertex (ratio such that the centre is 0 and the circumference is 1)
            ratio = math.sqrt(p[0]*p[1]*((t[0]**2*b**2) + (t[1]**2*a**2)))

            if ratio < 0.0:
                ratio = 0
            elif ratio > 1.0:
                ratio = 1.0

            labels_based_on_label[mode_label].append(i)
            angles_based_on_label[mode_label].append(ratio)
                

    return labels_based_on_label, angles_based_on_label


def labeling_for_sphere(sph_vertices, R_rot, parameters, rings):

    labels_based_on_label = []
    angles_based_on_label = []

    for R, p in zip(R_rot, parameters):
        ids = []
        angles = []
        # Coordinate transformation
        vertices_in_new_coordinate = np.transpose(np.dot(R, np.transpose(sph_vertices)))
        table_of_ids = np.where(vertices_in_new_coordinate[:,2]>0)[0]
        target_vertices = vertices_in_new_coordinate[table_of_ids]
        # Projection onto the xy-plane
        target_vertices[:,2] = 0.0
        
        for i, t in enumerate(target_vertices):
            a = 1.0/math.sqrt(p[0])
            b = 1.0/math.sqrt(p[1])
            if t[0] < -1.0*a or a < t[0] or t[1] < -1.0*b or b < t[1]:
                continue 
            # Check whether the interior and exterior of a vertex
            r = (t[0]**2*p[0]) + (t[1]**2*p[1])
            if  r <= 1:
                ids.append(table_of_ids[i])

                #  Calculate the ratio of the distances from the centre of the ellipse to the target vertex (ratio such that the centre is 0 and the circumference is 1)
                ratio = math.sqrt(p[0]*p[1]*((t[0]**2*b**2) + (t[1]**2*a**2)))
                angles.append(ratio)
                #if ratio < 0.0 or 1.0 < ratio:
                #    print(ratio)
                
        if len(ids) > 0:
            labels_based_on_label.append(ids)
            angles_based_on_label.append(angles)



    # Checking that all vertices are labelled
    labels_based_on_label, angles_based_on_label = check_of_labeling(sph_vertices, labels_based_on_label, angles_based_on_label, rings, R_rot, parameters)

    return labels_based_on_label, angles_based_on_label


def save_label(labels_based_on_label, angles_based_on_label, filename1, filename2):

    with open(filename1, mode='w') as f:
        with open(filename2, mode='w') as f2:

            for i, lb in enumerate(labels_based_on_label):
                for j, l in enumerate(lb):
                    f.write('{} {}\n'.format(l, i))
                    f2.write(str(angles_based_on_label[i][j]))
                    f2.write('\n')


def obtain_label_of_input_points(filename, filename_of_spherical_fitting_result, labels_based_on_label, angles_based_on_label, rings):

    points, _, _ = pcu.read_obj(filename, dtype=np.float32)

    vertices, _, _ = pcu.read_obj(filename_of_spherical_fitting_result, dtype=np.float32)

    input_labels_based_on_label = [-1 for i in range(len(labels_based_on_label))]
    input_angles_based_on_label = [-1 for i in range(len(labels_based_on_label))]

    # Creation of a table of labels based on ids of input points
    table_based_on_id = [[-1] for i in range(vertices.shape[0])]
    angle_table_based_on_id = [[-1] for i in range(vertices.shape[0])]
    for i, p_ids in enumerate(labels_based_on_label):
        for j, p_id in enumerate(p_ids):
            table_based_on_id[p_id].append(i)
            angle_table_based_on_id[p_id].append(angles_based_on_label[i][j])

    nn = NearestNeighbors(metric='euclidean')
    nn.fit(vertices)
    _, ids = nn.kneighbors(points, 1, 10)

    for i, n_id in enumerate(ids):
        labels = table_based_on_id[n_id[0]][1:]
        angles = angle_table_based_on_id[n_id[0]][1:]

        for a, l in zip(angles, labels):
            if input_labels_based_on_label[l] == -1:
                input_labels_based_on_label[l] = [i]
                input_angles_based_on_label[l] = [a]

            else:
                input_labels_based_on_label[l].append(i)
                input_angles_based_on_label[l].append(a)

    nn2 = NearestNeighbors(metric='euclidean')
    nn2.fit(points)
    _, ids2 = nn2.kneighbors(vertices, 1)

    for i, n_id in enumerate(ids2):
        labels = table_based_on_id[i][1:]
        angles = angle_table_based_on_id[i][1:]

        for a, l in zip(angles, labels):
            if input_labels_based_on_label[l] == -1:
                input_labels_based_on_label[l] = [n_id[0]]
                input_angles_based_on_label[l] = [a]

            else:
                input_labels_based_on_label[l].append(n_id[0])
                input_angles_based_on_label[l].append(a)

    # Delete duplicates
    for i, il in enumerate(input_labels_based_on_label):
        if il == -1:
            continue

        il_np = np.array(il)
        u_il_np, indices = np.unique(il_np, return_index=True)

        ia_np = np.array(input_angles_based_on_label[i])
        u_ia_np = ia_np[indices]
           
        input_labels_based_on_label[i] = u_il_np.tolist()
        input_angles_based_on_label[i] = u_ia_np.tolist()

    # Delete labels with a low number of elements.
    del_labels = []
    for i, ilbol in enumerate(input_labels_based_on_label):

        # If the number of elements is less than 3, merge into adjacent labels
        if len(ilbol) < 3:
            del_labels.append(i)
            ids = labels_based_on_label[i]
            labels = []
            for t_id in ids:
                for r_id in rings[t_id]:
                    # Search and collect label numbers containing r_id
                    add_labels = [0]
                    for j, lbol in enumerate(labels_based_on_label):
                        if r_id in lbol:
                            add_labels.append(j)

                    labels = labels + add_labels

            # Find the most peripheral labels
            c = collections.Counter(labels)

            # Integrated into surrounding labels
            input_labels_based_on_label[c.most_common()[0][0]] = input_labels_based_on_label[c.most_common()[0][0]] + ilbol
            input_angles_based_on_label[c.most_common()[0][0]] = input_angles_based_on_label[c.most_common()[0][0]] + input_angles_based_on_label[i]

            labels_based_on_label[c.most_common()[0][0]] = labels_based_on_label[c.most_common()[0][0]] + labels_based_on_label[i]
            angles_based_on_label[c.most_common()[0][0]] = angles_based_on_label[c.most_common()[0][0]] + angles_based_on_label[i]
    
    # Erase removed label information
    counter = 0
    for dl in del_labels:
        dl -= counter

        del input_labels_based_on_label[dl]
        del input_angles_based_on_label[dl]
        del labels_based_on_label[dl]
        del angles_based_on_label[dl]

        counter += 1


    return input_labels_based_on_label, input_angles_based_on_label, labels_based_on_label, angles_based_on_label, points


def display(points, x):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Elliptical generation
    ellips = []
    a = 1/(math.sqrt(x[0]))
    for i in range(1000):
        e_x = i*2*a/1000 - a
        tmp = 1 - x[0]*e_x**2
        if tmp < 0:
            tmp = 0.0
        e_y = math.sqrt(tmp/x[1])
        ellips.append([e_x, e_y, 0])
        ellips.append([e_x, -1.0*e_y, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(ellips)


    p_colors = [[0, 1, 0] for i in range(len(points))]
    p_colors2 = [[1, 0, 0] for i in range(len(points))]
    all_lines_a = np.array([[0, 1], [0, 2]])
    colors = [[1, 0, 0] for i in range(len(all_lines_a))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 1], [1, 0, 0]]),
        lines=o3d.utility.Vector2iVector(all_lines_a),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    pcd.colors = o3d.utility.Vector3dVector(p_colors)

    o3d.visualization.draw_geometries([pcd, pcd2, line_set])


def surface_fitting_and_point_sampling(R_rot, center, parameters, input_points, input_labels_based_on_label, labels_based_on_label):

    all_input_points_sampled_from_surface = []
    # Centripetal shift
    input_points -= center
    for i, point_ids in enumerate(input_labels_based_on_label):
        if point_ids == -1:
            continue
        # Coordinate transformation of point clouds
        R = R_rot[i]
        vertices_in_new_coordinate = np.transpose(np.dot(R, np.transpose(input_points[point_ids])))
        # Curved surface fitting
        params = fit_func55(vertices_in_new_coordinate)
        # Sampling from within the ellipse
        sampled_points = sampling_from_ellips_distance(parameters[i], len(labels_based_on_label[i]), params, vertices_in_new_coordinate)
        if len(sampled_points) != len(labels_based_on_label[i]):
            print("The number of vertices is different!")
            #print(len(sampled_points))
            #print(len(labels_based_on_label[i]))

        # Coordinate transformation
        R_inv = np.linalg.inv(R)
        input_points_sampled_from_surface = np.transpose(np.dot(R_inv, np.transpose(sampled_points)))
        input_points_sampled_from_surface += center
        all_input_points_sampled_from_surface.append(input_points_sampled_from_surface.astype(np.float32))

    return all_input_points_sampled_from_surface


def save_list(filename, list_data):
    f = open(filename, 'wb')
    pickle.dump(list_data, f)


def correct_label_main(i_filename, second_mapping_result, o_folder):

    sph_vertices, sph_faces, sph_nrmls = read_obj("inputfile/common/sphere.obj")
    table_based_on_id, table_based_on_label = read_label(o_folder + "/label_overlapping.txt", sph_vertices.shape[0])

    labeling_regions, rings, table_based_on_id = judge_enclave_and_area(sph_faces, table_based_on_id, table_based_on_label)

    # Check whether the nearest vertex of the centre of gravity of a group is included in that group
    labeling_regions, table_based_on_id = judge_centroid_position(sph_vertices, labeling_regions, rings, table_based_on_id)

    # Calculate the coordinate transformation matrix and elliptic parameters
    R_rot, _, parameters = calc_parameters_of_ellipsoids(sph_vertices, labeling_regions)

    # Labelling for spherical surfaces
    labels_based_on_label, angles_based_on_label = labeling_for_sphere(sph_vertices, R_rot, parameters, rings)

    input_labels_based_on_label, input_angles_based_on_label, labels_based_on_label, angles_based_on_label, input_points = obtain_label_of_input_points(i_filename, second_mapping_result, labels_based_on_label, angles_based_on_label, rings)

    # Calculate the coordinate transformation matrix, centre of rotational gravity and elliptic parameters
    R_rot, center, parameters = calc_parameters_of_ellipsoids(input_points, input_labels_based_on_label)


    # Vertex sampling from fitting surfaces
    all_input_points_sampled_from_surface = surface_fitting_and_point_sampling(R_rot, center, parameters, input_points, input_labels_based_on_label, labels_based_on_label)

    save_label(labels_based_on_label, angles_based_on_label, o_folder + "/label_overlapping_ellipses_after_2nd_mapping.txt", o_folder + "/angles_overlapping_ellipses_after_2nd_mapping.txt")
    save_label(input_labels_based_on_label, input_angles_based_on_label, o_folder + "/input_label_overlapping_ellipses_after_2nd_mapping.txt", o_folder + "/input_angles_overlapping_ellipses_after_2nd_mapping.txt")
    save_list(o_folder + "/sampled_input_points_in_each_region_overlapping_ellipses_after_2nd_mapping.txt", all_input_points_sampled_from_surface)

