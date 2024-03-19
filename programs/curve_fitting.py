#                                                                                                                                     
# sample of polynominal surface fitting                                                                                               
#                                                                                                                                     
import numpy as np
import point_cloud_utils as pcu
import open3d as o3d
from programs.utils_open3d import judge_inner_or_outer, make_boxes, create_bounding_boxes
from sklearn.neighbors import NearestNeighbors

# Specification of the minimum number of vertices in one local region
THRESHOLD_OF_VERTEX_NUM = 100


def func55(param, x, y): 
    p0, px, py, pxx, pyy, pxy, pxxy, pxyy, pxxx, pyyy, pxxxy, pxxyy, pxyyy, pxxxx, pyyyy, pxxxxy, pxxxyy, pxxyyy, pxyyyy, pxxxxx, pyyyyy = param
    return  p0 + px*x +py*y +pxx*x*x + pyy*y*y + pxy*x*y + pxxy*x*x*y + pxyy*x*y*y + pxxx*x*x*x + pyyy*y*y*y + pxxxy*x*x*x*y + pxxyy*x*x*y*y + pxyyy*x*y*y*y + pxxxx*x*x*x*x + pyyyy*y*y*y*y + pxxxxy*x*x*x*x*y + pxxxyy*x*x*x*y*y + pxxyyy*x*x*y*y*y + pxyyyy*x*y*y*y*y + pxxxxx*x*x*x*x*x + pyyyyy*y*y*y*y*y


def update_box_tree(id_of_box, box_tree, box_tree_in_sub, num_of_division):
    
    nod = num_of_division - 1
    # Direction: i-
    n_id = box_tree[id_of_box][5]
    if n_id != -1:
        for i in range((nod)**2):
            box_tree_in_sub[i][5] = n_id

    # Direction: i+
    n_id = box_tree[id_of_box][6]
    if n_id != -1:
        for i in range((nod)**3-(nod)**2, (nod)**3):
            box_tree_in_sub[i][6] = n_id

    # Direction: j-
    n_id = box_tree[id_of_box][3]
    if n_id != -1:
        for i in range(nod):
            target = i
            for j in range(nod):
                box_tree_in_sub[target][3] = n_id
                target += nod**2

    # Direction: j+
    n_id = box_tree[id_of_box][4]
    if n_id != -1:
        for i in range(nod**2-nod, nod**2):
            target = i
            for j in range(nod):
                box_tree_in_sub[target][4] = n_id
                target += nod**2

    # Direction: k-
    n_id = box_tree[id_of_box][1]
    if n_id != -1:
        for i in range(nod):
            target = i*nod
            for j in range(nod):
                box_tree_in_sub[target][1] = n_id
                target += nod**2

    # Direction: k+
    n_id = box_tree[id_of_box][2]
    if n_id != -1:
        for i in range(nod):
            target = i*nod+nod-1
            for j in range(nod):
                box_tree_in_sub[target][2] = n_id
                target += nod**2


    return box_tree_in_sub

def subdivision(id_of_box, all_lines, points_of_boxes, points_in_boxes, points_ids_in_boxes, box_tree, num_of_division=3, visualization_flag=0):
    p_id1 = all_lines[id_of_box][0][0]
    p_id2 = all_lines[id_of_box][0][1]
    p_id3 = all_lines[id_of_box][1][1]
    p_id4 = all_lines[id_of_box][2][1]

    x_min = points_of_boxes[p_id1, 0]
    x_max = points_of_boxes[p_id2, 0]
    y_min = points_of_boxes[p_id1, 1]
    y_max = points_of_boxes[p_id3, 1]
    z_min = points_of_boxes[p_id1, 2]
    z_max = points_of_boxes[p_id4, 2]

    # Set start_id to a large value to differentiate it from the original block id
    start_id = len(box_tree)
    points_in_sub, all_lines_in_sub, box_tree_in_sub = make_boxes(x_min, x_max, y_min, y_max, z_min, z_max, num_of_division, num_of_division, num_of_division, start_id)

    box_tree_in_sub = update_box_tree(id_of_box, box_tree, box_tree_in_sub, num_of_division)

    ids_in_boxes = []
    del_candidate = []
    num_of_boxes = len(all_lines_in_sub)
    
    #Size of offset
    margin = 0.01

    for i in range(num_of_boxes):
            ids_in_box = judge_inner_or_outer(i, all_lines_in_sub, points_in_sub, points_in_boxes[id_of_box], margin)
            if ids_in_box == []:
                    del_candidate.append(i)
            else:
                    ids_in_boxes.append(ids_in_box)
    
    for i, d in enumerate(del_candidate):
            del_id = d - i
            del all_lines_in_sub[del_id]

            # Update the box_tree
            # Delete del_id
            box_tree_in_sub.pop((del_id))
            # elete if del_id+start_id is included, or -1 if a value greater than d+start_id is included.
            for j, bt in enumerate(box_tree_in_sub):
                    bt = np.where(bt == del_id+start_id, -1, bt)
                    box_tree_in_sub[j] = np.where(bt > del_id+start_id, bt-1, bt)


    #　Get the vertex information inside the box.
    points_in_sub_boxes = []
    points_ids_in_sub_boxes = []
    for ids in ids_in_boxes:
            points_in_sub_boxes.append(points_in_boxes[id_of_box][ids])
            points_ids_in_sub_boxes.append(points_ids_in_boxes[id_of_box][ids])



    if(visualization_flag):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_in_boxes[id_of_box])


        p_colors = [[0, 0, 0] for i in range(len(points_in_boxes))]
        all_lines_a = np.array(all_lines_in_sub, dtype='int')
        all_lines_a = np.reshape(all_lines_a, (all_lines_a.shape[0]*all_lines_a.shape[1], 2))
        colors = [[1, 0, 0] for i in range(len(all_lines_a))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_in_sub),
            lines=o3d.utility.Vector2iVector(all_lines_a),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        pcd.colors = o3d.utility.Vector3dVector(p_colors)
        o3d.visualization.draw_geometries([pcd, line_set])

    return all_lines_in_sub, points_in_sub, points_in_sub_boxes, points_ids_in_sub_boxes, box_tree_in_sub



def fit_surface(points_in_boxes, THRESHOLD):
    low_distance = []
    high_distance = []

    for b_id, v in enumerate(points_in_boxes):
        if isinstance(v, int):
            continue
        
        data_x = v[:, 0]
        data_y = v[:, 1]
        obj = v[:, 2]
        data_n = data_x.shape[0]

        #(5, 5)
        exp=np.array([np.ones(data_n),data_x,data_y,(lambda x: x*x)(data_x),(lambda y: y*y)(data_y),(lambda x,y: x*y)(data_x,data_y),(lambda x,y: x*x*y)(data_x,data_y),(lambda x,y: x*y*y)(data_x,data_y),(lambda x: x*x*x)(data_x),(lambda y: y*y*y)(data_y),(lambda x,y: x*x*x*y)(data_x,data_y), (lambda x,y: x*x*y*y)(data_x,data_y), (lambda x,y: x*y*y*y)(data_x,data_y),(lambda x: x*x*x*x)(data_x),(lambda y: y*y*y*y)(data_y),(lambda x,y: x*x*x*x*y)(data_x,data_y), (lambda x,y: x*x*x*y*y)(data_x,data_y), (lambda x,y: x*x*y*y*y)(data_x,data_y),(lambda x,y: x*y*y*y*y)(data_x,data_y),(lambda x: x*x*x*x*x)(data_x),(lambda y: y*y*y*y*y)(data_y)])
        popt=np.linalg.lstsq(exp.T,obj)[0]

        # Error between the estimated surface and the input point cloud.
        check = func55(popt, data_x, data_y)
        max_distance = np.max(np.abs(check- obj))


        if max_distance < THRESHOLD:
            low_distance.append(b_id)
        else:
            high_distance.append(b_id)

    return low_distance, high_distance, max_distance

def update_after_concatenation(points_in_boxes, points_ids_in_boxes, box_id, idd, original_box_id, original_box_sub_id, check_boxes, box_tree, all_lines):
    tmp_points = np.concatenate([points_in_boxes[box_id], points_in_boxes[idd]])
    tmp_points = np.unique(tmp_points, axis=0)
    points_in_boxes[box_id] = tmp_points

    tmp_points_ids = np.concatenate([points_ids_in_boxes[box_id], points_ids_in_boxes[idd]])
    tmp_points_ids = np.unique(tmp_points_ids, axis=0)
    points_ids_in_boxes[box_id] = tmp_points_ids
    
    tmp_lines = np.concatenate([all_lines[box_id], all_lines[idd]])
    all_lines[box_id] = tmp_lines

    # Update box_id's box_tree
    tmp = box_tree[idd][1:]
    tmp = tmp[(tmp>=0) & (tmp!=original_box_id) & (tmp!=box_id)]
    check_boxes.remove(idd)
    tmp_c = np.array(check_boxes, dtype='int')
    check_boxes = np.concatenate([tmp_c, tmp])
    check_boxes = np.unique(check_boxes)

    # Deleted if box_id is included
    check_boxes = check_boxes[check_boxes != box_id]
    check_boxes = np.concatenate([[box_id], check_boxes])
    if original_box_sub_id == -1:
        box_tree[original_box_id] = check_boxes
    else:    
        box_tree[original_box_id][original_box_sub_id] = check_boxes
    box_tree[box_id] = check_boxes
    
    # Put -2 in the nodes and point clouds of the vanished box_tree
    box_tree[idd] = -2
    points_in_boxes[idd] = -2
    points_ids_in_boxes[idd] = -2
    all_lines[idd] = -2

    # Change adjacency information including extinct node numbers to box_id
    for j, bt in enumerate(box_tree):
        idd_index = np.where(bt == idd)
        if len(idd_index[0]) > 0:
            box_tree[j][idd_index[0]] = box_id

    return points_in_boxes, points_ids_in_boxes, box_tree, check_boxes[1:].tolist(), all_lines
    

def concatenate_boxes(original_box_id, original_box_sub_id, neighbors, points_in_boxes, points_ids_in_boxes, box_tree, all_lines, threshold_of_fitting_error):

    box_id = neighbors[0]
    #　Temporary merging between adjacent boxes in sequence.
    check_boxes = []
    for n in neighbors[1:]:
        if n != -1:

            #Check that the n-th box is not subdivided.
            if isinstance(box_tree[n], list):
                # Extract box_id containing box_id as adjacent box
                for bt in box_tree[n]:
                    if len(np.where(bt==original_box_id)[0]) != 0:
                        check_boxes.append(bt[0])
                    if len(np.where(bt==box_id)[0]) != 0:
                        check_boxes.append(bt[0])
            
            else:
                check_boxes.append(n)

    id_and_distance = []
    for cb in check_boxes:
        if isinstance(points_in_boxes[cb], int):
            continue

        else:
            if(box_id == cb):
                #print(box_tree[box_id])
                print("error! @ concatenate_boxes: {}".format(box_id))
                #for n in neighbors[1:]:
                    #print(box_tree[n])

            tmp_points_list = []
            tmp_points = np.concatenate([points_in_boxes[box_id], points_in_boxes[cb]])
            tmp_points = np.unique(tmp_points, axis=0)
            tmp_points_list.append(tmp_points)

            # Surface fitting
            low_distance, _, distance = fit_surface(tmp_points_list, threshold_of_fitting_error)
            if len(low_distance) > 0:
                id_and_distance.append([cb, distance])

    id_and_distance.sort(key=lambda x:x[1])

    for idd in id_and_distance:
        if isinstance(points_in_boxes[idd[0]], int):
            continue
        if idd == id_and_distance[0]:
            # Final merge
            points_in_boxes, points_ids_in_boxes, box_tree, check_boxes, all_lines = update_after_concatenation(points_in_boxes, points_ids_in_boxes, box_id, idd[0], original_box_id, original_box_sub_id, check_boxes, box_tree, all_lines)


        else:
            # Temporal merge
            tmp_points_list = []
            tmp_points = np.concatenate([points_in_boxes[box_id], points_in_boxes[idd[0]]])
            tmp_points = np.unique(tmp_points, axis=0)
            tmp_points_list.append(tmp_points)

            # Surface fitting
            low_distance, high_distance, distance = fit_surface(tmp_points_list, threshold_of_fitting_error)
            if len(low_distance) > 0:
                # Final merge
                points_in_boxes, points_ids_in_boxes, box_tree, check_boxes, all_lines = update_after_concatenation(points_in_boxes, points_ids_in_boxes, box_id, idd[0], original_box_id, original_box_sub_id, check_boxes, box_tree, all_lines)

    return points_in_boxes, points_ids_in_boxes, box_tree, all_lines


def unique_neighbors(neighbors):
    box_id = neighbors[0]
    tmp = neighbors[1:]
    check_boxes = np.unique(tmp)

    # Deleted if box_id is included
    check_boxes = check_boxes[check_boxes != box_id]
    check_boxes = np.concatenate([[box_id], check_boxes])

    return check_boxes


def output_label(points_ids, label, filename1, filename2):
    counter = -1
    with open(filename1, mode='w') as f:
        with open(filename2, mode='w') as f2:
            for i, ids in enumerate(points_ids):
                if isinstance(ids, int):
                    continue
                counter += 1

                for j in range(len(ids)):
                    f.write('{} {}\n'.format(ids[j], counter))
                    
                for j in range(len(label[i])):
                    f2.write('{} {}\n'.format(label[i][j], counter))




def obtain_label(points_of_boxes, points_ids, all_lines, xyz_load, filename, margin):
    points, _, _ = pcu.read_obj(filename, dtype=np.float32)

    label = [[-1] for i in range(len(points_of_boxes))]

    # Create a table of labels based on ids of input points.
    table_based_on_id = [[-1] for i in range(xyz_load.shape[0])]
    for i, p_ids in enumerate(points_ids):
        if isinstance(p_ids, int):
            continue
        for p_id in p_ids:
            table_based_on_id[p_id].append(i)

    # Nearest neighbor search
    nn = NearestNeighbors(metric='euclidean')
    nn.fit(xyz_load)

    dists, ids = nn.kneighbors(points, 1, 10)

    
    # Add search resutls to the label table
    for o_id, n_id in enumerate(ids):
        for n in table_based_on_id[n_id[0]][1:]:
            label[n].append(o_id)

    # Nearest neighbour search for spherical fitting results from the input points side
    nn_ip = NearestNeighbors(metric='euclidean')
    nn_ip.fit(points)
    _, ids = nn_ip.kneighbors(xyz_load, 1)

    # Add search resutls to the label table
    for i in range(xyz_load.shape[0]):
        for n in table_based_on_id[i][1:]:
            label[n].append(ids[i][0])

    # Remove duplicates in labels
    for i in range(len(label)):
        if len(label[i]) == 1:
            label[i] = -2
            continue
        label[i] = label[i][1:]
        label[i] = list(set(label[i]))

    return label

def obtain_label_based_on_sphere(points_of_boxes, points_ids, all_lines, xyz_load, box_tree, filename, margin):

    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)

    # Create a table of labels based on ids of input points
    table_based_on_id = [[-1] for i in range(xyz_load.shape[0])]
    for i, p_ids in enumerate(points_ids):
        if isinstance(p_ids, int):
            continue
        for p_id in p_ids:
            table_based_on_id[p_id].append(i)

    # Nearest neighbor search
    nn = NearestNeighbors(metric='euclidean')
    nn.fit(xyz_load)
    _, ids = nn.kneighbors(points, 1, 10)

    tmp_label = [[-1] for i in range(len(all_lines))]
    for i, n_id in enumerate(ids):
        labels = table_based_on_id[n_id[0]][1:]

        for l in labels:
            tmp_label[l].append(i)


    label = []
    for i, _ in enumerate(all_lines):
        if isinstance(points_ids[i], int):
            label.append(-2)
            continue

        label.append(list(set(tmp_label[i][1:])))

    # Delete labels with fewer vertices
    for i, l in enumerate(label):
        if isinstance(l, int):
            continue

        if len(l) < THRESHOLD_OF_VERTEX_NUM:
            for bt in box_tree[i][1:]:
                if not isinstance(bt, int): 
                    if not isinstance(label[bt], int):
                        #print(label[i])
                        label[bt] = label[bt] + label[i]
                        #print(label[bt])
                        label[bt] = list(set(label[bt]))
                        label[i] = -2
                        #print(points_ids[bt])
                        #print(points_ids[i])
                        #print(np.concatenate([points_ids[bt], points_ids[i]]))
                        points_ids[bt] = np.unique(np.concatenate([points_ids[bt], points_ids[i]]))

                        points_ids[i] = -2
                        break




    return label, points_ids



def curve_fitting_main(input_filename, second_mapping_result, o_folder, threshold_of_fitting_error):

    all_lines, points_of_boxes, points_in_boxes, points_ids_in_boxes, box_tree, _, xyz_load = create_bounding_boxes(input_filename)
    
    low_distance, high_distance, _ = fit_surface(points_in_boxes, threshold_of_fitting_error)

    
    #print("num of low: ", len(low_distance))
    #print("num of high: ", len(high_distance))


    # Subdivision
    for i, id_h in enumerate(high_distance):
        n = 2
        high_distance_after_subdivision = [0]

        while len(high_distance_after_subdivision)!=0:
            n += 1
            # Divide a box int (n-1)*(n-1)*(n-1) sub-blocks
            all_lines_after_subdivision, points_after_subdivision, points_in_boxes_after_subdivision, points_ids_in_boxes_after_subdivision, box_tree_in_sub = subdivision(id_h, all_lines, points_of_boxes, points_in_boxes, points_ids_in_boxes, box_tree, n)

            low_distance_after_subdivision, high_distance_after_subdivision, _ = fit_surface(points_in_boxes_after_subdivision, threshold_of_fitting_error)

            #print("num of low: ", len(low_distance_after_subdivision))
            #print("num of high: ", len(high_distance_after_subdivision))

        # Put -2 in the divided node
        box_tree[id_h] = -2
        # Add a new node behind
        box_tree = box_tree + box_tree_in_sub
        # Add point cloud information
        points_in_boxes = points_in_boxes + points_in_boxes_after_subdivision
        points_ids_in_boxes = points_ids_in_boxes + points_ids_in_boxes_after_subdivision
        # Put -2 in the divided point cloud.
        points_in_boxes[id_h] = -2
        points_ids_in_boxes[id_h] = -2

        start_id_points = points_of_boxes.shape[0]
        for j, n in enumerate(all_lines_after_subdivision):
            for k, nn in enumerate(n):
                all_lines_after_subdivision[j][k] = [nn[0] + start_id_points, nn[1] + start_id_points]
        all_lines = all_lines + all_lines_after_subdivision
        points_of_boxes = np.concatenate([points_of_boxes, points_after_subdivision])



    # Merge
    for loop in range(2):
        for i, bt in enumerate(box_tree):
            if not isinstance(bt, int):

                if not isinstance(points_in_boxes[bt[0]], int):
                    # Merge boxes
                    bt = unique_neighbors(bt)
                    points_in_boxes, points_ids_in_boxes, box_tree, all_lines = concatenate_boxes(i, -1, bt, points_in_boxes, points_ids_in_boxes, box_tree, all_lines, threshold_of_fitting_error)

        counter = 0
        for p in points_in_boxes:
            if not isinstance(p, int):
                counter += 1
        print("loop{}=".format(loop), counter)


    # Check that all vertices are labelled
    check_points = np.zeros(xyz_load.shape[0])
    for piib in points_ids_in_boxes:
        if isinstance(piib, int):
            continue
        for p in piib:
            check_points[p] = 1

    print("Unlabelled vertex ids : ", np.where(check_points==0)[0])


    # Obtain labels for spherical fitting results
    label = obtain_label(points_of_boxes, points_ids_in_boxes, all_lines, xyz_load, second_mapping_result, 0.1)
    output_label(points_ids_in_boxes, label, o_folder + "/label_overlapping_for_input_points.txt", o_folder + "/label_overlapping.txt")
    
    visualization_flag = 0
    if visualization_flag:
        pcd_original = o3d.io.read_triangle_mesh(input_filename)
        pcd_original.paint_uniform_color([0.9, 0.9, 0.9])
        for j in range(len(points_in_boxes)):
            if isinstance(points_in_boxes[j], int):
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_boxes[j])
            all_lines_a = np.array(all_lines[j], dtype='int')
            all_lines_a = np.reshape(all_lines_a, (all_lines_a.shape[0], 2))

            colors = [[1, 0, 0] for i in range(len(all_lines_a))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_of_boxes),
                lines=o3d.utility.Vector2iVector(all_lines_a),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            pcd.paint_uniform_color([0, 0, 0.9])

            o3d.visualization.draw_geometries([pcd, line_set])

