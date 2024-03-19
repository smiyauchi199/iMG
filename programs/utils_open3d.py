
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import point_cloud_utils as pcu

def judge_inner_or_outer(counter, all_lines, points, xyz_load, margin, counter2=0):
        p_id1 = all_lines[counter][12*counter2][0]
        p_id2 = all_lines[counter][12*counter2][1]
        p_id3 = all_lines[counter][12*counter2+1][1]
        p_id4 = all_lines[counter][12*counter2+2][1]
        
        margin = abs(points[p_id2, 0] - points[p_id1, 0])*margin

        ids_in_box = list(*np.where((points[p_id1, 0] - margin < xyz_load[:, 0]) & (xyz_load[:, 0] < points[p_id2, 0] + margin)
                        & (points[p_id1, 1] - margin < xyz_load[:, 1]) & (xyz_load[:, 1] < points[p_id3, 1] + margin)
                        & (points[p_id1, 2] - margin < xyz_load[:, 2]) & (xyz_load[:, 2] < points[p_id4, 2] + margin)))
        
        return ids_in_box


def neighbor_search(b_id_counter, i, j, k, l, m, n):

        # (k-，k+，j-，j+，i-，i+)
        # Put -1 in non-adjacent directions
        # If i==0
        if (j==0) and (i==0):
                if k==0:
                        return b_id_counter, -1, b_id_counter+1, -1, b_id_counter+(l-1), -1, b_id_counter+(m-1)*(l-1)
                elif k==l-2:
                        return b_id_counter, b_id_counter-1, -1, -1, b_id_counter+(l-1), -1, b_id_counter+(m-1)*(l-1)
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, -1, b_id_counter+(l-1), -1, b_id_counter+(m-1)*(l-1)
        elif (j==m-2) and (i==0):
                if k==0:
                        return b_id_counter, -1, b_id_counter+1, b_id_counter-(l-1), -1, -1, b_id_counter+(m-1)*(l-1)
                elif k==l-2:
                        return b_id_counter, b_id_counter-1, -1, b_id_counter-(l-1), -1, -1, b_id_counter+(m-1)*(l-1)
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, b_id_counter-(l-1), -1, -1, b_id_counter+(m-1)*(l-1)
        elif (i==0):
                if k==0:
                        return b_id_counter, -1, b_id_counter+1, b_id_counter-(l-1), b_id_counter+(l-1), -1, b_id_counter+(m-1)*(l-1)
                elif k==l-2:
                        return b_id_counter, b_id_counter-1, -1, b_id_counter-(l-1), b_id_counter+(l-1), -1, b_id_counter+(m-1)*(l-1)
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, b_id_counter-(l-1), b_id_counter+(l-1), -1, b_id_counter+(m-1)*(l-1)

        # If i==n-2
        elif (j==0) and (i==n-2):
                if k==0:
                        return b_id_counter, -1, b_id_counter+1, -1, b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), -1
                elif k==l-2:
                        return b_id_counter, b_id_counter-1, -1, -1, b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), -1
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, -1, b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), -1
        elif (j==m-2) and (i==n-2):
                if k==0:
                        return b_id_counter, -1, b_id_counter+1, b_id_counter-(l-1), -1, b_id_counter-(m-1)*(l-1), -1
                elif k==l-2:
                        return b_id_counter, b_id_counter-1, -1, b_id_counter-(l-1), -1, b_id_counter-(m-1)*(l-1), -1
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, b_id_counter-(l-1), -1, b_id_counter-(m-1)*(l-1), -1
        elif (i==n-2):
                if k==0:
                        return b_id_counter, -1, b_id_counter+1, b_id_counter-(l-1), b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), -1
                elif k==l-2:
                        return b_id_counter, b_id_counter-1, -1, b_id_counter-(l-1), b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), -1
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, b_id_counter-(l-1), b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), -1

        # Side
        else:
                if k==0:
                        if j==0:
                                return b_id_counter, -1, b_id_counter+1, -1, b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)
                        elif j==m-2:
                                return b_id_counter, -1, b_id_counter+1, b_id_counter-(l-1), -1, b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)
                        else:
                                return b_id_counter, -1, b_id_counter+1, b_id_counter-(l-1), b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)

                elif k==l-2:
                        if j==0:
                                return b_id_counter, b_id_counter-1, -1, -1, b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)
                        elif j==m-2:
                                return b_id_counter, b_id_counter-1, -1, b_id_counter-(l-1), -1, b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)
                        else:
                                return b_id_counter, b_id_counter-1, -1, b_id_counter-(l-1), b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)

                elif j==0:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, -1, b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)
                elif j==m-2:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, b_id_counter-(l-1), -1, b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)
                else:
                        return b_id_counter, b_id_counter-1, b_id_counter+1, b_id_counter-(l-1), b_id_counter+(l-1), b_id_counter-(m-1)*(l-1), b_id_counter+(m-1)*(l-1)


def make_boxes(x_min, x_max, y_min, y_max, z_min, z_max, l, m, n, start_id_of_tree=0):
        x = np.linspace(x_min, x_max, l)
        y = np.linspace(y_min, y_max, m)
        z = np.linspace(z_min, z_max, n)

        X, Y, Z = np.meshgrid(x, y, z)


        points = []
        for i in range(n):
                for j in range(m):
                        for k in range(l):
                                points.append([X[j, k, i], Y[j, k, i], Z[j, k, i]])


        all_lines = []
        # Stores adjacency information between boxes
        box_tree =[]
        b_id_counter = start_id_of_tree
        for i in range(n-1):
                for j in range(m-1):
                        for k in range(l-1):
                                counter = (m*l)*i+l*j+k
                                lines = [
                                        [counter, counter+1],
                                        [counter, counter+l],
                                        [counter, counter+m*l],
                                        [counter+1, counter+1+l],
                                        [counter+1, counter+1+m*l],
                                        [counter+l, counter+l+1],
                                        [counter+l, counter+l+m*l],
                                        [counter+1+l, counter+1+l+m*l],
                                        [counter+l*m, counter+l*m+1],
                                        [counter+l*m, counter+l*m+l],
                                        [counter+1+m*l, counter+1+(m+1)*l],
                                        [counter+(m+1)*l, counter+(m+1)*l+1],
                                ]
                                all_lines.append(lines)

                                neighbors = neighbor_search(b_id_counter, i, j, k, l, m, n)
                                box_tree.append(np.array(neighbors, dtype='int'))
                                
                                b_id_counter += 1

        points = np.array(points, dtype='float')

        return points, all_lines, box_tree


def create_bounding_boxes(file_of_point_clouds, visualization_flag=False):

        xyz_load, _, _ = pcu.read_obj(file_of_point_clouds, dtype=np.float32)

        # Size of offset for overlapping between boxes
        margin = 0.01

        x_max = xyz_load[:, 0].max()+margin
        x_min = xyz_load[:, 0].min()-margin
        y_max = xyz_load[:, 1].max()+margin
        y_min = xyz_load[:, 1].min()-margin
        z_max = xyz_load[:, 2].max()+margin
        z_min = xyz_load[:, 2].min()-margin

        # Number of array elements
        l = 10

        if x_max - x_min > y_max - y_min and x_max - x_min > z_max - z_min:
                box_size = (x_max - x_min)/l
                m = int(round((y_max - y_min)/box_size))
                n = int(round((z_max - z_min)/box_size))
        elif y_max - y_min > x_max - x_min and y_max - y_min > z_max - z_min:
                box_size = (y_max - y_min)/l
                m = l
                l = int(round((x_max - x_min)/box_size))
                n = int(round((z_max - z_min)/box_size))
        
        elif z_max - z_min > x_max - x_min and z_max - z_min > y_max - y_min:
                box_size = (z_max - z_min)/l
                n = l
                l = int(round((x_max - x_min)/box_size))
                m = int(round((y_max - y_min)/box_size))
                
        else:
                m = l
                n = l

        if l <= 2:
                l = 3
        if m <= 2:
                m = 3
        if n <= 2:
                n = 3

        points, all_lines, box_tree = make_boxes(x_min, x_max, y_min, y_max, z_min, z_max, l, m, n)

        del_candidate = []
        num_of_boxes = len(all_lines)



        for i in range(num_of_boxes):
                ids_in_box = judge_inner_or_outer(i, all_lines, points, xyz_load, 0)

                if ids_in_box == []:
                        del_candidate.append(i)


        for i, d in enumerate(del_candidate):
                del_id = d - i
                all_lines.pop(del_id)
                # Update box_tree
                # Delete del_id
                box_tree.pop(del_id)
                # Delete if del_id is included, subtract 1 if it contains a value greater than d
                for j, bt in enumerate(box_tree):
                        bt = np.where(bt == del_id, -1, bt)
                        box_tree[j] = np.where(bt > del_id, bt-1, bt)

        # Get the coordinates in the box
        num_of_boxes = len(all_lines)
        points_in_boxes = []
        points_ids_in_boxes = []
        for i in range(num_of_boxes):
                ids_in_box = judge_inner_or_outer(i, all_lines, points, xyz_load, margin)
                points_in_boxes.append(xyz_load[ids_in_box])
                points_ids_in_boxes.append(np.array(ids_in_box))

        if(visualization_flag):
                p_colors = [[0, 0, 0] for i in range(len(xyz_load))]

                all_lines_a = np.array(all_lines, dtype='int')
                all_lines_a = np.reshape(all_lines_a, (all_lines_a.shape[0]*all_lines_a.shape[1], 2))
                colors = [[1, 0, 0] for i in range(len(all_lines_a))]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(all_lines_a),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                pcd = o3d.geometry.PointCloud()
                pcd.vertex_colors = o3d.utility.Vector3dVector(p_colors)
                o3d.visualization.draw_geometries([pcd, line_set])

        return all_lines, points, points_in_boxes, points_ids_in_boxes, box_tree, margin, xyz_load



def create_bounding_boxes_for_vertices(point_clouds, num_of_division, visualization_flag=False):

        xyz_load = point_clouds.copy()

        x_max = xyz_load[:, 0].max()
        x_min = xyz_load[:, 0].min()
        y_max = xyz_load[:, 1].max()
        y_min = xyz_load[:, 1].min()
        z_max = xyz_load[:, 2].max()
        z_min = xyz_load[:, 2].min()

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)
        z_len = abs(z_max - z_min)

        if x_len > y_len and x_len > z_len:
                l = num_of_division
                m = 2
                n = 2

        elif y_len > x_len and y_len > z_len:
                l = 2
                m = num_of_division
                n = 2

        else:
                l = 2
                m = 2
                n = num_of_division


        points, all_lines, box_tree = make_boxes(x_min, x_max, y_min, y_max, z_min, z_max, l, m, n)


        del_candidate = []
        num_of_boxes = len(all_lines)



        for i in range(num_of_boxes):
                ids_in_box = judge_inner_or_outer(i, all_lines, points, xyz_load, 0)

                if ids_in_box == []:
                        del_candidate.append(i)


        for i, d in enumerate(del_candidate):
                del_id = d - i
                all_lines.pop(del_id)
                # Update box_tree
                # Delete del_id
                box_tree.pop(del_id)
                # Delete if del_id is included, subtract 1 if it contains a value greater than d
                for j, bt in enumerate(box_tree):
                        bt = np.where(bt == del_id, -1, bt)
                        box_tree[j] = np.where(bt > del_id, bt-1, bt)



        # Get the coordinates in the box
        num_of_boxes = len(all_lines)
        points_in_boxes = []
        points_ids_in_boxes = []
        check_ids = np.zeros(xyz_load.shape[0])
        for i in range(num_of_boxes):
                ids_in_box = judge_inner_or_outer(i, all_lines, points, xyz_load, 0.0)
                points_in_boxes.append(xyz_load[ids_in_box])
                points_ids_in_boxes.append(np.array(ids_in_box))
                check_ids[ids_in_box] = 1
 
        # Checking whether all vertices belong to either
        zero_ids = np.where(check_ids == 0)[0]
        search_range = 20
        if search_range > xyz_load.shape[0]:
                search_range = xyz_load.shape[0]
                print(search_range)

        if len(zero_ids) != 0:
                nn = NearestNeighbors(metric='euclidean')
                nn.fit(xyz_load)
                for zi in zero_ids:
                        _, results = nn.kneighbors([xyz_load[zi]], search_range, 10)
                        for r in results[0]:
                                if check_ids[r] == 1:
                                        for j, points_ids in  enumerate(points_ids_in_boxes):
                                                if r in points_ids:
                                                        points_ids_in_boxes[j] = np.insert(points_ids_in_boxes[j], 0, zi)
                                                        break
                                        break
                        
        check_ids = np.zeros(xyz_load.shape[0])
        for i in range(num_of_boxes):
                check_ids[points_ids_in_boxes[i]] = 1
        
        if len(np.where(check_ids == 0)[0]) != 0:
               print("ERROR: please change the size of earch range.")

 
        if(visualization_flag):
                p_colors = [[0, 0, 0] for i in range(len(xyz_load))]


                all_lines_a = np.array(all_lines, dtype='int')
                all_lines_a = np.reshape(all_lines_a, (all_lines_a.shape[0]*all_lines_a.shape[1], 2))
                colors = [[1, 0, 0] for i in range(len(all_lines_a))]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(all_lines_a),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_load)

                pcd.colors = o3d.utility.Vector3dVector(p_colors)
                o3d.visualization.draw_geometries([pcd, line_set])

        return points_ids_in_boxes


