import point_cloud_utils as pcu
import numpy as np
import math

def convert_from_3d_coordinate_to_2d_parameters(points):
    parameters = []
    R = math.sqrt(np.sum(np.power(points[0,:],2)))
    for i in range(points.shape[0]):

        y = points[i, 1]/R
        x = points[i, 0]/R
        z = points[i, 2]/R
        if y > 1.0:
            y = 1.0
        elif y < -1.0:
            y = -1.0
        v = math.asin(y)
        u = math.atan2(points[i, 2], points[i, 0])
        
        u = math.degrees(u)
        v = math.degrees(v)
        parameters.append([x, y, z])

    return np.array(parameters)


def calc_distance(point, normal):
    sum = 0
    for i in range(3):
        sum += normal[i]*point[i]

    return sum

def divide_sphere(anchor, points, normals):

    label = []
    thetas = []

    for i, point in enumerate(points):
        for j, a in enumerate(anchor):
            flag = True
            
            # Calculate angle from inner product of the anchor and points
            cos_theta = np.dot(points[a], point) / (np.linalg.norm(points[a], ord=2) * np.linalg.norm(point, ord=2))
            if cos_theta >1:
                cos_theta = 1.0

            elif cos_theta < -1:
                cos_theta = -1.0
            theta = math.acos(cos_theta)
            
            if theta < 0.55:
                label.append([i, j])
                thetas.append(theta)
    
    return label, thetas


def spherical_division_main(o_folder):
    v, f, n = pcu.read_obj("inputfile/common/sphere.obj", dtype=np.float32)
    anchor = np.loadtxt("inputfile/common/anchor32.txt", dtype='int')

    label, thetas = divide_sphere(anchor, v, n)

    with open("{}/label_overlapping_32vertices_theta0.55.txt".format(o_folder), 'wt') as fd_l:
        np.savetxt(fd_l, np.array(label), fmt="%d")

    with open("{}/thetas_overlapping_32vertices_theta0.55.txt".format(o_folder), 'wt') as fd_d:
        np.savetxt(fd_d, np.array(thetas), fmt="%f")


if __name__ == "__main__":
    spherical_division_main()
