import numpy as np
import math
import random

def coordinate_transformation_z(original_normal):

	on_xy = np.array([original_normal[0], original_normal[1], 0.0])
	on_xy /= np.linalg.norm(on_xy)
	nn_xy = np.array([1.0, 0.0, 0.0])
	nn_xy /= np.linalg.norm(nn_xy)
	outer_z = np.cross(on_xy, nn_xy)
	theta_z = math.asin(np.linalg.norm(outer_z)/(np.linalg.norm(on_xy)*np.linalg.norm(nn_xy)))
	
	if outer_z[2] < 0:
		theta_z = 2*math.pi - theta_z

	# Rotation matrix
	if original_normal[0]> 0:
		Rz = np.array([[math.cos(theta_z), -1.0*math.sin(theta_z), 0.0], [math.sin(theta_z), math.cos(theta_z), 0.0], [0.0, 0.0, 1.0]])
	else:
		theta_z *= -1
		theta_z += math.pi
		Rz = np.array([[math.cos(theta_z), -1.0*math.sin(theta_z), 0.0], [math.sin(theta_z), math.cos(theta_z), 0.0], [0.0, 0.0, 1.0]])

	return Rz


def coordinate_transformation(original_normal):
	# Calculate theta_z
	# External product of (xn, yn, 0) and (1, 1, 0)
	Rz = coordinate_transformation_z(original_normal)
    
	original_normal_xy = np.dot(Rz, original_normal)

	# Calculate theta_y
	on_xz = np.array([original_normal_xy[0], 0.0, original_normal_xy[2]])
	on_xz /= np.linalg.norm(on_xz)
	nn_xz = np.array([0.0, 0.0, 1.0])
	nn_xz /= np.linalg.norm(nn_xz)
	outer_y = np.cross(on_xz, nn_xz)
	theta_y = math.asin(np.linalg.norm(outer_y))

	if outer_y[1] < 0:
		theta_y = 2*math.pi - theta_y
	# Rotation matrix
	if original_normal_xy[2]> 0:
		Ry = np.array([[math.cos(theta_y), 0.0, math.sin(theta_y)], [0.0, 1.0, 0.0], [-1.0*math.sin(theta_y), 0.0, math.cos(theta_y)]])
	else:
		theta_y *= -1
		theta_y += math.pi
		Ry = np.array([[math.cos(theta_y), 0.0, math.sin(theta_y)], [0.0, 1.0, 0.0], [-1.0*math.sin(theta_y), 0.0, math.cos(theta_y)]])
    
	R = np.dot(Ry, Rz)

	return R


def func55(param, x, y): 
    p0, px, py, pxx, pyy, pxy, pxxy, pxyy, pxxx, pyyy, pxxxy, pxxyy, pxyyy, pxxxx, pyyyy, pxxxxy, pxxxyy, pxxyyy, pxyyyy, pxxxxx, pyyyyy = param
    return  p0 + px*x +py*y +pxx*x*x + pyy*y*y + pxy*x*y + pxxy*x*x*y + pxyy*x*y*y + pxxx*x*x*x + pyyy*y*y*y + pxxxy*x*x*x*y + pxxyy*x*x*y*y + pxyyy*x*y*y*y + pxxxx*x*x*x*x + pyyyy*y*y*y*y + pxxxxy*x*x*x*x*y + pxxxyy*x*x*x*y*y + pxxyyy*x*x*y*y*y + pxyyyy*x*y*y*y*y + pxxxxx*x*x*x*x*x + pyyyyy*y*y*y*y*y


def fit_func55(points):
	v = points
	data_x = v[:, 0]
	data_y = v[:, 1]
	obj = v[:, 2]
	data_n = data_x.shape[0]
	exp=np.array([np.ones(data_n),data_x,data_y,(lambda x: x*x)(data_x),(lambda y: y*y)(data_y),(lambda x,y: x*y)(data_x,data_y),(lambda x,y: x*x*y)(data_x,data_y),(lambda x,y: x*y*y)(data_x,data_y),(lambda x: x*x*x)(data_x),(lambda y: y*y*y)(data_y),(lambda x,y: x*x*x*y)(data_x,data_y), (lambda x,y: x*x*y*y)(data_x,data_y), (lambda x,y: x*y*y*y)(data_x,data_y),(lambda x: x*x*x*x)(data_x),(lambda y: y*y*y*y)(data_y),(lambda x,y: x*x*x*x*y)(data_x,data_y), (lambda x,y: x*x*x*y*y)(data_x,data_y), (lambda x,y: x*x*y*y*y)(data_x,data_y),(lambda x,y: x*y*y*y*y)(data_x,data_y),(lambda x: x*x*x*x*x)(data_x),(lambda y: y*y*y*y*y)(data_y)])
	# Estimation of surface parameters
	popt=np.linalg.lstsq(exp.T,obj)[0]

	# Surface parameters
	return popt


def sampling_from_ellips_distance(parameters, num_of_sampling_points, params_of_surface, vertices_in_new_coordinate):

	x_min = vertices_in_new_coordinate[:, 0].min()
	x_max = vertices_in_new_coordinate[:, 0].max()
	y_min = vertices_in_new_coordinate[:, 1].min()
	y_max = vertices_in_new_coordinate[:, 1].max()
	z_min = vertices_in_new_coordinate[:, 2].min()
	z_max = vertices_in_new_coordinate[:, 2].max()
	
	a = 1.0/(math.sqrt(parameters[0]))
	b = 1.0/(math.sqrt(parameters[1]))

	error_flag = 0
	if -1*a > x_max or a < x_min:
		error_flag = 1

	sampled_points = []
	distance = []
	counter = 0
	while len(sampled_points) < 1.5*num_of_sampling_points:
		rand_x = [random.uniform(-1*a,a) for i in range(20*num_of_sampling_points)]
		rand_y = [random.uniform(-1*b, b) for i in range(20*num_of_sampling_points)]

		for x, y in zip(rand_x, rand_y):
			r = (x**2*parameters[0]) + (y**2*parameters[1])
			while r > 1 and error_flag==0:
				x *= 0.9
				y *= 0.9
				r = (x**2*parameters[0]) + (y**2*parameters[1])
			
			if x_min < x and x < x_max and y_min < y and y < y_max:
				z = func55(params_of_surface, x, y)
				if z_min < z and z < z_max:
					v_diff = vertices_in_new_coordinate - np.array([x, y, z])
					v_power = np.power(v_diff,2)
					v_sum = np.sum(v_power, axis=1)
					v_sqrt = np.sqrt(v_sum)
					dist = np.min(v_sqrt)

					sampled_points.append(np.array([x, y, z]))
					distance.append([counter, dist])
					counter += 1

	# Sort in ascending order with respect to distance
	distance.sort(key=lambda x:x[1])
	ids = np.array(distance, dtype='int')[:, 0]
	sampled_points = np.array(sampled_points, dtype='float')
	sampled_points_after_sorted = sampled_points[ids]

	sampled_points = sampled_points_after_sorted[0:num_of_sampling_points]

	return sampled_points


