import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def theta_phi_to_cartesian(t, p, r):
    return r * np.array([
        np.sin(p) * np.cos(t),
        np.sin(p) * np.sin(t),
        np.cos(p)
    ])

def theta_phi_to_cartesian_norm(t, p, r):
    l = theta_phi_to_cartesian(t, p, r)
    return l / np.linalg.norm(l)

def calculate_intersection(n_points, r, norm, offset_angle):
    # Number of points to generate for the intersection circle
    points = []

    w = np.sin((offset_angle*np.pi/180))
    offset = w * norm
    r = np.sqrt(r**2 - np.power(np.dot(norm, offset) / np.linalg.norm(norm), 2))

    # Vector 1: perpendicular to normal, in the plane of intersection
    v1 = np.array([1, 1, -(norm[0]+norm[1]) / norm[2]])
    v1 /= np.linalg.norm(v1)  # Normalize

    # Vector 2: perpendicular to both the normal vector and v1
    v2 = np.cross(norm, v1)
    v2 /= np.linalg.norm(v2)  # Normalize

    # Parametrize the circle using the angle around the circle (theta)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        
        # Parametrize the circle along v1 and v2
        x = r * np.cos(angle) * v1[0] + r * np.sin(angle) * v2[0]
        y = r * np.cos(angle) * v1[1] + r * np.sin(angle) * v2[1]
        z = r * np.cos(angle) * v1[2] + r * np.sin(angle) * v2[2]
        
        points.append([x,y,z])

    # Convert the list of points to a numpy array
    points = np.array(points) + offset
    return points

# Rodrigues' rotation formula
def rotate_point(point, axis, theta):
    # Convert inputs to numpy arrays
    point = np.array(point)
    axis = np.array(axis)
    theta = np.deg2rad(theta)
    
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    a, b, c = axis

    # Rotation matrix using Rodrigues' formula
    K = np.array([[0, -c, b], 
                  [c, 0, -a], 
                  [-b, a, 0]])

    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (np.dot(K, K))

    # Apply rotation
    rotated_point = np.dot(R, point)
    rotated_point = rotated_point / np.linalg.norm(rotated_point)
    
    return np.array(rotated_point)

def get_relative_theta_phi(norm, point):
    new_z = norm
    new_x = 0
    if norm[2] == 1:
        new_x = np.array([1, 0, 0])
    elif norm[2] == -1:
        new_x = np.array([-1, 0, 0])
    else:
        new_x = np.array([-1 * norm[1], norm[0], 0]) * 1/np.sqrt(norm[0]**2 + norm[1] ** 2)
    new_y = np.cross(new_z, new_x)

    new_px = point[0] * new_x[0] + point[0] * new_x[1] + point[0] * new_x[2]
    new_py = point[1] * new_y[0] + point[1] * new_y[1] + point[1] * new_y[2]
    new_pz = point[2] * new_z[0] + point[2] * new_z[1] + point[2] * new_z[2]

    new_p = np.array([new_px, new_py, new_pz])
    new_p = new_p / np.linalg.norm(new_p)

    new_theta = np.arctan2(new_py, new_px)
    new_phi = np.arccos(new_pz)

    return np.array([new_theta, new_phi])

# Latitude and longitude (in degrees)
lng = -77.675936
lat = 43.082149

# Convert latitude and longitude to radians
theta = np.deg2rad(lng) # Longitude
phi = np.deg2rad(lat) # Latitude

up_vec = np.array([0, 0, 1])

# Earth's diameter and radius (in kilometers)
E_diam = 12_756  # Earth's diameter in kilometers
E_radius = E_diam / 2  # Earth's radius in kilometers
E_axis = np.array([np.sin(np.radians(23.44)), 0, np.cos(np.radians(23.44))])
E_axis /=  np.linalg.norm(E_axis)

#  Calculate the 3D Cartesian coordinates of the given location and normalize
loc_norm = theta_phi_to_cartesian_norm(theta, phi, E_radius)

print(f'RIT normal vector:\t{loc_norm}')

horizon_points = calculate_intersection(100, 1, loc_norm, 0)
angle_limited_points = calculate_intersection(100, 1, loc_norm, 20)

print('Processing SIMBAD data...')

simbad_data = pd.read_csv('data.csv')
simbad_data['theta'] = np.deg2rad(simbad_data['dec'])
simbad_data['phi'] = np.deg2rad(simbad_data['ra'])
simbad_points = list()

simbad_data.apply(
    lambda x: simbad_points.append(
        theta_phi_to_cartesian_norm(x['theta'], x['phi'], 1)
    ), axis=1)
simbad_points = np.array(simbad_points)

print('SIMBAD data processing complete!')

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.xlim(-1.25, 1.25)
plt.ylim(-1.25, 1.25)
ax.set_zlim(-1.25, 1.25)

# Earth's axis
ax.plot([-3 * E_axis[0], 3 * E_axis[0]], 
        [-3 * E_axis[1], 3 * E_axis[1]], 
        [-3 * E_axis[2], 3 * E_axis[2]]
)

# Up vector
ax.plot([-3 * up_vec[0], 3 * up_vec[0]], 
        [-3 * up_vec[1], 3 * up_vec[1]], 
        [-3 * up_vec[2], 3 * up_vec[2]]
)

p_idx = 150
#relative_angles = get_relative_theta_phi(loc_norm, simbad_points[150])
#relative_p = theta_phi_to_cartesian_norm(relative_angles[0], relative_angles[1], 1)
#ax.scatter3D(relative_p[0], relative_p[1], relative_p[2], color='purple')

ax.scatter3D(simbad_points[p_idx, 0], simbad_points[p_idx, 1], simbad_points[p_idx, 2], color='black')
time_p = rotate_point(simbad_points[p_idx], [0, 0, 1], 180)
ax.scatter3D(time_p[0], time_p[1], time_p[2], color='orange')

ax.scatter3D(horizon_points[:, 0], horizon_points[:, 1], horizon_points[:, 2], color='blue')
ax.scatter3D(angle_limited_points[:, 0], angle_limited_points[:, 1], angle_limited_points[:, 2], color='red')
ax.scatter3D(loc_norm[0], loc_norm[1], loc_norm[2], color='green')

plt.show()

# Loop: rotate theta and phi along [0, 0, 1], 15deg every hour
#   Get theta and phi for simbad objects
#   Get their xyz coords on unit sphere
#   Get relative angles using location normal using xyz coords