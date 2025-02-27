import numpy as np

lat = 43.082149
lng = -77.675936

theta = lng * np.pi / 180
psi = lat * np .pi / 180

E_diam = 12_756_000 # Earths diameter in meters

location = np.sqrt(E_diam) * np.array([
    np.sin(psi) * np.cos(theta),
    np.sin(psi) * np.sin(theta),
    np.cos(psi)
])

location_norm = location / np.sqrt(np.sum(np.power(location, 2)))

print(f'(X,Y,Z):\t{location}\nNormal vector:\t{location_norm}')
