# generate a 3d field of 20 points
import numpy as np
import plotly.graph_objects as go
import pandas as pd

N = 99

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
z = np.linspace(-1, 1, N)
X, Y, Z = np.meshgrid(x, y, z)

center = np.array([0.0, 0.0, 0.0])
r = 1.0
a = 0.95
b = 0.3

# compute the distance from center
YZ = np.sqrt((Y - center[1])**2 + (Z - center[2])**2) - r
l = np.sqrt(((X - center[0])/b)**2 + (YZ/a)**2)

# compute the field
field = np.log(1.0+8.0*(1.0-np.clip(l, 0.0, 1.0)))

# plot 4d Isosurface with plotly
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=field.flatten(),
    isomin=0.0,
    isomax=1.0,
    opacity=0.2,
    surface_count=5, # number of isosurfaces, 2 by default: only min and max
    colorscale='jet',
    caps=dict(x_show=False, y_show=False, z_show=False),
    ))
fig.show()