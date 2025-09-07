"""Lightweight 3D plotting utilities for Solidly geometries using matplotlib (Axes3D).

Optional: PyVista/vedo hooks for interactive visualization.
"""
import numpy as np

def plot_point3d(point, ax=None, **kwargs):
    ax = ax or _get_3d_ax()
    arr = point.to_array()
    ax.scatter([arr[0]], [arr[1]], [arr[2]], **kwargs)
    return ax

def plot_line3d(line, ax=None, **kwargs):
    ax = ax or _get_3d_ax()
    a, b = line.p0.to_array(), line.p1.to_array()
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **kwargs)
    return ax

def plot_polygon3d(poly, ax=None, **kwargs):
    ax = ax or _get_3d_ax()
    verts = np.array([v.to_array() for v in poly.vertices])
    # Close the polygon
    verts = np.vstack([verts, verts[0]])
    ax.plot(verts[:,0], verts[:,1], verts[:,2], **kwargs)
    return ax

def plot_polyhedron(polyh, ax=None, facecolor='cyan', edgecolor='k', alpha=0.3, **kwargs):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax = ax or _get_3d_ax()
    faces = []
    for face in polyh.faces:
        verts = np.array([v.to_array() for v in face.vertices])
        faces.append(verts)
    poly3d = Poly3DCollection(faces, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs)
    ax.add_collection3d(poly3d)
    return ax

def _get_3d_ax():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.gcf()
    if not fig.axes:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0]
    return ax

# Optional: PyVista/vedo hooks (not implemented, placeholder)
def to_pyvista(mesh):
    raise NotImplementedError('PyVista support not yet implemented.')
def to_vedo(mesh):
    raise NotImplementedError('vedo support not yet implemented.')
