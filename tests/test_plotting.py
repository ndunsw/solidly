import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from solidly import Point3D, Line3D, Polygon3D, Polyhedron

def test_point3d_plot():
    p = Point3D(1,2,3)
    ax = plt.figure().add_subplot(111, projection='3d')
    p.plot(ax, color='r')
    plt.close()

def test_line3d_plot():
    l = Line3D(Point3D(0,0,0), Point3D(1,1,1))
    ax = plt.figure().add_subplot(111, projection='3d')
    l.plot(ax, color='g')
    plt.close()

def test_polygon3d_plot():
    poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    ax = plt.figure().add_subplot(111, projection='3d')
    poly.plot(ax, color='b')
    plt.close()

def test_polyhedron_plot():
    v = [Point3D(x, y, z) for x in (0,1) for y in (0,1) for z in (0,1)]
    faces = [
        (0,1,3,2), (4,5,7,6), (0,1,5,4), (2,3,7,6), (0,2,6,4), (1,3,7,5)
    ]
    polys = []
    for f in faces:
        polys.append(Polygon3D((v[f[0]], v[f[1]], v[f[2]])))
        polys.append(Polygon3D((v[f[0]], v[f[2]], v[f[3]])))
    cube = Polyhedron(tuple(polys))
    ax = plt.figure().add_subplot(111, projection='3d')
    cube.plot(ax, facecolor='cyan', edgecolor='k', alpha=0.3)
    plt.close()
