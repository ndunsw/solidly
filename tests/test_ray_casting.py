import numpy as np
from solidly import Point3D, Polygon3D, Polyhedron

def make_unit_cube():
    # Vertices
    v = [Point3D(x, y, z) for x in (0,1) for y in (0,1) for z in (0,1)]
    # Faces (each as a quad split into two triangles)
    faces = [
        (0,1,3,2), (4,5,7,6), (0,1,5,4), (2,3,7,6), (0,2,6,4), (1,3,7,5)
    ]
    polys = []
    for f in faces:
        polys.append(Polygon3D((v[f[0]], v[f[1]], v[f[2]])))
        polys.append(Polygon3D((v[f[0]], v[f[2]], v[f[3]])))
    return Polyhedron(tuple(polys))

def test_ray_intersects_cube():
    cube = make_unit_cube()
    # Ray from outside, through center
    origin = np.array([-1,0.5,0.5])
    direction = np.array([1,0,0])
    assert cube.ray_intersects(origin, direction)
    # Ray from inside, should always intersect
    origin = np.array([0.5,0.5,0.5])
    direction = np.array([1,0,0])
    assert cube.ray_intersects(origin, direction)
    # Ray missing the cube
    origin = np.array([-1,-1,-1])
    direction = np.array([-1,0,0])
    assert not cube.ray_intersects(origin, direction)
