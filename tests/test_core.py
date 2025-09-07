import numpy as np
from solidly import Mesh

def test_tetrahedron_volume_area():
    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    assert abs(mesh.volume() - 1.0/6.0) < 1e-9
    assert mesh.surface_area() > 0
    assert mesh.is_closed() 