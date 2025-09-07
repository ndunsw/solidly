#!/usr/bin/env python3
from solidly import Mesh
import numpy as np

def demo_tetra():
    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    print("Tetra mesh:", mesh)
    print("BBox:", mesh.bounding_box())
    print("Surface area:", mesh.surface_area())
    print("Volume:", mesh.volume())
    print("Closed:", mesh.is_closed())

if __name__ == '__main__':
    demo_tetra()
