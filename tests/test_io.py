import numpy as np
import os
from solidly import Mesh, Point3D, Line3D, Polygon3D

def test_mesh_obj_io(tmp_path):
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]])
    faces = np.array([[0,1,2]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    path = tmp_path / 'test.obj'
    mesh.to_obj(str(path))
    mesh2 = Mesh.from_obj(str(path))
    assert np.allclose(mesh.vertices, mesh2.vertices)
    assert np.all(mesh.faces == mesh2.faces)

def test_mesh_stl_io(tmp_path):
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]])
    faces = np.array([[0,1,2]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    path = tmp_path / 'test.stl'
    mesh.to_stl(str(path))
    mesh2 = Mesh.from_stl(str(path))
    assert np.allclose(mesh.vertices, mesh2.vertices)
    assert mesh2.faces.shape[1] == 3

def test_mesh_ply_io(tmp_path):
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]])
    faces = np.array([[0,1,2]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    path = tmp_path / 'test.ply'
    mesh.to_ply(str(path))
    mesh2 = Mesh.from_ply(str(path))
    assert np.allclose(mesh.vertices, mesh2.vertices)
    assert np.all(mesh.faces == mesh2.faces)

def test_mesh_numpy():
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]])
    faces = np.array([[0,1,2]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    v2, f2 = mesh.to_numpy()
    mesh2 = Mesh.from_numpy(v2, f2)
    assert np.allclose(mesh.vertices, mesh2.vertices)
    assert np.all(mesh.faces == mesh2.faces)

def test_wkt():
    p = Point3D(1,2,3)
    assert p.to_wkt() == 'POINT Z (1 2 3)'
    l = Line3D(Point3D(0,0,0), Point3D(1,1,1))
    assert l.to_wkt().startswith('LINESTRING Z')
    poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    assert poly.to_wkt().startswith('POLYGON Z')
