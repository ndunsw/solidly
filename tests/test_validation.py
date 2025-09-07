import pytest
import numpy as np
from solidly import Point3D, Line3D, Polygon3D, Polyhedron, Mesh, Triangle

def test_point3d_almost_equals():
    a = Point3D(1, 2, 3)
    b = Point3D(1+1e-9, 2-1e-9, 3)
    assert a.almost_equals(b)
    assert not a.almost_equals(Point3D(1, 2, 4))

def test_line3d_is_simple():
    a = Point3D(0,0,0)
    b = Point3D(1,0,0)
    l = Line3D(a, b)
    assert l.is_simple()
    l2 = Line3D(a, a)
    assert not l2.is_simple()

def test_polygon3d_is_simple_and_convex():
    p = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    assert p.is_simple()
    assert p.is_convex()

def test_polyhedron_is_closed_and_convex():
    v = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)]
    faces = [Polygon3D((v[0], v[1], v[2])),
             Polygon3D((v[0], v[1], v[3])),
             Polygon3D((v[1], v[2], v[3])),
             Polygon3D((v[0], v[2], v[3]))]
    poly = Polyhedron(tuple(faces))
    assert poly.is_closed()
    assert poly.is_convex()

def test_triangle_is_degenerate():
    t = Triangle(np.array([0,0,0]), np.array([1,0,0]), np.array([2,0,0]))
    assert t.is_degenerate()
    t2 = Triangle(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]))
    assert not t2.is_degenerate()

def test_mesh_has_degenerate_faces():
    verts = np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    assert mesh.has_degenerate_faces()
    verts2 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces2 = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])
    mesh2 = Mesh.from_vertices_faces(verts2, faces2)
    assert not mesh2.has_degenerate_faces()
