import numpy as np
import pytest
from solidly import Point3D, Line3D, Plane, Polygon3D, Polyhedron

def test_point3d_transform():
    p = Point3D(1,2,3)
    assert p.translate([1,0,0]).almost_equals(Point3D(2,2,3))
    assert p.scale(2).almost_equals(Point3D(2,4,6))
    # 90 deg rotation about z
    p2 = p.rotate([0,0,1], np.pi/2)
    assert np.allclose([p2.x, p2.y, p2.z], [-2,1,3], atol=1e-8)

def test_line3d_transform():
    l = Line3D(Point3D(0,0,0), Point3D(1,0,0))
    l2 = l.translate([0,1,0])
    assert l2.p0.almost_equals(Point3D(0,1,0))
    assert l2.p1.almost_equals(Point3D(1,1,0))
    l3 = l.scale(2)
    assert l3.p1.almost_equals(Point3D(2,0,0))
    l4 = l.rotate([0,0,1], np.pi/2)
    assert np.allclose([l4.p1.x, l4.p1.y, l4.p1.z], [0,1,0], atol=1e-8)

def test_plane_transform():
    plane = Plane(Point3D(1,2,3), [0,0,1])
    plane2 = plane.translate([1,0,0])
    assert plane2.point.almost_equals(Point3D(2,2,3))
    plane3 = plane.scale(2)
    assert plane3.point.almost_equals(Point3D(2,4,6))
    plane4 = plane.rotate([0,0,1], np.pi/2)
    assert np.allclose([plane4.point.x, plane4.point.y, plane4.point.z], [-2,1,3], atol=1e-8)
    assert np.allclose(plane4.normal.to_array(), [0,0,1], atol=1e-8)

def test_polygon3d_transform():
    poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    poly2 = poly.translate([1,1,1])
    for v, v2 in zip(poly.vertices, poly2.vertices):
        assert v2.almost_equals(v.translate([1,1,1]))
    poly3 = poly.scale(2)
    for v, v3 in zip(poly.vertices, poly3.vertices):
        assert v3.almost_equals(v.scale(2))
    poly4 = poly.rotate([0,0,1], np.pi/2)
    arr = np.array([v.to_array() for v in poly4.vertices])
    arr0 = np.array([v.to_array() for v in poly.vertices])
    # Should be rotated 90 deg in xy
    assert np.allclose(arr[:,0], -arr0[:,1], atol=1e-8)
    assert np.allclose(arr[:,1], arr0[:,0], atol=1e-8)

def test_polyhedron_transform():
    v = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)]
    faces = [Polygon3D((v[0], v[1], v[2])),
             Polygon3D((v[0], v[1], v[3])),
             Polygon3D((v[1], v[2], v[3])),
             Polygon3D((v[0], v[2], v[3]))]
    poly = Polyhedron(tuple(faces))
    poly2 = poly.translate([1,2,3])
    for f, f2 in zip(poly.faces, poly2.faces):
        for v, v2 in zip(f.vertices, f2.vertices):
            assert v2.almost_equals(v.translate([1,2,3]))
    poly3 = poly.scale(2)
    for f, f3 in zip(poly.faces, poly3.faces):
        for v, v3 in zip(f.vertices, f3.vertices):
            assert v3.almost_equals(v.scale(2))
    poly4 = poly.rotate([0,0,1], np.pi/2)
    arr = np.array([v.to_array() for f in poly4.faces for v in f.vertices])
    arr0 = np.array([v.to_array() for f in poly.faces for v in f.vertices])
    # Should be rotated 90 deg in xy
    assert np.allclose(arr[:,0], -arr0[:,1], atol=1e-8)
    assert np.allclose(arr[:,1], arr0[:,0], atol=1e-8)
