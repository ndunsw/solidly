import numpy as np
import pytest
from solidly import Point3D, Line3D, Plane, Polygon3D, Polyhedron

def test_point3d_predicates():
    p = Point3D(1,2,3)
    q = Point3D(1,2,3)
    r = Point3D(1,2,4)
    assert p.contains(q)
    assert p.intersects(q)
    assert not p.contains(r)
    assert not p.touches(r)
    assert p.within(q)

def test_line3d_predicates():
    a = Point3D(0,0,0)
    b = Point3D(1,0,0)
    l = Line3D(a, b)
    p_on = Point3D(0.5,0,0)
    p_off = Point3D(0,1,0)
    l2 = Line3D(Point3D(0,0,0), Point3D(0.5,0,0))
    l3 = Line3D(Point3D(1,0,0), Point3D(2,0,0))
    assert l.contains(p_on)
    assert not l.contains(p_off)
    assert l.contains(l2)
    assert not l.contains(l3)
    assert l.intersects(l2)
    assert not l.touches(l2)
    assert l.within(Line3D(Point3D(-1,0,0), Point3D(2,0,0)))

def test_plane_predicates():
    plane = Plane(Point3D(0,0,0), [0,0,1])
    p_on = Point3D(1,2,0)
    p_off = Point3D(1,2,1)
    plane2 = Plane(Point3D(0,0,0), [0,0,1])
    plane3 = Plane(Point3D(0,0,1), [0,0,1])
    assert plane.contains(p_on)
    assert not plane.contains(p_off)
    assert plane.intersects(plane2)
    assert not plane.touches(plane2)
    assert plane.touches(plane3)
    assert plane.within(plane2)

def test_polygon3d_predicates():
    poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    p_in = Point3D(0.2,0.2,0)
    p_out = Point3D(1,1,0)
    poly2 = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    assert poly.contains(p_in)
    assert not poly.contains(p_out)
    assert poly.intersects(poly2)
    assert poly.within(poly2)
    assert not poly.touches(poly2)

def test_polyhedron_predicates():
    v = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)]
    faces = [Polygon3D((v[0], v[1], v[2])),
             Polygon3D((v[0], v[1], v[3])),
             Polygon3D((v[1], v[2], v[3])),
             Polygon3D((v[0], v[2], v[3]))]
    poly = Polyhedron(tuple(faces))
    p_in = Point3D(0.1,0.1,0.1)
    p_out = Point3D(1,1,1)
    poly2 = Polyhedron(tuple(faces))
    assert poly.contains(p_in)
    assert not poly.contains(p_out)
    assert poly.intersects(poly2)
    assert poly.within(poly2)
    assert not poly.touches(poly2)
