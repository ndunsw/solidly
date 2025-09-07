import pytest
from solidly.core import Point3D, Line3D, Plane, Polygon3D, Polyhedron


def test_point3d_basic():
    p = Point3D(1, 2, 3)
    arr = p.to_array()
    assert (arr == [1, 2, 3]).all()
    # Stubs: just check methods exist
    p.union(p)
    p.intersection(p)
    p.difference(p)
    p.contains(p)
    p.intersects(p)
    p.within(p)
    p.touches(p)
    p.translate([1, 0, 0])
    p.rotate([0, 0, 1], 1.57)
    p.scale(2)

def test_line3d_basic():
    p0 = Point3D(0, 0, 0)
    p1 = Point3D(1, 0, 0)
    l = Line3D(p0, p1)
    l.union(l)
    l.intersection(l)
    l.difference(l)
    l.contains(p0)
    l.intersects(l)
    l.within(l)
    l.touches(l)
    l.translate([0, 1, 0])
    l.rotate([0, 0, 1], 1.57)
    l.scale(2)

def test_plane_basic():
    p = Point3D(0, 0, 0)
    n = Point3D(0, 0, 1)
    from solidly.core import Vector3
    plane = Plane(p, Vector3(0, 0, 1))
    plane.contains(p)
    plane.intersection(plane)
    plane.translate([0, 0, 1])
    plane.rotate([1, 0, 0], 1.57)
    plane.scale(2)

def test_polygon3d_basic():
    verts = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)]
    poly = Polygon3D(tuple(verts))
    poly.union(poly)
    poly.intersection(poly)
    poly.difference(poly)
    poly.contains(verts[0])
    poly.intersects(poly)
    poly.within(poly)
    poly.touches(poly)
    poly.translate([0,0,1])
    poly.rotate([0,1,0], 1.57)
    poly.scale(2)
    poly.convex_hull()
    poly.simplify(0.1)
    poly.surface_area()

def test_polyhedron_basic():
    verts = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(0,0,1)]
    faces = [Polygon3D((verts[0], verts[1], verts[2])),
             Polygon3D((verts[0], verts[1], verts[3])),
             Polygon3D((verts[1], verts[2], verts[3])),
             Polygon3D((verts[0], verts[2], verts[3]))]
    polyh = Polyhedron(tuple(faces))
    polyh.union(polyh)
    polyh.intersection(polyh)
    polyh.difference(polyh)
    polyh.contains(verts[0])
    polyh.intersects(polyh)
    polyh.within(polyh)
    polyh.touches(polyh)
    polyh.translate([1,1,1])
    polyh.rotate([0,0,1], 1.57)
    polyh.scale(2)
    polyh.convex_hull()
    polyh.simplify(0.1)
    polyh.surface_area()
    polyh.volume()
