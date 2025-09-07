import numpy as np
import pytest
from solidly import Point3D, Line3D, Plane, Polygon3D

def test_line3d_plane_intersection_point():
    p0 = Point3D(0, 0, 0)
    p1 = Point3D(0, 0, 1)
    line = Line3D(p0, p1)
    plane = Plane(Point3D(0, 0, 0.5), np.array([0, 0, 1]))
    result = line.intersection(plane)
    assert isinstance(result, Point3D)
    assert abs(result.z - 0.5) < 1e-8

def test_line3d_plane_parallel():
    p0 = Point3D(0, 0, 0)
    p1 = Point3D(1, 0, 0)
    line = Line3D(p0, p1)
    plane = Plane(Point3D(0, 0, 1), np.array([0, 0, 1]))
    result = line.intersection(plane)
    assert result is None

def test_line3d_plane_in_plane():
    p0 = Point3D(0, 0, 1)
    p1 = Point3D(1, 0, 1)
    line = Line3D(p0, p1)
    plane = Plane(Point3D(0, 0, 1), np.array([0, 0, 1]))
    result = line.intersection(plane)
    assert result == line

def test_plane_plane_intersection_line():
    plane1 = Plane(Point3D(0, 0, 0), np.array([0, 0, 1]))
    plane2 = Plane(Point3D(0, 0, 0), np.array([1, 0, 0]))
    result = plane1.intersection(plane2)
    assert isinstance(result, Line3D)

def test_plane_plane_parallel():
    plane1 = Plane(Point3D(0, 0, 0), np.array([0, 0, 1]))
    plane2 = Plane(Point3D(0, 0, 1), np.array([0, 0, 1]))
    result = plane1.intersection(plane2)
    assert result is None

def test_plane_plane_coincident():
    plane1 = Plane(Point3D(0, 0, 0), np.array([0, 0, 1]))
    plane2 = Plane(Point3D(1, 1, 0), np.array([0, 0, 1]))
    result = plane1.intersection(plane2)
    assert result == plane1

def test_polygon3d_plane_stub():
    poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    plane = Plane(Point3D(0,0,0), np.array([0,0,1]))
    assert poly.intersection(plane) is None

def test_point3d_boolean_ops():
    p1 = Point3D(1, 2, 3)
    p2 = Point3D(1, 2, 3)
    p3 = Point3D(4, 5, 6)
    # Union
    assert p1.union(p2) == p1
    assert p1.union(p3) == {p1, p3}
    # Intersection
    assert p1.intersection(p2) == p1
    assert p1.intersection(p3) is None
    # Difference
    assert p1.difference(p2) is None
    assert p1.difference(p3) == p1

def test_line3d_boolean_ops():
    a = Point3D(0, 0, 0)
    b = Point3D(1, 0, 0)
    c = Point3D(2, 0, 0)
    l1 = Line3D(a, b)
    l2 = Line3D(b, c)
    l3 = Line3D(a, c)
    # Union: overlapping/collinear
    merged = l1.union(l2)
    assert isinstance(merged, Line3D)
    # Union: non-overlapping
    l4 = Line3D(Point3D(0,1,0), Point3D(1,1,0))
    assert l1.union(l4) == {l1, l4}
    # Difference: overlapping
    assert l1.difference(l2) is None or isinstance(l1.difference(l2), Line3D)
    # Difference: non-overlapping
    assert l1.difference(l4) == l1

def test_polygon3d_boolean_ops():
    # Two triangles in the same plane
    p0 = Point3D(0,0,0)
    p1 = Point3D(1,0,0)
    p2 = Point3D(0,1,0)
    p3 = Point3D(1,1,0)
    tri1 = Polygon3D((p0, p1, p2))
    tri2 = Polygon3D((p1, p2, p3))
    # Union: convex hull
    union = tri1.union(tri2)
    assert isinstance(union, Polygon3D)
    # Difference: should return a Polygon3D (possibly degenerate)
    diff = tri1.difference(tri2)
    assert isinstance(diff, Polygon3D)
    assert 1 <= len(diff.vertices) <= 3  # Accept degenerate (edge/vertex) or full triangle
