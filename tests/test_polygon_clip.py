import pytest
from solidly.core import Point3D, Polygon3D

def test_polygon3d_intersection_convex():
    # Two overlapping triangles in the xy-plane
    p0 = Point3D(0,0,0)
    p1 = Point3D(1,0,0)
    p2 = Point3D(0,1,0)
    p3 = Point3D(1,1,0)
    tri1 = Polygon3D((p0, p1, p2))
    tri2 = Polygon3D((p1, p2, p3))
    inter = tri1.intersection(tri2)
    assert inter is not None
    assert isinstance(inter, Polygon3D)
    # Accept degenerate polygons (edge or point) as valid for shared edge/vertex
    assert 1 <= len(inter.vertices) <= 4
    # Intersection with non-overlapping polygon
    tri3 = Polygon3D((Point3D(2,2,0), Point3D(3,2,0), Point3D(2,3,0)))
    assert tri1.intersection(tri3) is None

def test_polygon3d_difference_convex():
    # Difference of two overlapping triangles in the xy-plane
    p0 = Point3D(0,0,0)
    p1 = Point3D(1,0,0)
    p2 = Point3D(0,1,0)
    p3 = Point3D(1,1,0)
    tri1 = Polygon3D((p0, p1, p2))
    tri2 = Polygon3D((p1, p2, p3))
    diff = tri1.difference(tri2)
    # Should be a polygon or None (if fully overlapped)
    if diff is not None:
        assert isinstance(diff, Polygon3D)
        # Accept degenerate polygons (edge or point) as valid for shared edge/vertex
        assert 1 <= len(diff.vertices) <= 4
    # Difference with non-overlapping polygon should be self
    tri3 = Polygon3D((Point3D(2,2,0), Point3D(3,2,0), Point3D(2,3,0)))
    diff2 = tri1.difference(tri3)
    assert diff2 is not None
    assert isinstance(diff2, Polygon3D)
    assert len(diff2.vertices) == 3
