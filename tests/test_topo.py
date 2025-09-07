from solidly import Point3D, Polygon3D, convex_hull

def test_convex_hull_2d():
    pts = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(0.5,0.5,0)]
    hull = convex_hull(pts)
    assert isinstance(hull, Polygon3D)
    assert all(p in hull.vertices for p in pts if p != Point3D(0.5,0.5,0))

def test_polygon3d_simplify():
    pts = [Point3D(0,0,0), Point3D(0.5,0.01,0), Point3D(1,0,0), Point3D(1,1,0), Point3D(0,1,0)]
    poly = Polygon3D(tuple(pts))
    simple = poly.simplify(0.05)
    assert isinstance(simple, Polygon3D)
    assert len(simple.vertices) < len(poly.vertices)
