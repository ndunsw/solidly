import numpy as np
import pytest
from solidly import Point3D
from solidly.spatial import AABBTree, compute_aabb, nearest_neighbor

def test_aabb_tree_query():
    pts = [Point3D(0,0,0), Point3D(1,1,1), Point3D(2,2,2)]
    tree = AABBTree(pts, compute_aabb)
    box = compute_aabb([Point3D(0.5,0.5,0.5), Point3D(1.5,1.5,1.5)])
    found = tree.query_aabb(box)
    assert Point3D(1,1,1) in found
    assert Point3D(0,0,0) not in found

def test_aabb_tree_nearest():
    pts = [Point3D(0,0,0), Point3D(1,1,1), Point3D(2,2,2)]
    tree = AABBTree(pts, compute_aabb)
    pt, dist = tree.nearest(np.array([1.2,1.2,1.2]))
    assert isinstance(pt, Point3D)
    assert abs(dist - np.linalg.norm(np.array([1,1,1])-np.array([1.2,1.2,1.2]))) < 1e-6

def test_nearest_neighbor():
    pts = [Point3D(0,0,0), Point3D(1,1,1), Point3D(2,2,2)]
    pt, dist = nearest_neighbor(pts, [1.1,1.1,1.1])
    assert pt == Point3D(1,1,1)
    assert abs(dist - np.linalg.norm(np.array([1,1,1])-np.array([1.1,1.1,1.1]))) < 1e-6
