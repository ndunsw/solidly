"""Spatial indexing and nearest neighbor search for Solidly 3D geometries.

Includes:
- AABB (axis-aligned bounding box) utilities
- AABBTree for fast intersection and nearest neighbor queries
- Simple KD-tree fallback for nearest neighbor search
"""
import numpy as np
from collections import namedtuple

AABB = namedtuple('AABB', ['min', 'max'])

def compute_aabb(points):
    # Accepts a single Point3D or an iterable of points
    if hasattr(points, 'to_array'):
        arr = np.array([points.to_array()])
    else:
        arr = np.array([p.to_array() for p in points])
    return AABB(arr.min(axis=0), arr.max(axis=0))

class AABBNode:
    def __init__(self, aabb, left=None, right=None, obj=None):
        self.aabb = aabb
        self.left = left
        self.right = right
        self.obj = obj  # Leaf: reference to geometry (e.g., triangle, polygon, etc.)

    def is_leaf(self):
        return self.obj is not None

class AABBTree:
    def __init__(self, objects, get_aabb):
        """
        objects: list of geometry objects
        get_aabb: function(obj) -> AABB
        """
        self.root = self._build(objects, get_aabb)

    def _build(self, objects, get_aabb):
        if len(objects) == 1:
            return AABBNode(get_aabb(objects[0]), obj=objects[0])
        # Compute overall AABB
        aabbs = [get_aabb(obj) for obj in objects]
        mins = np.array([aabb.min for aabb in aabbs])
        maxs = np.array([aabb.max for aabb in aabbs])
        overall = AABB(mins.min(axis=0), maxs.max(axis=0))
        # Split along largest axis
        axis = np.argmax(overall.max - overall.min)
        centers = np.array([(aabb.min[axis] + aabb.max[axis]) / 2 for aabb in aabbs])
        order = np.argsort(centers)
        mid = len(objects) // 2
        left = [objects[i] for i in order[:mid]]
        right = [objects[i] for i in order[mid:]]
        return AABBNode(overall, left=self._build(left, get_aabb), right=self._build(right, get_aabb))

    def query_aabb(self, query_aabb, node=None, results=None):
        if node is None:
            node = self.root
        if results is None:
            results = []
        if not self._aabb_overlap(node.aabb, query_aabb):
            return results
        if node.is_leaf():
            results.append(node.obj)
        else:
            self.query_aabb(query_aabb, node.left, results)
            self.query_aabb(query_aabb, node.right, results)
        return results

    @staticmethod
    def _aabb_overlap(a, b):
        return np.all(a.max >= b.min) and np.all(b.max >= a.min)

    def nearest(self, point, node=None, best=None):
        if node is None:
            node = self.root
        if best is None:
            best = [None, float('inf')]
        # Compute distance from point to node's AABB
        dist = self._aabb_distance(point, node.aabb)
        if dist > best[1]:
            return best
        if node.is_leaf():
            obj = node.obj
            obj_center = (node.aabb.min + node.aabb.max) / 2
            d = np.linalg.norm(point - obj_center)
            if d < best[1]:
                best[0], best[1] = obj, d
        else:
            self.nearest(point, node.left, best)
            self.nearest(point, node.right, best)
        return best

    @staticmethod
    def _aabb_distance(point, aabb):
        # Clamp point to aabb
        clamped = np.maximum(aabb.min, np.minimum(point, aabb.max))
        return np.linalg.norm(point - clamped)

# Simple KD-tree fallback for nearest neighbor search
try:
    from scipy.spatial import KDTree
    def nearest_neighbor(points, query):
        arr = np.array([p.to_array() for p in points])
        tree = KDTree(arr)
        dist, idx = tree.query(query)
        return points[idx], dist
except ImportError:
    def nearest_neighbor(points, query):
        arr = np.array([p.to_array() for p in points])
        dists = np.linalg.norm(arr - query, axis=1)
        idx = np.argmin(dists)
        return points[idx], dists[idx]
