"""Solidly: a small from-scratch 3D geometry library (prototype)."""
from .core import Vector3, Triangle, Mesh, Point3D, Line3D, Plane, Polygon3D, Polyhedron, convex_hull

__all__ = [
	"Vector3", "Triangle", "Mesh",
	"Point3D", "Line3D", "Plane", "Polygon3D", "Polyhedron", "convex_hull"
]
__version__ = "0.1.0"
