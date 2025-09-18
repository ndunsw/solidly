# Solidly
A modular Shapely-style 3D geometry library for Python

## Overview
Solidly is a 3D computational geometry library designed in the spirit of Shapely (2D). It provides robust primitives, predicates, boolean operations, and geometric transformations in a clean, NumPy-first API.

### Use cases:
- CAD / modeling workflows
- Robotics & path planning
- GIS & spatial analysis in 3D
- Scientific computing

## Features
## Core Primitives
- Point3D, Line3D, Plane, Polygon3D, Polyhedron, Mesh, Vector3
## Transformations
- translate, rotate (Rodrigues’ formula), scale
## Spatial Predicates
- contains, intersects, within, touches
- Robust for convex shapes and simple edge cases
## Boolean Operations
- Points, lines, polygons: union, intersection, difference
- Convex polyhedra: union (via convex hull), difference, intersection partially implemented (fails on vertex-only intersection)
- Coplanar convex polygons: intersection/difference via Sutherland–Hodgman
## Robustness & Validation
- is_simple, is_convex, is_closed, almost_equals
- Area, surface area, volume, convex hull, simplify

## Spatial Index & Queries
- AABB tree for fast spatial queries
- Nearest neighbor search
- Ray casting with Möller–Trumbore for Polyhedron.contains

## Visualization
- Lightweight 3D plotting via matplotlib (plot() methods on all primitives)
- PyVista / vedo hooks planned

## I/O & Interop
- Mesh import/export: OBJ, STL, PLY
- WKT 3D for points, lines, polygons
- Conversion from NumPy arrays
- Stubs for Trimesh/Open3D interop

## Example:
```
from solidly import Point3D, Polygon3D, Polyhedron

# Create a polygon
p, q, r = Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)
tri = Polygon3D([p, q, r])
print(tri.surface_area())  # 0.5

# Build a tetrahedron
tetra = Polyhedron.from_tetrahedron(p, q, r, Point3D(0,0,1))
print(tetra.volume())      # 0.1667
assert tetra.is_closed()

# Spatial query
assert tri.contains(Point3D(0.25,0.25,0))
```

## Roadmap
- Next sprint
    - Polygon–Plane clipping (Polygon3D ∩ Plane)
    - Convex polyhedron intersection & difference (half-space clipping)
    - More robust degeneracy handling
- Future
    - Full mesh boolean ops (non-convex, general CSG)
    - Topology utilities (edge manifold checks, mesh repair)
    - More formats (glTF, OFF)
    - Accelerated backend (C++/Rust/numba)
    - Visualization: interactive (PyVista, vedo)

Contributing
- The living design doc is in solidly.md.
- Every change must update it with summary, rationale, and limitations.
- Add/expand unit tests for new methods.
- Keep the API Shapely-style and composable.