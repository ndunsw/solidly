## Geometric Transformations (September 2025)

### Summary of Change
Implemented geometric transformations (`translate`, `rotate`, `scale`) for all core primitives:
- `Point3D`, `Line3D`, `Plane`, `Polygon3D`, `Polyhedron`, and `Vector3`
- All methods return new instances and use robust vector math
- Thoroughly tested for correctness and immutability

### Rationale
- Enables flexible manipulation and composition of 3D geometries
- Essential for interoperability, modeling, and geometric algorithms

### New Limitations/Considerations
- Only uniform scaling is supported (no shearing or non-uniform scale)
- Rotation uses right-hand rule and Rodrigues' formula
- Transformations assume Euclidean 3D space

---
## Spatial Predicates (September 2025)

### Summary of Change
Implemented spatial predicates (`contains`, `intersects`, `within`, `touches`) for all core primitives:
- `Point3D`, `Line3D`, `Plane`, `Polygon3D`, `Polyhedron`
- Covers point equality, point-in-polygon, point-in-polyhedron, line/plane/polygon/polyhedron overlap, and touching logic
- All methods are robust for convex shapes and simple cases

### Rationale
- Enables Shapely-style geometric queries and workflows
- Lays the foundation for robust boolean and topological operations

### New Limitations/Considerations
- Only convex polygons/polyhedra are fully supported for containment/intersection
- Some edge cases (non-convex, degenerate, or non-coplanar) may not be handled
- Touching logic is simple (vertex/endpoint sharing)

---
## Boolean Operations: Convex Polyhedra (September 2025)

### Summary of Change
Implemented MVP boolean operations for convex polyhedra:
- `Polyhedron.union(Polyhedron)`: Returns convex hull of all vertices (requires scipy)
- `Polyhedron.intersection(Polyhedron)`: Not implemented (would require half-space intersection)
- `Polyhedron.difference(Polyhedron)`: Not implemented
All methods only work for convex polyhedra (e.g., box, tetrahedron).

### Rationale
- Enables basic CSG workflows for convex shapes, which are common in CAD and modeling.
- Convex hull is a simple, robust over-approximation for union.
- Lays groundwork for more advanced mesh boolean operations in the future.

### New Limitations/Considerations
- Only convex polyhedra are supported; non-convex and general mesh booleans are future work.
- `intersection` and `difference` are not implemented for MVP.
- `union` requires `scipy` (for ConvexHull); if not available, returns None.

---

## Boolean Operations: Intersections (September 2025)


### Summary of Change
Implemented robust boolean operations for:
- `Point3D.union(Point3D)`: Returns self if equal, else set of both points.
- `Point3D.intersection(Point3D)`: Returns self if equal, else None.
- `Point3D.difference(Point3D)`: Returns self if not equal, else None.
- `Line3D.union(Line3D)`: If collinear and overlapping, returns merged segment; else set of both lines.
- `Line3D.intersection(Line3D)`: Returns intersection point, self, or None for skew/parallel.
- `Line3D.difference(Line3D)`: Returns self minus overlap if collinear, else self.
- `Polygon3D.union(Polygon3D)`: For convex, coplanar polygons, returns convex hull of all vertices.
- `Polygon3D.intersection(Polygon3D)`: Returns intersection polygon for coplanar convex polygons, or degenerate (edge/vertex) if overlap is lower-dimensional, or None if disjoint or not coplanar.
- `Polygon3D.difference(Polygon3D)`: Returns the difference polygon for coplanar convex polygons. If polygons do not overlap, returns self. If overlap is degenerate (edge/vertex), returns the appropriate degenerate polygon. Never returns None for non-overlapping polygons.
- Scaffolded `Polygon3D ∩ Plane` (stub)
All methods handle degenerate, parallel, and non-overlapping cases, and are covered by new unit tests.

### Rationale
- These are the simplest and most robust 3D boolean operations, foundational for more complex CSG and mesh boolean ops.
- Robust handling of edge cases (parallel, coincident, coplanarity, etc.) is critical for downstream geometry workflows.
- Extends boolean operations to points, lines, and polygons for Shapely-style API consistency.

### New Limitations/Considerations
- Only convex, coplanar polygons are supported for polygon booleans.
- `Polygon3D ∩ Plane` is a stub; full implementation will require polygon clipping.
- Polyhedron and mesh booleans are only implemented for convex union (see above); intersection/difference are not yet implemented.

### Module Structure (Boolean Operations)
- All primitives (`Line3D`, `Plane`, `Polygon3D`, `Polyhedron`) implement an `intersection` method.
- `Line3D.intersection(Line3D)` computes intersection or overlap, or None for skew/parallel.
- `Polygon3D.intersection(Polygon3D)` uses 2D projection and Sutherland-Hodgman for coplanar convex polygons.
- `Polygon3D.intersection(Plane)` is stubbed for future polygon-plane clipping.
- Polyhedron and mesh boolean operations are not yet implemented; planned for convex shapes first.

---

---
## Robustness & Validation (September 2025)

### Summary of Change
- `is_simple`, `is_convex`, `is_closed`, `almost_equals` (tolerance-based equality) are now fully implemented for `Polygon3D` and `Polyhedron` (including robust edge and convexity checks for `Polyhedron`).
- `convex_hull`, `simplify`, `surface_area`, and `volume` are now fully implemented for `Polygon3D` and `Polyhedron`.
- Degeneracy checks for triangles and meshes (e.g., zero-area triangles)

### Rationale
- Ensures all geometric primitives are well-formed before running boolean or topological operations
- Lays groundwork for robust, numerically stable algorithms
- Tolerance-based comparisons help avoid floating-point issues (like Shapely’s `almost_equals`)

### New Limitations/Considerations
- Some validation methods for non-convex or degenerate cases may still be limited.
- Tolerance values may need tuning for specific workflows

---
## Architectural Expansion: 3D Geometry Primitives & Shapely-style API (September 2025)

### Summary of Change
The codebase has been expanded from a triangle-mesh-centric core to a modular 3D geometry module, inspired by the API and usability of Shapely (for 2D). New core primitives have been scaffolded: `Point3D`, `Line3D`, `Plane`, `Polygon3D`, and `Polyhedron`, each with stubs for boolean operations (union, intersection, difference), spatial predicates (contains, intersects, within, touches), and transformations (translate, rotate, scale). This lays the foundation for a comprehensive, extensible 3D geometry library.

### Rationale
- **API Consistency:** Mimics Shapely's style for ease of use and learning.
- **Extensibility:** New geometry types and algorithms can be added with minimal friction.
- **Performance:** The design allows for future backend acceleration (C/C++/bindings) without changing the API.
- **Testing:** Modular classes enable targeted unit and integration tests.

### New Limitations/Considerations
- Most new methods are currently stubs; full implementations and robust edge-case handling are planned for subsequent sprints.
- Some operations (e.g., boolean on polyhedra) will require advanced algorithms or external libraries for robust performance.

---
Details: 

# Minimal design & priorities (first sprint)

## Goal: get a small, dependency-light, well-tested core you can iterate on and later accelerate with C/C++ or bindings.

# Core principles
## Correct, readable, well-documented NumPy-first implementations for basic ops (no compiled deps).

## Keep API small and composable so I can replace implementations later with faster backends.

## Design for solids (closed triangular meshes) rather than just surfaces.

# MVP features (v0.1)
- Vector3 basic 3D vector utilities.
- Triangle for triangle-specific ops (area, normal, centroid).
- Mesh for triangular meshes with:
    - creation from_vertices_faces
    - AABB bounding box
    - translate / scale
    - surface_area(), centroid(), volume() (signed), is_closed() heuristic
    - simple OBJ loader/saver
- unit test for volume/area on a tetrahedron
- small demo script

Next sprints
- robust mesh validation / repair
- BVH/octree spatial index
- mesh boolean (CSG) via exact predicates or plugin to CGAL / libigl
- topology utilities (edge-manifold checks, non-manifold vertex detection)
- prebuilt binary wheels using cibuildwheel

---

## Performance: Spatial Indexing & Nearest Neighbor

Solidly provides fast 3D spatial queries using an AABB tree and nearest neighbor search utilities.

### Usage Example
```python
from solidly.spatial import AABBTree, compute_aabb, nearest_neighbor
from solidly import Point3D

points = [Point3D(0,0,0), Point3D(1,2,3), Point3D(5,5,5)]
tree = AABBTree(points, compute_aabb)
# Query all points in a region:
query_box = compute_aabb([Point3D(0,0,0), Point3D(2,2,2)])
found = tree.query_aabb(query_box)
# Nearest neighbor:
pt, dist = nearest_neighbor(points, [1,1,1])
```

## Fast Contains Checks via Ray Casting

Solidly provides a fast `ray_intersects` method for `Polyhedron` using ray casting (Möller–Trumbore algorithm). This enables efficient point-in-polyhedron and intersection queries.

### Usage Example
```python
from solidly import Polyhedron
cube = ... # build a Polyhedron
origin = [0.5, 0.5, 0.5]
direction = [1, 0, 0]
inside = cube.ray_intersects(origin, direction)  # True if ray from origin in direction hits the polyhedron
```

## Visualization & Plotting

Solidly provides lightweight 3D plotting for all core primitives using matplotlib (Axes3D):
- `Point3D.plot(ax=None, **kwargs)`
- `Line3D.plot(ax=None, **kwargs)`
- `Polygon3D.plot(ax=None, **kwargs)`
- `Polyhedron.plot(ax=None, **kwargs)`

### Usage Example
```python
from solidly import Point3D, Line3D, Polygon3D, Polyhedron
import matplotlib.pyplot as plt
p = Point3D(1,2,3)
l = Line3D(Point3D(0,0,0), Point3D(1,1,1))
poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
# Plot all on the same axes:
ax = plt.figure().add_subplot(111, projection='3d')
p.plot(ax, color='r')
l.plot(ax, color='g')
poly.plot(ax, color='b')
plt.show()
```

Optional: PyVista/vedo hooks are planned for interactive mesh visualization.

---

## Higher-level Topological Operations

Solidly now provides:
- `convex_hull(points)`: Returns the convex hull of a set of `Point3D` as a `Polygon3D` (planar) or `Polyhedron` (3D).
- `Polygon3D.simplify(tolerance)`: Ramer-Douglas-Peucker simplification for polygons.
- `Mesh.simplify(tolerance)`: (stub) Mesh decimation (planned).
- `Polygon3D.minkowski_sum(other)`, `Mesh.minkowski_sum(other)`: (stub) Minkowski sum for planning/robotics.
- `Polygon3D.offset(distance)`, `Mesh.offset(distance)`: (stub) Offset/buffer surfaces.

### Usage Example
```python
from solidly import Point3D, convex_hull
points = [Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0), Point3D(0.5,0.5,0)]
hull = convex_hull(points)  # Returns Polygon3D
```

## I/O & Interop

Solidly supports mesh import/export and text-based geometry formats:
- `Mesh.to_obj(path)`, `Mesh.from_obj(path)`
- `Mesh.to_stl(path)`, `Mesh.from_stl(path)`
- `Mesh.to_ply(path)`, `Mesh.from_ply(path)`
- `Mesh.to_numpy()`, `Mesh.from_numpy(vertices, faces)`
- `Point3D.to_wkt()`, `Line3D.to_wkt()`, `Polygon3D.to_wkt()`
- Stubs: `Mesh.from_trimesh(mesh)`, `Mesh.from_open3d(mesh)`

### Usage Example
```python
from solidly import Mesh, Point3D, Line3D, Polygon3D
# OBJ
mesh = Mesh.from_obj('input.obj')
mesh.to_stl('output.stl')
verts, faces = mesh.to_numpy()
mesh2 = Mesh.from_numpy(verts, faces)
# WKT-3D
p = Point3D(1,2,3)
print(p.to_wkt())  # POINT Z (1 2 3)
l = Line3D(Point3D(0,0,0), Point3D(1,1,1))
print(l.to_wkt())  # LINESTRING Z (...)
poly = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
print(poly.to_wkt())
```