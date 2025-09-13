SOLIDLY LIBRARY - FULL DOCUMENTATION

============================================================

This document provides a complete reference for the Solidly library,

covering all classes, functions, and features from core.py, plotting.py,

and spatial.py.


FILE: core.py

------------------------------------------------------------

Contains fundamental 3D geometry primitives, polyhedral operations,

mesh data structures, and higher-level topological utilities.


CLASSES & METHODS:

1. Vector3 (x, y, z)

   - to_array() -> ndarray: Returns (3,) numpy array.

   - from_array(arr) -> Vector3: Construct from array.

   - __add__, __sub__, __mul__: Vector arithmetic.

   - dot(other) -> float: Dot product.

   - cross(other) -> Vector3: Cross product.

   - norm() -> float: Vector magnitude.

   - normalize() -> Vector3: Unit vector.

   - rotate(axis, angle): Rotate around axis by angle (rad).

   - scale(factor, origin): Scale about origin.


2. Point3D (x, y, z)

   Boolean Ops: union, intersection, difference

   Predicates: contains, intersects, within, touches

   Transforms: translate, rotate, scale

   - almost_equals(other, tol): Floating-point equality check

   - to_array() -> ndarray

   - to_wkt() -> str: WKT representation

   - plot(ax=None, **kwargs): Matplotlib 3D scatter


3. Line3D (p0, p1)

   Boolean Ops: union, intersection (with Plane or Line3D), difference

   Predicates: contains, intersects, within, touches

   Transforms: translate, rotate, scale

   - is_simple(): Returns True if p0 != p1

   - almost_equals(other, tol)

   - to_wkt() -> str

   - plot(ax=None, **kwargs)


4. Plane (point, normal)

   - contains(point)

   - intersection(other_plane) -> Line3D or self or None

   - translate/rotate/scale

   - intersects(point|line|plane)

   - within(other_plane), touches(other_plane)

   - almost_equals(other, tol)


5. Polygon3D (vertices: Tuple[Point3D])

   Boolean Ops: union, intersection (Sutherland–Hodgman), difference

   Predicates: contains(point), intersects, within, touches

   Transforms: translate, rotate, scale

   - convex_hull(), simplify(tolerance), surface_area()

   - is_simple(), is_convex(), almost_equals()

   - to_wkt(), plot(ax)

   - aabb(): Axis-aligned bounding box

   - minkowski_sum(), offset() (not yet implemented)


6. Polyhedron (faces: Tuple[Polygon3D])

   Boolean Ops: union (convex hull of vertices), intersection (stub), difference (stub)

   Predicates: contains(point), intersects, within, touches

   Transforms: translate, rotate, scale

   - convex_hull(), simplify(tolerance)

   - surface_area(), volume()

   - is_closed(), is_convex(), almost_equals()

   - ray_intersects(origin, direction): Ray casting test

   - plot(ax)


7. Triangle (v0, v1, v2)

   - area(), is_degenerate(), normal(), centroid()


8. Mesh (vertices, faces)

   Factory: from_vertices_faces(vertices, faces)

   - has_degenerate_faces(tol)

   - copy(), bounding_box()

   - translate(offset), scale(factor, origin)

   - triangles(), surface_area(), centroid(), volume()

   - is_closed()

   I/O: to_obj/from_obj, to_stl/from_stl, to_ply/from_ply, to_numpy/from_numpy

   - aabb_tree(), nearest_vertex(query)

   - simplify(), minkowski_sum(), offset() (not yet implemented)


GLOBAL FUNCTIONS:

- convex_hull(points) -> Polygon3D | Polyhedron

- rdp_simplify(points, tolerance) -> List[Point3D]


FILE: plotting.py

------------------------------------------------------------

Matplotlib-based visualization utilities for Solidly geometries.


FUNCTIONS:

- plot_point3d(point, ax=None, **kwargs): Scatter plot of a point.

- plot_line3d(line, ax=None, **kwargs): Plot line segment.

- plot_polygon3d(poly, ax=None, **kwargs): Plot polygon edges.

- plot_polyhedron(polyh, ax=None, facecolor='cyan', ...): Draw polyhedron faces.

- to_pyvista(mesh): (NotImplemented)

- to_vedo(mesh): (NotImplemented)


FILE: spatial.py

------------------------------------------------------------

Provides spatial indexing (AABB tree) and nearest neighbor search.


CLASSES & FUNCTIONS:

1. compute_aabb(points) -> AABB(min, max)

2. AABBNode(aabb, left, right, obj): Node in bounding box tree

   - is_leaf(): Returns True if node contains an object.

3. AABBTree(objects, get_aabb)

   - query_aabb(query_aabb) -> List[objects]

   - nearest(point) -> (object, distance)

4. nearest_neighbor(points, query) -> (nearest_point, distance)
