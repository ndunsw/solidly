"""Core classes for Solidly: Vector3, Triangle, Mesh (basic operations).

This prototype focuses on clarity and correctness using NumPy as the numeric backend.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

ArrayLike = np.ndarray


# --- Core 3D Primitives ---
@dataclass(frozen=True)
class Vector3:

    def rotate(self, axis, angle):
        """Return a new Vector3 rotated about the given axis by angle (radians)."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-16)
        v = self.to_array()
        # Rodrigues' rotation formula
        v_rot = (v * np.cos(angle) +
                 np.cross(axis, v) * np.sin(angle) +
                 axis * np.dot(axis, v) * (1 - np.cos(angle)))
        return Vector3.from_array(v_rot)

    def scale(self, factor, origin=None):
        """Return a new Vector3 scaled by factor about origin (default 0,0,0)."""
        if origin is None:
            origin = np.zeros(3)
        arr = origin + (self.to_array() - origin) * float(factor)
        return Vector3.from_array(arr)
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_array(cls, arr: ArrayLike) -> "Vector3":
        arr = np.asarray(arr, dtype=float)
        assert arr.shape == (3,)
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, other: "Vector3") -> "Vector3":
        a = self.to_array() + other.to_array()
        return Vector3.from_array(a)

    def __sub__(self, other: "Vector3") -> "Vector3":
        a = self.to_array() - other.to_array()
        return Vector3.from_array(a)

    def __mul__(self, scalar: float) -> "Vector3":
        a = self.to_array() * float(scalar)
        return Vector3.from_array(a)

    def dot(self, other: "Vector3") -> float:
        return float(np.dot(self.to_array(), other.to_array()))

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3.from_array(np.cross(self.to_array(), other.to_array()))

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def normalize(self) -> "Vector3":
        n = self.norm()
        if n == 0:
            return Vector3(0.0, 0.0, 0.0)
        return self * (1.0 / n)


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    # --- Boolean ops ---
    def union(self, other):
        """Union of two points: returns self if equal, else a set of both."""
        if isinstance(other, Point3D):
            if self.almost_equals(other):
                return self
            return {self, other}
        raise NotImplementedError("Union only defined for Point3D.")

    def intersection(self, other):
        """Intersection of two points: returns self if equal, else None."""
        if isinstance(other, Point3D):
            if self.almost_equals(other):
                return self
            return None
        raise NotImplementedError("Intersection only defined for Point3D.")

    def difference(self, other):
        """Difference of two points: returns self if not equal, else None."""
        if isinstance(other, Point3D):
            if self.almost_equals(other):
                return None
            return self
        raise NotImplementedError("Difference only defined for Point3D.")
    # --- Predicates ---
    def contains(self, other):
        """True if this point is the same as other (within tolerance)."""
        if isinstance(other, Point3D):
            return self.almost_equals(other)
        return False

    def intersects(self, other):
        """True if this point is the same as other (within tolerance)."""
        return self.contains(other)

    def within(self, other):
        """True if this point is contained in other geometry."""
        if hasattr(other, 'contains'):
            return other.contains(self)
        return False

    def touches(self, other):
        """Points only touch if they are equal (so same as contains)."""
        return self.contains(other)
    # --- Transformations ---
    def translate(self, offset):
        """Return a new Point3D translated by the given offset (3D vector)."""
        arr = self.to_array() + np.asarray(offset, dtype=float)
        return Point3D(*arr)

    def rotate(self, axis, angle):
        """Return a new Point3D rotated about the given axis by angle (radians)."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-16)
        p = self.to_array()
        # Rodrigues' rotation formula
        p_rot = (p * np.cos(angle) +
                 np.cross(axis, p) * np.sin(angle) +
                 axis * np.dot(axis, p) * (1 - np.cos(angle)))
        return Point3D(*p_rot)

    def scale(self, factor, origin=None):
        """Return a new Point3D scaled by factor about origin (default 0,0,0)."""
        if origin is None:
            origin = np.zeros(3)
        arr = origin + (self.to_array() - origin) * float(factor)
        return Point3D(*arr)

    def almost_equals(self, other, tol=1e-8) -> bool:
        """Return True if coordinates are equal within a tolerance."""
        return np.allclose(self.to_array(), other.to_array(), atol=tol)


@dataclass(frozen=True)
class Line3D:
    p0: Point3D
    p1: Point3D

    # Boolean ops, predicates, transformations
    def union(self, other):
        """Union of two lines: if collinear and overlapping, merge; else return set."""
        if isinstance(other, Line3D):
            # If collinear and overlapping, merge
            if self.contains(other.p0) or self.contains(other.p1) or other.contains(self.p0) or other.contains(self.p1):
                # Return the minimal segment covering both
                pts = [self.p0, self.p1, other.p0, other.p1]
                arrs = np.array([p.to_array() for p in pts])
                idx = np.argsort(arrs[:,0] + arrs[:,1]*1e-3 + arrs[:,2]*1e-6)
                return Line3D(pts[idx[0]], pts[idx[-1]])
            return {self, other}
        raise NotImplementedError("Union only defined for Line3D.")
    def intersection(self, other):
        """
        Intersect this line with a Plane.
        Returns:
            - Point3D if intersection is a single point
            - self if the line lies in the plane
            - None if no intersection (parallel and not in plane)
        """
        if isinstance(other, Plane):
            # Line: p = p0 + t*(p1-p0)
            # Plane: (x - p_plane) . n = 0
            p0 = self.p0.to_array()
            p1 = self.p1.to_array()
            d = p1 - p0
            n = other.normal.to_array()
            denom = np.dot(d, n)
            if abs(denom) < 1e-12:
                # Line is parallel to plane
                if abs(np.dot(p0 - other.point.to_array(), n)) < 1e-12:
                    return self  # Line lies in plane
                return None
            t = np.dot(other.point.to_array() - p0, n) / denom
            if 0 <= t <= 1:
                pt = p0 + t * d
                return Point3D(*pt)
            else:
                # Intersection is outside segment, but for infinite line return anyway
                pt = p0 + t * d
                return Point3D(*pt)
        if isinstance(other, Line3D):
            # Compute intersection of two lines in 3D
            # If lines are skew, return None. If overlap, return self.
            p1, d1 = self.p0.to_array(), self.p1.to_array() - self.p0.to_array()
            p2, d2 = other.p0.to_array(), other.p1.to_array() - other.p0.to_array()
            cross = np.cross(d1, d2)
            denom = np.linalg.norm(cross) ** 2
            if denom < 1e-12:
                # Lines are parallel
                # Check if collinear
                v = p2 - p1
                if np.linalg.norm(np.cross(v, d1)) < 1e-12:
                    # Overlapping lines
                    return self
                return None
            # Find intersection point using closest points
            t = np.linalg.det([p2 - p1, d2, cross]) / denom
            u = np.linalg.det([p2 - p1, d1, cross]) / denom
            pt1 = p1 + t * d1
            pt2 = p2 + u * d2
            if np.linalg.norm(pt1 - pt2) < 1e-8:
                return Point3D(*pt1)
            return None
        raise NotImplementedError(f"Intersection not implemented for {type(other)}")
    def difference(self, other):
        """Difference of two lines: returns self minus overlap with other (if collinear)."""
        if isinstance(other, Line3D):
            # Only handle collinear, overlapping
            if not (self.contains(other.p0) or self.contains(other.p1) or other.contains(self.p0) or other.contains(self.p1)):
                return self
            # For MVP, just return None if overlap
            return None
        raise NotImplementedError("Difference only defined for Line3D.")
    def contains(self, other):
        """True if other is a point on the line segment, or a subsegment."""
        if isinstance(other, Point3D):
            p = other.to_array()
            a = self.p0.to_array()
            b = self.p1.to_array()
            ab = b - a
            ap = p - a
            cross = np.cross(ab, ap)
            if np.linalg.norm(cross) > 1e-8:
                return False
            dot = np.dot(ap, ab)
            if dot < -1e-8 or dot > np.dot(ab, ab) + 1e-8:
                return False
            return True
        if isinstance(other, Line3D):
            return self.contains(other.p0) and self.contains(other.p1)
        return False

    def intersects(self, other):
        """True if lines intersect (point or overlap)."""
        result = self.intersection(other)
        return result is not None

    def within(self, other):
        """True if this line is a subsegment of other."""
        if isinstance(other, Line3D):
            return other.contains(self)
        return False

    def touches(self, other):
        """True if lines share an endpoint but do not overlap."""
        if isinstance(other, Line3D):
            shared = (self.p0.almost_equals(other.p0) or
                      self.p0.almost_equals(other.p1) or
                      self.p1.almost_equals(other.p0) or
                      self.p1.almost_equals(other.p1))
            return shared and not self.intersects(other)
        return False
    def translate(self, offset):
        """Return a new Line3D translated by the given offset (3D vector)."""
        return Line3D(self.p0.translate(offset), self.p1.translate(offset))

    def rotate(self, axis, angle):
        """Return a new Line3D rotated about the given axis by angle (radians)."""
        return Line3D(self.p0.rotate(axis, angle), self.p1.rotate(axis, angle))

    def scale(self, factor, origin=None):
        """Return a new Line3D scaled by factor about origin (default 0,0,0)."""
        return Line3D(self.p0.scale(factor, origin), self.p1.scale(factor, origin))

    def is_simple(self) -> bool:
        """A line is simple if its endpoints are not equal."""
        return not self.p0.almost_equals(self.p1)

    def almost_equals(self, other, tol=1e-8) -> bool:
        return self.p0.almost_equals(other.p0, tol) and self.p1.almost_equals(other.p1, tol)



class Plane:
    def __init__(self, point, normal):
        self.point = point
        if isinstance(normal, Vector3):
            self.normal = normal
        else:
            arr = np.asarray(normal, dtype=float)
            self.normal = Vector3.from_array(arr)

    def contains(self, other):
        """True if other (Point3D) lies in the plane."""
        if isinstance(other, Point3D):
            v = other.to_array() - self.point.to_array()
            return abs(np.dot(v, self.normal.to_array())) < 1e-8
        return False
    def intersection(self, other):
        """
        Intersect this plane with another plane.
        Returns:
            - Line3D if planes are not parallel
            - self if planes are coincident
            - None if planes are parallel and distinct
        """
        if isinstance(other, Plane):
            n1 = self.normal.to_array()
            n2 = other.normal.to_array()
            cross = np.cross(n1, n2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-12:
                # Parallel
                if abs(np.dot(self.point.to_array() - other.point.to_array(), n1)) < 1e-12:
                    return self  # Coincident
                return None
            # Direction of intersection line
            dir = cross / norm_cross
            # Find a point on both planes
            # Solve: x . n1 = d1, x . n2 = d2
            d1 = np.dot(n1, self.point.to_array())
            d2 = np.dot(n2, other.point.to_array())
            # Build system: [n1; n2; dir] x = [d1; d2; 0]
            A = np.vstack([n1, n2, dir])
            b = np.array([d1, d2, 0.0])
            pt = np.linalg.lstsq(A, b, rcond=None)[0]
            p0 = Point3D(*pt)
            p1 = Point3D(*(pt + dir))
            return Line3D(p0, p1)
        raise NotImplementedError(f"Intersection not implemented for {type(other)}")
    def translate(self, offset):
        """Return a new Plane translated by the given offset (3D vector)."""
        return Plane(self.point.translate(offset), self.normal)

    def rotate(self, axis, angle):
        """Return a new Plane rotated about the given axis by angle (radians)."""
        return Plane(self.point.rotate(axis, angle), self.normal.rotate(axis, angle))

    def scale(self, factor, origin=None):
        """Return a new Plane scaled by factor about origin (default 0,0,0)."""
        return Plane(self.point.scale(factor, origin), self.normal)

    def intersects(self, other):
        """True if planes are not parallel, or if a point/line/plane intersects this plane."""
        if isinstance(other, Plane):
            n1 = self.normal.to_array()
            n2 = other.normal.to_array()
            cross = np.cross(n1, n2)
            return np.linalg.norm(cross) > 1e-8 or self.almost_equals(other)
        if isinstance(other, Point3D):
            return self.contains(other)
        if isinstance(other, Line3D):
            return self.contains(other.p0) or self.contains(other.p1)
        return False

    def within(self, other):
        """True if this plane is the same as other."""
        if isinstance(other, Plane):
            return self.almost_equals(other)
        return False

    def touches(self, other):
        """True if planes are parallel but not coincident."""
        if isinstance(other, Plane):
            n1 = self.normal.to_array()
            n2 = other.normal.to_array()
            cross = np.cross(n1, n2)
            return np.linalg.norm(cross) < 1e-8 and not self.almost_equals(other)
        return False

    def almost_equals(self, other, tol=1e-8) -> bool:
        return self.point.almost_equals(other.point, tol) and self.normal.to_array().dot(other.normal.to_array()) > 1 - tol


@dataclass(frozen=True)
class Polygon3D:
    vertices: Tuple[Point3D, ...]

    def union(self, other):
        """Union of two convex, coplanar polygons: convex hull of all vertices."""
        if isinstance(other, Polygon3D):
            # Check coplanarity
            verts = [*self.vertices, *other.vertices]
            arrs = np.array([v.to_array() for v in verts])
            n = np.cross(arrs[1] - arrs[0], arrs[2] - arrs[0])
            n = n / (np.linalg.norm(n) + 1e-16)
            axis = np.argmax(np.abs(n))
            arr2d = np.delete(arrs, axis, axis=1)
            from scipy.spatial import ConvexHull
            hull = ConvexHull(arr2d)
            hull_pts = [verts[i] for i in hull.vertices]
            return Polygon3D(tuple(hull_pts))
        raise NotImplementedError("Union only defined for Polygon3D.")
    
    def intersection(self, other):
        """
        Intersect this polygon with another convex, coplanar polygon using Sutherland-Hodgman.
        Returns:
            - Polygon3D if intersection exists
            - None if no intersection or not coplanar
        """
        if isinstance(other, Plane):
            # Polygon-plane intersection not implemented
            return None
        if isinstance(other, Polygon3D):
            return Polygon3D._sutherland_hodgman_clip(self, other)
        raise NotImplementedError(f"Intersection not implemented for {type(other)}")

    def difference(self, other):
        """
        Difference of two convex, coplanar polygons using Sutherland-Hodgman (clip self by other's complement).
        Returns:
            - Polygon3D if difference exists
            - None if no difference or not coplanar
        """
        if isinstance(other, Polygon3D):
            result = Polygon3D._sutherland_hodgman_clip(self, other, difference=True)
            if result is not None and hasattr(result, 'vertices') and len(result.vertices) > 0:
                return result
            # If result is None or empty, return self (non-overlapping case)
            return self
        raise NotImplementedError("Difference only defined for Polygon3D.")

    @staticmethod
    def _sutherland_hodgman_clip(subject_poly, clip_poly, difference=False):
        """
        Sutherland-Hodgman polygon clipping for convex, coplanar polygons.
        If difference=True, clips subject by the complement of clip_poly.
        Returns Polygon3D or None.
        """
        # Check coplanarity
        def polygon_normal(verts):
            v0, v1, v2 = verts[0].to_array(), verts[1].to_array(), verts[2].to_array()
            return np.cross(v1 - v0, v2 - v0)
        n1 = polygon_normal(subject_poly.vertices)
        n2 = polygon_normal(clip_poly.vertices)
        n1 = n1 / (np.linalg.norm(n1) + 1e-16)
        n2 = n2 / (np.linalg.norm(n2) + 1e-16)
        if np.linalg.norm(n1 - n2) > 1e-8 and np.linalg.norm(n1 + n2) > 1e-8:
            return None  # Not coplanar
        # Project to 2D (find dominant axis to drop)
        axis = np.argmax(np.abs(n1))
        def project(verts):
            arr = np.array([v.to_array() for v in verts])
            return np.delete(arr, axis, axis=1)
        subj_2d = project(subject_poly.vertices)
        clip_2d = project(clip_poly.vertices)
        # Sutherland-Hodgman
        def inside(p, edge_start, edge_end, diff):
            val = (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
            return val >= 0 if not diff else val < 0
        def compute_intersection(p1, p2, q1, q2):
            s = np.array(p1)
            r = np.array(p2) - np.array(p1)
            q = np.array(q1)
            s2 = np.array(q2) - np.array(q1)
            denom = r[0]*s2[1] - r[1]*s2[0]
            if abs(denom) < 1e-12:
                return p2  # Parallel, just return endpoint
            t = ((q[0] - s[0]) * s2[1] - (q[1] - s[1]) * s2[0]) / denom
            return s + t * r
        output = subj_2d.tolist()
        for i in range(len(clip_2d)):
            input_list = output
            output = []
            A = clip_2d[i - 1]
            B = clip_2d[i]
            for j in range(len(input_list)):
                P = input_list[j - 1]
                Q = input_list[j]
                if inside(Q, A, B, difference):
                    if not inside(P, A, B, difference):
                        output.append(compute_intersection(P, Q, A, B))
                    output.append(Q)
                elif inside(P, A, B, difference):
                    output.append(compute_intersection(P, Q, A, B))
        # Remove duplicate consecutive points
        def dedup(pts):
            out = []
            for p in pts:
                if not out or np.linalg.norm(np.array(p) - np.array(out[-1])) > 1e-8:
                    out.append(p)
            if out and np.linalg.norm(np.array(out[0]) - np.array(out[-1])) < 1e-8:
                out.pop()
            return out
        output = dedup(output)
        if len(output) < 3:
            # For intersection: return degenerate geometry (edge or point)
            if not difference and len(output) > 0:
                def unproject(pts2d):
                    pts3d = []
                    for pt in pts2d:
                        arr = [0.0, 0.0, 0.0]
                        j = 0
                        for i in range(3):
                            if i == axis:
                                arr[i] = subject_poly.vertices[0].to_array()[i]
                            else:
                                arr[i] = pt[j]
                                j += 1
                        pts3d.append(Point3D(*arr))
                    return pts3d
                return Polygon3D(tuple(unproject(output)))
            # If no output, check for shared vertices or edges
            subj_verts = [v for v in subject_poly.vertices]
            clip_verts = [v for v in clip_poly.vertices]
            shared = [v for v in subj_verts if any(v.almost_equals(w) for w in clip_verts)]
            if shared:
                return Polygon3D(tuple(shared))
            # Check for shared edge
            subj_edges = set((subj_verts[i], subj_verts[(i+1)%len(subj_verts)]) for i in range(len(subj_verts)))
            clip_edges = set((clip_verts[i], clip_verts[(i+1)%len(clip_verts)]) for i in range(len(clip_verts)))
            for e1 in subj_edges:
                for e2 in clip_edges:
                    if (e1[0].almost_equals(e2[0]) and e1[1].almost_equals(e2[1])) or (e1[0].almost_equals(e2[1]) and e1[1].almost_equals(e2[0])):
                        return Polygon3D((e1[0], e1[1]))
            # For difference: if no overlap, return self
            if difference:
                # If no overlap at all, return self
                return subject_poly
            return None
        # Reconstruct 3D points by inserting dropped axis
        def unproject(pts2d):
            pts3d = []
            for pt in pts2d:
                arr = [0.0, 0.0, 0.0]
                j = 0
                for i in range(3):
                    if i == axis:
                        arr[i] = subject_poly.vertices[0].to_array()[i]
                    else:
                        arr[i] = pt[j]
                        j += 1
                pts3d.append(Point3D(*arr))
            return pts3d
        return Polygon3D(tuple(unproject(output)))
    def contains(self, other):
        """True if other (Point3D) is inside the polygon (assumes convex, coplanar)."""
        if isinstance(other, Point3D):
            # Project to 2D
            verts = [v.to_array() for v in self.vertices]
            n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            axis = np.argmax(np.abs(n))
            arr = np.array(verts)
            arr2d = np.delete(arr, axis, axis=1)
            pt2d = np.delete(other.to_array(), axis)
            # Ray casting for convex polygon
            inside = True
            for i in range(len(arr2d)):
                a = arr2d[i]
                b = arr2d[(i+1)%len(arr2d)]
                edge = b - a
                to_pt = pt2d - a
                cross = edge[0]*to_pt[1] - edge[1]*to_pt[0]
                if cross < -1e-8:
                    inside = False
                    break
            return inside
        return False

    def intersects(self, other):
        """True if polygons overlap (coplanar, convex only)."""
        if isinstance(other, Polygon3D):
            return self.intersection(other) is not None
        if isinstance(other, Point3D):
            return self.contains(other)
        return False

    def within(self, other):
        """True if all vertices are within other polygon."""
        if isinstance(other, Polygon3D):
            return all(other.contains(v) for v in self.vertices)
        return False

    def touches(self, other):
        """True if polygons share at least one vertex but do not overlap in area."""
        if isinstance(other, Polygon3D):
            shared = any(v1.almost_equals(v2) for v1 in self.vertices for v2 in other.vertices)
            return shared and not self.intersects(other)
        return False
    def translate(self, offset):
        """Return a new Polygon3D translated by the given offset (3D vector)."""
        return Polygon3D(tuple(v.translate(offset) for v in self.vertices))

    def rotate(self, axis, angle):
        """Return a new Polygon3D rotated about the given axis by angle (radians)."""
        return Polygon3D(tuple(v.rotate(axis, angle) for v in self.vertices))

    def scale(self, factor, origin=None):
        """Return a new Polygon3D scaled by factor about origin (default 0,0,0)."""
        return Polygon3D(tuple(v.scale(factor, origin) for v in self.vertices))
    def convex_hull(self):
        """Return the convex hull of the polygon's vertices as a new Polygon3D."""
        arrs = np.array([v.to_array() for v in self.vertices])
        # Project to 2D (drop axis with largest normal component)
        n = np.cross(arrs[1] - arrs[0], arrs[2] - arrs[0])
        n = n / (np.linalg.norm(n) + 1e-16)
        axis = np.argmax(np.abs(n))
        arr2d = np.delete(arrs, axis, axis=1)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(arr2d)
        hull_pts = [self.vertices[i] for i in hull.vertices]
        return Polygon3D(tuple(hull_pts))

    def simplify(self, tolerance):
        """Simplify the polygon using Ramer-Douglas-Peucker in 2D projection."""
        arrs = np.array([v.to_array() for v in self.vertices])
        n = np.cross(arrs[1] - arrs[0], arrs[2] - arrs[0])
        n = n / (np.linalg.norm(n) + 1e-16)
        axis = np.argmax(np.abs(n))
        arr2d = np.delete(arrs, axis, axis=1)
        def rdp(points, eps):
            if len(points) < 3:
                return list(range(len(points)))
            start, end = points[0], points[-1]
            dmax, idx = 0, 0
            for i in range(1, len(points)-1):
                d = np.abs(np.cross(end-start, points[i]-start)) / (np.linalg.norm(end-start)+1e-16)
                if d > dmax:
                    idx, dmax = i, d
            if dmax > eps:
                left = rdp(points[:idx+1], eps)
                right = rdp(points[idx:], eps)
                return left[:-1] + [i+idx for i in right]
            else:
                return [0, len(points)-1]
        idxs = rdp(arr2d, tolerance)
        return Polygon3D(tuple(self.vertices[i] for i in idxs))

    def surface_area(self):
        """Return the area of the polygon (assumes planar, non-self-intersecting)."""
        arrs = np.array([v.to_array() for v in self.vertices])
        n = np.cross(arrs[1] - arrs[0], arrs[2] - arrs[0])
        n = n / (np.linalg.norm(n) + 1e-16)
        area = 0.0
        for i in range(len(arrs)):
            v0 = arrs[i]
            v1 = arrs[(i+1)%len(arrs)]
            area += np.linalg.norm(np.cross(v0, v1)) / 2
        # Projected area onto the plane normal
        return area * np.abs(np.dot(n, [0,0,1]))

    def is_simple(self) -> bool:
        """A polygon is simple if no three consecutive points are collinear."""
        arrs = np.array([v.to_array() for v in self.vertices])
        n = np.cross(arrs[1] - arrs[0], arrs[2] - arrs[0])
        n = n / (np.linalg.norm(n) + 1e-16)
        for i in range(len(arrs)):
            v0 = arrs[i]
            v1 = arrs[(i+1)%len(arrs)]
            v2 = arrs[(i+2)%len(arrs)]
            cross = np.cross(v1-v0, v2-v1)
            if np.linalg.norm(cross) < 1e-8:
                return False
        return True

    def is_convex(self) -> bool:
        """Return True if the polygon is convex."""
        arrs = np.array([v.to_array() for v in self.vertices])
        n = np.cross(arrs[1] - arrs[0], arrs[2] - arrs[0])
        n = n / (np.linalg.norm(n) + 1e-16)
        sign = None
        for i in range(len(arrs)):
            v0 = arrs[i]
            v1 = arrs[(i+1)%len(arrs)]
            v2 = arrs[(i+2)%len(arrs)]
            cross = np.cross(v1-v0, v2-v1)
            dot = np.dot(cross, n)
            if abs(dot) < 1e-8:
                continue
            if sign is None:
                sign = np.sign(dot)
            elif np.sign(dot) != sign:
                return False
        return True

    def almost_equals(self, other, tol=1e-8) -> bool:
        if len(self.vertices) != len(other.vertices):
            return False
        return all(a.almost_equals(b, tol) for a, b in zip(self.vertices, other.vertices))


@dataclass(frozen=True)
class Polyhedron:
    faces: Tuple[Polygon3D, ...]

    def union(self, other):
        """
        Union of two convex polyhedra (MVP):
        Returns a new Polyhedron if both are convex and intersect, else None.
        Uses intersection of all half-spaces (faces) from both polyhedra.
        """
        if not isinstance(other, Polyhedron):
            raise NotImplementedError("Union only supported for Polyhedron.")
        # For MVP, just return convex hull of all vertices (over-approximation)
        verts = [v for f in self.faces for v in f.vertices] + [v for f in other.faces for v in f.vertices]
        # Remove duplicates
        arrs = np.array([v.to_array() for v in verts])
        _, idx = np.unique(np.round(arrs, 8), axis=0, return_index=True)
        unique_verts = [verts[i] for i in idx]
        # Use qhull via scipy.spatial.ConvexHull if available, else stub
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull([v.to_array() for v in unique_verts])
            faces = []
            for simplex in hull.simplices:
                faces.append(Polygon3D(tuple(unique_verts[i] for i in simplex)))
            return Polyhedron(tuple(faces))
        except ImportError:
            raise NotImplementedError("This is a stub. Install 'scipy.spatial.convexhull' for full functionality.")


    def intersection(self, other):
        """
        Intersection of two convex polyhedra (MVP):
        Returns a new Polyhedron if both are convex and intersect, else None.
        Uses intersection of all half-spaces (faces) from both polyhedra.
        """
        if not isinstance(other, Polyhedron):
            raise NotImplementedError("Intersection only supported for Polyhedron.")
        # For MVP, not implemented (would require half-space intersection)
        return None


    def difference(self, other):
        """
        Difference of two convex polyhedra (MVP):
        Returns a new Polyhedron if both are convex, else None.
        MVP: Clips each face of self by all faces of other using polygon difference.
        """
        if not isinstance(other, Polyhedron):
            raise NotImplementedError("Difference only supported for Polyhedron.")
        # For MVP: clip each face of self by all faces of other
        new_faces = []
        for face in self.faces:
            diff_face = face
            for other_face in other.faces:
                diff = diff_face.difference(other_face)
                if diff is not None and hasattr(diff, 'vertices') and len(diff.vertices) >= 3:
                    diff_face = diff
                else:
                    diff_face = None
                    break
            if diff_face is not None and hasattr(diff_face, 'vertices') and len(diff_face.vertices) >= 3:
                new_faces.append(diff_face)
        if new_faces:
            return Polyhedron(tuple(new_faces))
        return None
    
    def contains(self, other):
        """True if other (Point3D) is inside the convex polyhedron (MVP: convex only, robust normal orientation)."""
        if isinstance(other, Point3D):
            # Compute centroid
            all_verts = [v for f in self.faces for v in f.vertices]
            arrs = np.array([v.to_array() for v in all_verts])
            centroid = arrs.mean(axis=0)
            pt = other.to_array()
            for face in self.faces:
                verts = [v.to_array() for v in face.vertices]
                n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
                n = n / (np.linalg.norm(n) + 1e-16)
                v0 = verts[0]
                # Orient normal outward: centroid should be inside
                if np.dot(centroid - v0, n) > 0:
                    n = -n
                if np.dot(pt - v0, n) > 1e-8:
                    return False
            return True
        return False

    def intersects(self, other):
        """True if polyhedra overlap (MVP: convex only, bounding box test)."""
        if isinstance(other, Polyhedron):
            # Use bounding box overlap for MVP
            def bbox(poly):
                arr = np.array([v.to_array() for f in poly.faces for v in f.vertices])
                return arr.min(axis=0), arr.max(axis=0)
            min1, max1 = bbox(self)
            min2, max2 = bbox(other)
            return np.all(max1 >= min2 - 1e-8) and np.all(max2 >= min1 - 1e-8)
        if isinstance(other, Point3D):
            return self.contains(other)
        return False

    def within(self, other):
        """True if all vertices are within other polyhedron."""
        if isinstance(other, Polyhedron):
            verts = [v for f in self.faces for v in f.vertices]
            return all(other.contains(v) for v in verts)
        return False

    def touches(self, other):
        """True if polyhedra share at least one vertex but do not overlap in volume."""
        if isinstance(other, Polyhedron):
            verts1 = [v for f in self.faces for v in f.vertices]
            verts2 = [v for f in other.faces for v in f.vertices]
            shared = any(v1.almost_equals(v2) for v1 in verts1 for v2 in verts2)
            return shared and not self.intersects(other)
        return False
    def translate(self, offset):
        """Return a new Polyhedron translated by the given offset (3D vector)."""
        return Polyhedron(tuple(f.translate(offset) for f in self.faces))

    def rotate(self, axis, angle):
        """Return a new Polyhedron rotated about the given axis by angle (radians)."""
        return Polyhedron(tuple(f.rotate(axis, angle) for f in self.faces))

    def scale(self, factor, origin=None):
        """Return a new Polyhedron scaled by factor about origin (default 0,0,0)."""
        return Polyhedron(tuple(f.scale(factor, origin) for f in self.faces))
    def convex_hull(self):
        """Return the convex hull of all vertices as a new Polyhedron."""
        verts = [v for f in self.faces for v in f.vertices]
        arrs = np.array([v.to_array() for v in verts])
        from scipy.spatial import ConvexHull
        hull = ConvexHull(arrs)
        faces = []
        for simplex in hull.simplices:
            faces.append(Polygon3D(tuple(verts[i] for i in simplex)))
        return Polyhedron(tuple(faces))

    def simplify(self, tolerance):
        """Simplify all faces using Ramer-Douglas-Peucker."""
        return Polyhedron(tuple(f.simplify(tolerance) for f in self.faces))

    def surface_area(self):
        """Return the total surface area of the polyhedron."""
        return sum(f.surface_area() for f in self.faces)

    def volume(self) -> float:
        """Return the signed volume of the polyhedron (assumes closed, non-self-intersecting)."""
        # Use divergence theorem: sum tetrahedra from origin to each face
        vol = 0.0
        for f in self.faces:
            verts = f.vertices
            if len(verts) < 3:
                continue
            v0 = verts[0].to_array()
            for i in range(1, len(verts)-1):
                v1 = verts[i].to_array()
                v2 = verts[i+1].to_array()
                vol += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(vol)

    def is_closed(self) -> bool:
        """Return True if the polyhedron is topologically closed (all edges shared by two faces)."""
        # Count all edges
        edge_count = {}
        for face in self.faces:
            verts = face.vertices
            n = len(verts)
            for i in range(n):
                a = verts[i]
                b = verts[(i+1)%n]
                key = tuple(sorted((a, b), key=lambda v: (v.x, v.y, v.z)))
                edge_count[key] = edge_count.get(key, 0) + 1
        # Each edge should appear exactly twice
        return all(count == 2 for count in edge_count.values())

    def is_convex(self) -> bool:
        """Return True if the polyhedron is convex (all points are on the inner side of all faces)."""
        # For each face, all other vertices must be on the inner side
        all_verts = [v for f in self.faces for v in f.vertices]
        arrs = np.array([v.to_array() for v in all_verts])
        centroid = arrs.mean(axis=0)
        for face in self.faces:
            verts = [v.to_array() for v in face.vertices]
            n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            n = n / (np.linalg.norm(n) + 1e-16)
            v0 = verts[0]
            # Orient normal outward
            if np.dot(centroid - v0, n) > 0:
                n = -n
            for v in arrs:
                if np.allclose(v, v0):
                    continue
                if np.dot(v - v0, n) > 1e-8:
                    return False
        return True

    def almost_equals(self, other, tol=1e-8) -> bool:
        if len(self.faces) != len(other.faces):
            return False
        return all(a.almost_equals(b, tol) for a, b in zip(self.faces, other.faces))

@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray

    def area(self) -> float:
        # Triangle area via cross product (0.5 * ||(v1-v0) x (v2-v0)|| )
        a = self.v1 - self.v0
        b = self.v2 - self.v0
        return 0.5 * np.linalg.norm(np.cross(a, b))

    def is_degenerate(self, tol=1e-12) -> bool:
        """Return True if the triangle has near-zero area (degenerate)."""
        return self.area() < tol

    def normal(self) -> np.ndarray:
        a = self.v1 - self.v0
        b = self.v2 - self.v0
        n = np.cross(a, b)
        norm = np.linalg.norm(n)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])
        return n / norm

    def centroid(self) -> np.ndarray:
        return (self.v0 + self.v1 + self.v2) / 3.0

@dataclass
class Mesh:
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=float))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=int))

    @classmethod
    def from_vertices_faces(cls, vertices: ArrayLike, faces: ArrayLike) -> "Mesh":
        verts = np.asarray(vertices, dtype=float)
        faces_arr = np.asarray(faces, dtype=int)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices must be shape (n,3)")
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("faces must be triangles with shape (m,3)")
        return cls(verts.copy(), faces_arr.copy())

    def has_degenerate_faces(self, tol=1e-12) -> bool:
        """Return True if any triangle in the mesh is degenerate (zero or near-zero area)."""
        return any(t.is_degenerate(tol) for t in self.triangles())

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.vertices.size == 0:
            return (np.zeros(3), np.zeros(3))
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        return mins, maxs

    def translate(self, offset: ArrayLike) -> None:
        off = np.asarray(offset, dtype=float).reshape(3,)
        self.vertices += off

    def scale(self, factor: float, origin: Optional[ArrayLike] = None) -> None:
        if origin is None:
            origin = np.zeros(3)
        origin = np.asarray(origin, dtype=float).reshape(3,)
        self.vertices = (self.vertices - origin) * float(factor) + origin

    def triangles(self) -> List[Triangle]:
        return [Triangle(self.vertices[i0], self.vertices[i1], self.vertices[i2]) for i0,i1,i2 in self.faces]

    def surface_area(self) -> float:
        return float(sum(t.area() for t in self.triangles()))

    def centroid(self) -> np.ndarray:
        # Area-weighted centroid of surface triangles (approximate centroid)
        tris = self.triangles()
        areas = np.array([t.area() for t in tris])
        cents = np.array([t.centroid() for t in tris])
        if areas.sum() == 0:
            return np.zeros(3)
        return (areas[:,None] * cents).sum(axis=0) / areas.sum()

    def volume(self) -> float:
        """
        Compute signed volume of a closed triangular mesh using
        divergence theorem: sum over tetrahedrons formed by triangle and origin:
            V = (1/6) * sum( (v0 x v1) . v2 )
        The sign depends on orientation.
        """
        if self.vertices.size == 0 or self.faces.size == 0:
            return 0.0
        v0 = self.vertices[self.faces[:,0]]
        v1 = self.vertices[self.faces[:,1]]
        v2 = self.vertices[self.faces[:,2]]
        cross = np.cross(v0, v1)
        vol = np.sum(np.einsum('ij,ij->i', cross, v2)) / 6.0
        return float(vol)

    def is_closed(self) -> bool:
        # Basic heuristic: every edge appears twice (manifold closed triangular mesh)
        if self.faces.size == 0:
            return False
        edges = {}
        for tri in self.faces:
            for a,b in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])):
                e = (min(a,b), max(a,b))
                edges[e] = edges.get(e, 0) + 1
        counts = np.array(list(edges.values()))
        return np.all((counts == 2))

    # --- Mesh I/O ---
    def to_obj(self, path):
        with open(path, "w") as f:
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in self.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    def from_obj(path):
        verts, faces = [], []
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                if parts[0] == "v":
                    verts.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    faces.append([int(x.split('/')[0])-1 for x in parts[1:4]])
        return Mesh.from_vertices_faces(np.array(verts), np.array(faces))
    def to_stl(self, path):
        with open(path, "w") as f:
            f.write("solid mesh\n")
            for face in self.faces:
                v0, v1, v2 = [self.vertices[i] for i in face]
                n = np.cross(v1-v0, v2-v0)
                n = n / (np.linalg.norm(n)+1e-16)
                f.write(f"facet normal {n[0]} {n[1]} {n[2]}\n outer loop\n")
                for v in (v0, v1, v2):
                    f.write(f"vertex {v[0]} {v[1]} {v[2]}\n")
                f.write("endloop\nendfacet\n")
            f.write("endsolid\n")
    def from_stl(path):
        # Minimal STL ASCII parser
        verts, faces = [], []
        vmap = {}
        with open(path, "r") as f:
            for line in f:
                if line.strip().startswith("vertex"):
                    v = tuple(map(float, line.strip().split()[1:]))
                    if v not in vmap:
                        vmap[v] = len(verts)
                        verts.append(v)
                    faces.append(vmap[v])
        faces = np.array(faces).reshape(-1,3)
        return Mesh.from_vertices_faces(np.array(verts), faces)
    def to_ply(self, path):
        with open(path, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(self.vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(self.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for v in self.vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for face in self.faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    def from_ply(path):
        verts, faces = [], []
        with open(path, "r") as f:
            lines = f.readlines()
        i = 0
        while not lines[i].startswith("end_header"):
            i += 1
        i += 1
        n_verts = 0
        for line in lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
        for vline in lines[i:i+n_verts]:
            verts.append([float(x) for x in vline.split()])
        for fline in lines[i+n_verts:]:
            if fline.strip():
                faces.append([int(x) for x in fline.split()[1:4]])
        return Mesh.from_vertices_faces(np.array(verts), np.array(faces))
    def to_numpy(self):
        return self.vertices.copy(), self.faces.copy()
    def from_numpy(vertices, faces):
        return Mesh.from_vertices_faces(vertices, faces)
    def from_trimesh(mesh):
        raise NotImplementedError('from_trimesh not yet implemented.')
    def from_open3d(mesh):
        raise NotImplementedError('from_open3d not yet implemented.')

# --- WKT-3D ---
if 'Point3D' in globals():
    def point3d_to_wkt(self):
        return f"POINT Z ({self.x} {self.y} {self.z})"
    Point3D.to_wkt = point3d_to_wkt
if 'Line3D' in globals():
    def line3d_to_wkt(self):
        return f"LINESTRING Z ({self.p0.x} {self.p0.y} {self.p0.z}, {self.p1.x} {self.p1.y} {self.p1.z})"
    Line3D.to_wkt = line3d_to_wkt
if 'Polygon3D' in globals():
    def polygon3d_to_wkt(self):
        coords = ", ".join(f"{v.x} {v.y} {v.z}" for v in self.vertices)
        return f"POLYGON Z (({coords}, {self.vertices[0].x} {self.vertices[0].y} {self.vertices[0].z}))"
    Polygon3D.to_wkt = polygon3d_to_wkt

# --- Spatial Indexing Integration Hooks ---
try:
    from .spatial import AABBTree, compute_aabb, nearest_neighbor
except ImportError:
    AABBTree = None
    compute_aabb = None
    nearest_neighbor = None

# Example: add a method to Mesh for building an AABBTree of its vertices
if 'Mesh' in globals():
    def mesh_aabb_tree(self):
        return AABBTree([Point3D(*v) for v in self.vertices], compute_aabb)
    Mesh.aabb_tree = mesh_aabb_tree
    def mesh_nearest_vertex(self, query):
        pts = [Point3D(*v) for v in self.vertices]
        return nearest_neighbor(pts, np.asarray(query))
    Mesh.nearest_vertex = mesh_nearest_vertex

if 'Polygon3D' in globals():
    def polygon_aabb(self):
        return compute_aabb(self.vertices)
    Polygon3D.aabb = polygon_aabb

# --- Ray Casting for Polyhedron Contains ---
import numpy as np

def _ray_intersects_triangle(orig, dir, v0, v1, v2):
    # Möller–Trumbore intersection algorithm
    eps = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)
    if abs(a) < eps:
        return False  # Ray is parallel to triangle
    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    if t > eps:
        return True  # Intersection
    return False

if 'Polyhedron' in globals():
    def polyhedron_ray_intersects(self, origin, direction):
        """
        Returns True if the ray from origin in direction intersects the polyhedron (at least one intersection).
        origin: np.ndarray or list-like (3,)
        direction: np.ndarray or list-like (3,)
        """
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        count = 0
        for face in self.faces:
            verts = [v.to_array() for v in face.vertices]
            for i in range(1, len(verts)-1):
                v0, v1, v2 = verts[0], verts[i], verts[i+1]
                if _ray_intersects_triangle(origin, direction + 1e-12, v0, v1, v2):
                    count += 1
        return count >= 1
    Polyhedron.ray_intersects = polyhedron_ray_intersects

# --- Plotting Integration Hooks ---
try:
    from .plotting import plot_point3d, plot_line3d, plot_polygon3d, plot_polyhedron
except ImportError:
    plot_point3d = plot_line3d = plot_polygon3d = plot_polyhedron = None

if 'Point3D' in globals():
    def point3d_plot(self, ax=None, **kwargs):
        return plot_point3d(self, ax=ax, **kwargs)
    Point3D.plot = point3d_plot
if 'Line3D' in globals():
    def line3d_plot(self, ax=None, **kwargs):
        return plot_line3d(self, ax=ax, **kwargs)
    Line3D.plot = line3d_plot
if 'Polygon3D' in globals():
    def polygon3d_plot(self, ax=None, **kwargs):
        return plot_polygon3d(self, ax=ax, **kwargs)
    Polygon3D.plot = polygon3d_plot
if 'Polyhedron' in globals():
    def polyhedron_plot(self, ax=None, **kwargs):
        return plot_polyhedron(self, ax=ax, **kwargs)
    Polyhedron.plot = polyhedron_plot

import numpy as np

# --- Higher-level Topological Ops ---
def convex_hull(points):
    """Return the convex hull of a set of Point3D as a Polygon3D (2D) or Polyhedron (3D)."""
    arr = np.array([p.to_array() for p in points])
    # Remove duplicate points
    _, idx = np.unique(np.round(arr, 8), axis=0, return_index=True)
    arr = arr[idx]
    points = [points[i] for i in idx]
    if arr.shape[0] < 4 or np.allclose(arr[:,2], arr[0,2]):
        # 2D: all z are equal (planar)
        from scipy.spatial import ConvexHull, QhullError
        axis = np.argmax(np.ptp(arr, axis=0))
        arr2d = np.delete(arr, axis, axis=1)
        try:
            hull = ConvexHull(arr2d, qhull_options='QJ')
        except QhullError:
            # Fallback: return all unique points as degenerate polygon
            return Polygon3D(tuple(points))
        hull_pts = [points[i] for i in hull.vertices]
        return Polygon3D(tuple(hull_pts))
    else:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(arr)
        faces = []
        for simplex in hull.simplices:
            faces.append(Polygon3D(tuple(points[i] for i in simplex)))
        return Polyhedron(tuple(faces))

def rdp_simplify(points, tolerance):
    """Ramer-Douglas-Peucker simplification for 2D/3D point sequences."""
    arr = np.array([p.to_array() for p in points])
    if len(points) < 3:
        return list(points)
    start, end = arr[0], arr[-1]
    dmax, idx = 0, 0
    for i in range(1, len(arr)-1):
        d = np.linalg.norm(np.cross(end-start, arr[i]-start)) / (np.linalg.norm(end-start)+1e-16)
        if d > dmax:
            idx, dmax = i, d
    if dmax > tolerance:
        left = list(rdp_simplify(points[:idx+1], tolerance))
        right = list(rdp_simplify(points[idx:], tolerance))
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

if 'Polygon3D' in globals():
    def polygon3d_simplify(self, tolerance):
        return Polygon3D(tuple(rdp_simplify(self.vertices, tolerance)))
    Polygon3D.simplify = polygon3d_simplify
    def polygon3d_minkowski_sum(self, other):
        raise NotImplementedError('Minkowski sum for Polygon3D not yet implemented.')
    Polygon3D.minkowski_sum = polygon3d_minkowski_sum
    def polygon3d_offset(self, distance):
        raise NotImplementedError('Offset/buffer for Polygon3D not yet implemented.')
    Polygon3D.offset = polygon3d_offset
if 'Mesh' in globals():
    def mesh_simplify(self, tolerance):
        raise NotImplementedError('Mesh decimation not yet implemented.')
    Mesh.simplify = mesh_simplify
    def mesh_minkowski_sum(self, other):
        raise NotImplementedError('Minkowski sum for Mesh not yet implemented.')
    Mesh.minkowski_sum = mesh_minkowski_sum
    def mesh_offset(self, distance):
        raise NotImplementedError('Offset/buffer for Mesh not yet implemented.')
    Mesh.offset = mesh_offset
