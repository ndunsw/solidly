
#!/usr/bin/env python3
import numpy as np
from solidly.core import Vector3, Point3D, Polygon3D, Mesh
from solidly.spatial import compute_aabb, AABBTree
from solidly.plotting import plot_point3d, plot_line3d, plot_polygon3d
import matplotlib.pyplot as plt

def demo_objects():
    # Vector and Point construction
    v = Vector3(1, 2, 3)
    p = Point3D(0, 0, 0)
    print("Vector3:", v)
    print("Point3D:", p)

    # Polygon construction
    poly1 = Polygon3D((Point3D(0,0,0), Point3D(1,0,0), Point3D(0,1,0)))
    poly2 = Polygon3D((Point3D(0,0,0), Point3D(0,1,0), Point3D(-1,0,0)))
    print("Polygon1:", poly1)
    print("Polygon2:", poly2)

    # Boolean operations
    union_poly = poly1.union(poly2)
    inter_poly = poly1.intersection(poly2)
    diff_poly = poly1.difference(poly2)
    print("Union:", union_poly)
    print("Intersection:", inter_poly)
    print("Difference:", diff_poly)

    # Mesh construction
    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])
    mesh = Mesh.from_vertices_faces(verts, faces)
    print("Mesh:", mesh)
    print("Bounding box:", mesh.bounding_box())
    print("Surface area:", mesh.surface_area())
    print("Volume:", mesh.volume())
    print("Closed:", mesh.is_closed())

    # Spatial calculations
    aabb = compute_aabb([Point3D(*v) for v in verts])
    print("AABB:", aabb)
    tree = AABBTree([poly1, poly2], lambda poly: compute_aabb(poly.vertices))
    print("AABBTree root:", tree.root)

    # Visualization
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_polygon3d(poly1, ax=ax1, color='b', label='poly1')
    plot_polygon3d(poly2, ax=ax1, color='g', label='poly2')
    if union_poly:
        plot_polygon3d(union_poly, ax=ax1, color='r', linestyle='--', label='union')
    ax1.set_title('Polygons and Union')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(verts[:,0], verts[:,1], verts[:,2], c='k')
    for tri in faces:
        tri_verts = verts[tri]
        tri_verts = np.vstack([tri_verts, tri_verts[0]])
        ax2.plot(tri_verts[:,0], tri_verts[:,1], tri_verts[:,2], c='orange')
    ax2.set_title('Mesh')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo_objects()
