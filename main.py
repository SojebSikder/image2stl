from stl import mesh
import cv2
import numpy as np

# Load grayscale image
img = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or unsupported format")

# Resize for detail control
target_size = (200, 200)
img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

# Scaling factors
scale = 1.0               # mm per pixel
height_scale = 10.0        # height exaggeration
base_thickness = 1.0      # baseplate thickness in mm

height, width = img.shape

# Prepare height map (z values)
z_top = img.astype(np.float32) / 255.0 * height_scale + base_thickness
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
x_coords = x_coords * scale
y_coords = y_coords * scale

# Top vertices (x, y, z)
verts_top = np.stack((x_coords, y_coords, z_top), axis=-1)

# Base vertices (x, y, z=0)
verts_base = np.stack((x_coords, y_coords, np.zeros_like(z_top)), axis=-1)

# Collect all faces
faces = []

# Helper to create two triangles for a quad
def quad_to_triangles(v00, v10, v01, v11):
    return [[v00, v10, v01], [v10, v11, v01]]

# Top surface faces
for y in range(height - 1):
    for x in range(width - 1):
        faces += quad_to_triangles(
            verts_top[y, x], verts_top[y + 1, x],
            verts_top[y, x + 1], verts_top[y + 1, x + 1]
        )

# Bottom surface (underside of baseplate)
for y in range(height - 1):
    for x in range(width - 1):
        faces += quad_to_triangles(
            verts_base[y, x], verts_base[y, x + 1],
            verts_base[y + 1, x], verts_base[y + 1, x + 1]
        )

# Side walls (connect top and base around the edges)
def add_side_faces(vt1, vt2, vb1, vb2):
    faces.append([vt1, vb1, vt2])
    faces.append([vt2, vb1, vb2])

# Left and right walls
for y in range(height - 1):
    add_side_faces(verts_top[y, 0], verts_top[y + 1, 0], verts_base[y, 0], verts_base[y + 1, 0])
    add_side_faces(verts_top[y + 1, -1], verts_top[y, -1], verts_base[y + 1, -1], verts_base[y, -1])

# Top and bottom walls
for x in range(width - 1):
    add_side_faces(verts_top[0, x], verts_top[0, x + 1], verts_base[0, x], verts_base[0, x + 1])
    add_side_faces(verts_top[-1, x + 1], verts_top[-1, x], verts_base[-1, x + 1], verts_base[-1, x])

# Convert faces to mesh
faces_np = np.array(faces, dtype=np.float32)
stl_mesh = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))
for i in range(faces_np.shape[0]):
    stl_mesh.vectors[i] = faces_np[i]

# Save STL
stl_mesh.save("output.stl")
print("Saved as output.stl")
