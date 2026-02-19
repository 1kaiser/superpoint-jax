import numpy as np
import trimesh
try:
    import laspy
except ImportError:
    laspy = None

def export_to_las(points, confidence, filename):
    """
    Exports a point cloud to a LAS file with confidence as an extra dimension.

    Args:
        points: (N, 3) numpy array of points.
        confidence: (N,) numpy array of confidence values.
        filename: Output filename (str or Path).
    """
    if laspy is None:
        print("laspy not found, skipping LAS export.")
        return

    try:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.add_extra_dims([laspy.ExtraBytesParams(name="confidence", type=np.float32)])

        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.confidence = confidence

        las.write(str(filename))
        print(f"Saved LAS to {filename}")
    except Exception as e:
        print(f"Error exporting LAS: {e}")

def export_to_glb(points, confidence, filename):
    """
    Exports a point cloud to a GLB file using Trimesh.
    Points are colored based on confidence (Grayscale).

    Args:
        points: (N, 3) numpy array.
        confidence: (N,) numpy array, expected range [0, 1].
        filename: Output filename.
    """
    try:
        # Create colors from confidence (White=High, Black=Low)
        # confidence is 0..1
        c = (confidence * 255).astype(np.uint8)
        colors = np.stack([c, c, c, np.full_like(c, 255)], axis=-1)

        pcd = trimesh.points.PointCloud(vertices=points, colors=colors)

        # Add simple anchor spheres?
        # The prompt mentioned "Node 1..N: Named Anchor Spheres (Annotatable)".
        # Without specific anchor locations, we'll skip adding extra nodes for now
        # or maybe add a bounding box visual.
        # Let's just export the merged cloud.

        pcd.export(str(filename))
        print(f"Saved GLB to {filename}")
    except Exception as e:
        print(f"Error exporting GLB: {e}")
