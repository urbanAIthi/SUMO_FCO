
import open3d as o3d
import numpy as np
from typing import List
import os
import plotly.graph_objects as go


def mesh_to_pc(mesh_path: str, point_density: int, new_length: float, rotation: List[float], translation: List[float] = [0,0,0],  vis_geometry: bool = False, vis_pc: bool = False, save_pc: bool=False) -> np.ndarray:
    """
    Convert a mesh to a point cloud with specified point density and bounding box length.

    Args:
        mesh_path (str): Path to the mesh file.
        point_density (int): Desired point density of the point cloud  [points/m**2].
        new_length (float): Desired length of the bounding box of the mesh after scaling [m].
        rotation (List[float]): List of three rotation angles in degrees [x, y, z].
        vis_geometry (bool, optional): Flag to visualize the mesh and coordinate frame. Defaults to False.
        vis_pc (bool, optional): Flag to visualize the point cloud and coordinate frame. Defaults to False.

    Returns:
        np.ndarray: Numpy array containing the point cloud.

    """
    # check if the mesh file exists
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f'{mesh_path} does not exist.')
    # check if the mesh fiel is a .ply file
    if not mesh_path.endswith('.npy'):
        mesh = o3d.io.read_triangle_mesh(mesh_path, True)
        mesh.compute_vertex_normals()

        if vis_geometry:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=100, origin=[0, 0, 0]
            )
            # Visualize the mesh and coordinate frame
            vis_geometry = [mesh, coordinate_frame]
            o3d.visualization.draw_geometries(vis_geometry)

        if new_length is not None:
            # Get the minimum and maximum coordinates
            min_bound = mesh.get_min_bound()
            max_bound = mesh.get_max_bound()

            old_length = (max_bound - min_bound).max()

            scale_factor = new_length / old_length

            mesh.scale(scale_factor, center=(0, 0, 0))

        mesh_surface_area = mesh.get_surface_area()
        print(f'{mesh_path} surface area: {mesh_surface_area}')
        num_points = int(mesh_surface_area * point_density * 100) # calculate the number and scale from cm2 to m2

        rotation = np.radians(rotation)
        x_angle, y_angle, z_angle = rotation[0], rotation[1], rotation[2]

        # Create rotation matrices for each axis
        Rx = np.array([[1, 0, 0], [0, np.cos(x_angle), -np.sin(x_angle)], [0, np.sin(x_angle), np.cos(x_angle)]])
        Ry = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
        Rz = np.array([[np.cos(z_angle), -np.sin(z_angle), 0], [np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])

        # Combine the rotation matrices
        R = np.dot(Rz, np.dot(Ry, Rx))

        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
        old_centroid = np.mean(point_cloud.points, axis=0)
        print(f'Old centroid: {old_centroid}')

        center_vec = np.array(translation)
        centered_pc = point_cloud.translate(center_vec, relative=False)
        new_centroid = np.mean(centered_pc.points, axis=0)
        print(f'New centroid: {new_centroid}')

        # Rotate the point cloud
        point_cloud.points = o3d.utility.Vector3dVector(np.dot(np.asarray(centered_pc.points), R))

    else:
        # read the point cloud from the .ply file
        arr = np.load(mesh_path)
        #translate the point cloud by x_trans and y_trans
        #arr[:,0] += 6797.25 + 15
        #arr[:,1] += 5033 + 140
        # get min value of z
        #arr[:,2] += np.min(arr[:,2]) * -1
        # Downsample your point cloud
        downsample_rate = 0.01  # Adjust this as needed
        downsampled_arr = arr[np.random.choice(arr.shape[0], int(arr.shape[0] * downsample_rate), replace=False)]
        print(f'Point cloud shape: {arr.shape}')
        # Create a scatter plot
        # Create a scatter plot
        scatter = go.Scatter3d(x=downsampled_arr[:, 0], y=downsampled_arr[:, 1], z=downsampled_arr[:, 2],
                               mode='markers')

        # Create a layout
        layout = go.Layout(scene=dict(xaxis=dict(title='X'),
                                      yaxis=dict(title='Y'),
                                      zaxis=dict(title='Z')))

        # set the axis range equal to -250 to 250
        layout.scene.xaxis.range = [-250 + 6797, 250 + 6797]
        layout.scene.yaxis.range = [-250 + 5033 + 140, 250 + 5033 + 140]
        layout.scene.zaxis.range = [-630, -130]

        #make the points smaller
        scatter.marker.size = 0.5

        # Create a figure
        fig = go.Figure(data=[scatter], layout=layout)

        # Register a click event listener
        def handle_click(trace, points, state):
            if points.point_inds:
                ind = points.point_inds[0]
                print("Coordinates of the selected point: ", downsampled_arr[ind])

        scatter.on_click(handle_click)

        fig.show()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(arr)

    if vis_pc:
        # Create a coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=10, origin=[0, 0, 0]
        )

        # Visualize the mesh and coordinate frame
        vis_geometry = [point_cloud, coordinate_frame]
        o3d.visualization.draw_geometries(vis_geometry)

    if save_pc:
        # get the file name and extract the file name without extension
        file_name = os.path.basename(mesh_path)
        file_name = os.path.splitext(file_name)[0]
        np.save(f'{file_name}_pd{point_density}_new.npy', point_cloud.points)

    return point_cloud

if __name__ == "__main__":

    mesh_path = '../meshes/Buildings/KIVI/KIVI_Vegetation.FBX'
    point_density = 2
    length = None

    point_cloud = mesh_to_pc(mesh_path, point_density, length, rotation = [0,0,0], translation = [-15.22 + 6797+15, 39.023 + 5033 + 140, 0],  vis_geometry=True, vis_pc=True, save_pc=True)

