import os
import open3d as o3d
from scipy.interpolate import interp1d
import numpy as np
from PIL import Image
from typing import Dict, Tuple


def create_camera_matrices_from_config(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Create camera matrices from config file"""
    camera_matrices = {}

    def add_to_dict(key, value):
        """Helper function to add a key-value pair to the camera_matrices dict"""
        if key not in camera_matrices:
            camera_matrices[key] = {}
        camera_matrices[key].update(value)

    # Front camera
    if config['FRONT_CAMERA']['ACTIVE']:
        front_cam_config = config['FRONT_CAMERA']
        K, P = init_artificial_camera(
            front_cam_config['FOV'], front_cam_config['IMAGE_WIDTH'], front_cam_config['IMAGE_HEIGHT'],
            rotation=front_cam_config['ROTATION_ANGLE'], translation_offset=front_cam_config['TRANSLATION_OFFSET']
        )
        add_to_dict('front',
                    {'K': K, 'P': P, 'fov': front_cam_config['FOV'], 'image_width': front_cam_config['IMAGE_WIDTH'],
                     'image_height': front_cam_config['IMAGE_HEIGHT']})

    # Back camera
    if config['BACK_CAMERA']['ACTIVE']:
        back_cam_config = config['BACK_CAMERA']
        K, P = init_artificial_camera(
            back_cam_config['FOV'], back_cam_config['IMAGE_WIDTH'], back_cam_config['IMAGE_HEIGHT'],
            rotation=back_cam_config['ROTATION_ANGLE'], translation_offset=back_cam_config['TRANSLATION_OFFSET']
        )
        add_to_dict('back',
                    {'K': K, 'P': P, 'fov': back_cam_config['FOV'], 'image_width': back_cam_config['IMAGE_WIDTH'],
                     'image_height': back_cam_config['IMAGE_HEIGHT']})

    # Left camera
    if config['LEFT_CAMERA']['ACTIVE']:
        left_cam_config = config['LEFT_CAMERA']
        K, P = init_artificial_camera(
            left_cam_config['FOV'], left_cam_config['IMAGE_WIDTH'], left_cam_config['IMAGE_HEIGHT'],
            rotation=left_cam_config['ROTATION_ANGLE'], translation_offset=left_cam_config['TRANSLATION_OFFSET']
        )
        add_to_dict('left',
                    {'K': K, 'P': P, 'fov': left_cam_config['FOV'], 'image_width': left_cam_config['IMAGE_WIDTH'],
                     'image_height': left_cam_config['IMAGE_HEIGHT']})

    # Right camera
    if config['RIGHT_CAMERA']['ACTIVE']:
        right_cam_config = config['RIGHT_CAMERA']
        K, P = init_artificial_camera(
            right_cam_config['FOV'], right_cam_config['IMAGE_WIDTH'], right_cam_config['IMAGE_HEIGHT'],
            rotation=right_cam_config['ROTATION_ANGLE'], translation_offset=right_cam_config['TRANSLATION_OFFSET']
        )
        add_to_dict('right',
                    {'K': K, 'P': P, 'fov': right_cam_config['FOV'], 'image_width': right_cam_config['IMAGE_WIDTH'],
                     'image_height': right_cam_config['IMAGE_HEIGHT']})

    return camera_matrices


def plot_point_cloud(pt_cloud: np.ndarray):
    # Convert points back to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_cloud)

    # Visualize filtered point cloud
    o3d.visualization.draw_geometries([pcd])


def init_artificial_camera(fov: int, image_width: int, image_height: int, rotation: np.ndarray = None,
                           translation_offset: np.ndarray = None) -> Tuple[np.array, np.array]:
    # camera intrinsics
    fx = fy = (image_width / 2) / np.tan(np.radians(fov / 2))
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # world coordinate sytem to camera coordinate sytem
    #
    #       z   y                      z
    #       |  /                      /
    #       | /                      /
    #       ----x          ->        ----x
    #                                |
    #                                |
    #                                y

    # Define the rotation angles in degrees around the x, y, and z axes
    rx, ry, rz = np.deg2rad(rotation[0]), np.deg2rad(rotation[1]), np.deg2rad(rotation[2])

    # Compute the rotation matrix around the x axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    # Compute the rotation matrix around the y axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Compute the rotation matrix around the z axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combine the rotation matrices into a single rotation matrix
    R = Rz @ Ry @ Rx

    # Define the projection matrix without translation and rotation
    P_initial = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    P_proj = np.zeros(shape=(4, 4))
    P_proj[3, 3] = 1.0

    # Add the rotation to the projection matrix
    P_proj[:3, :3] = R

    # Add the translation to the projection matrix
    P_proj[:3, 3] = -translation_offset

    P = P_proj @ P_initial

    return K, P


def is_point_in_image(pt_cloud: np.ndarray, K: np.ndarray, P: np.ndarray, image_size: Tuple[int, int],
                      max_distance: int = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Check if points in point cloud are within the image plane defined by the camera intrinsics matrix K.
    :param pt_cloud: numpy array of shape (N,3 or 4) representing 3D coordinates of point cloud.
    :param K: numpy array of shape (3,3) representing the camera intrinsics matrix.
    :param P: numpy array of shape (3,4) representing the camera extrinsics matrix.
    :param image_size: tuple of integers (width,height) representing the size of the image plane.
    :param max_distance: integer distance indicating which points can be projected into the image plane
    :return: boolean mask of length N indicating which points lie within the image plane
        and the projected points of size (N,4) where the fourth dimension is the inverse depth (disparity).
    '''

    # Project 3D world points to 3D camera points
    if pt_cloud.shape[1] == 3:
        pt_cloud = np.hstack((pt_cloud, np.ones((pt_cloud.shape[0], 1))))

    # Project 3D points onto 2D image plane using camera intrinsics matrix K
    # Make K 4x4
    if K.shape == (3, 3):
        _K = np.zeros(shape=(4, 4))
        _K[:3, :3] = K
        _K[3, 3] = 1.0
        K = _K

    if P.shape == (3, 4):
        _P = np.zeros(shape=(4, 4))
        _P[:3, :4] = P
        _P[3, 3] = 1.0
        P = _P

    projected_points = np.dot(np.dot(K, P), pt_cloud.T)

    # add eps for zeros values to avoid divide by zero error
    # Define epsilon value
    eps = 1e-6

    # Replace zero values with epsilon
    projected_points[2, :] = np.where(projected_points[2, :] == 0, eps, projected_points[2, :])

    zs = np.copy(projected_points[2, :])
    projected_points /= zs
    projected_points = projected_points.T

    # distance_offset = np.linalg.inv(P).dot(np.array([0,0,0,1]))[2]

    # Check if projected points lie within image boundaries
    max_distance = max_distance if max_distance else 9999
    point_mask = np.logical_and.reduce((
        projected_points[:, 0] >= 0,
        projected_points[:, 0] < image_size[0],
        projected_points[:, 1] >= 0,
        projected_points[:, 1] < image_size[1],
        zs > 0,
        zs <= max_distance  # + distance_offset
    ))

    return point_mask, projected_points


def filter_points_by_radius(pt_cloud: np.ndarray, center: np.ndarray, radius: int) -> np.ndarray:
    """
    Filter 3D points based on their distance from a center point.
    :param pt_cloud: numpy array of shape (N,3) representing 3D coordinates of point cloud.
    :param center: numpy array of shape (3,) representing the center point.
    :param radius: float representing the radius in which points will be kept.
    :return: numpy array of shape (M,3) containing only the points within the specified radius.
    """
    # Compute Euclidean distance between each point and the center point
    distances = np.sqrt(np.sum((pt_cloud - center) ** 2, axis=1))

    # Filter points based on distance from center point
    mask = distances <= radius

    return mask


def extract_unique_objects_from_point_cloud(pt_cloud: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Extract unique objects from a dense point cloud using DBSCAN clustering.
    :param pt_cloud: numpy array of shape (N,3) representing 3D coordinates of point cloud.
    :return: dictionary containing unique object IDs as keys and numpy arrays of corresponding points as values.
    """
    # Convert point cloud to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_cloud)

    # Define DBSCAN clustering parameters
    eps = 0.1
    min_points = 10

    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()

    # Extract objects from clusters
    objects = {}
    for i in range(max_label + 1):
        # Select points of the current cluster
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            cluster_points = np.asarray(pcd.select_by_index(cluster_indices).points)

            # Add points to objects dictionary
            objects[i] = cluster_points

    return objects


def project_circle_to_3d(center: np.ndarray, radius: int, points: int = 1000):
    # create 3d points for circle (x,y,z) where z (top) is always zero (0) and y is the depth
    circle_points = np.zeros(shape=(points, 3))
    circle_points[:, 0] = np.cos(np.linspace(0, 2 * np.pi, points)) * radius
    circle_points[:, 1] = np.sin(np.linspace(0, 2 * np.pi, points)) * radius
    circle_points[:, 2] = 0

    circle_points += center

    return circle_points


def project_rays_to_3d(K: np.ndarray, P: np.ndarray, image_size: np.ndarray, distance: int = 1) -> np.ndarray:
    """
    Project rays from every pixel of the image plane to 3D space.

    Args:
        K (numpy.ndarray): The camera intrinsic matrix.
        P (numpy.ndarray): The camera projection matrix.
        image_size (tuple): The size of the image in pixels (width, height).
        distance (int): Depth of the projection (in meters).

    Returns:
        numpy.ndarray: An array of 3D points representing the rays.
    """

    # image width and height
    width, height = image_size

    # Generate pixel coordinates for every pixel in the image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.column_stack((x.ravel(), y.ravel()))

    # Convert pixel coordinates to normalized image coordinates
    pixel_coords = np.hstack((pixels, np.ones((pixels.shape[0], 1))))

    norm_camera_coords = np.linalg.inv(K).dot(pixel_coords.T).T
    # norm_camera_coords = np.concatenate((norm_camera_coords, np.ones((norm_camera_coords.shape[0], 1))), axis=1)

    # Compute ray direction vectors in camera coordinate system
    inv_P = np.linalg.inv(P)
    min_distance_image_plane = norm_camera_coords.dot(inv_P[:3, :3].T)

    # max distance image plane
    max_distance_image_plane = min_distance_image_plane * distance
    min_distance_image_plane = min_distance_image_plane + inv_P[:3, 3]

    # Add a line of 500 3D points from each of the four corners of the min_distance_image_plane to the four corners of the max_distance_image_plane

    rays = []
    #             top left, top right, bottom left, bottom right
    for corner in [0, width - 1, width * height - width, width * height - 1]:
        _corner = np.zeros(shape=(500, 3))
        step_size_0 = (max_distance_image_plane[corner, 0] - min_distance_image_plane[corner, 0]) / 500
        _corner[:, 0] = np.arange(min_distance_image_plane[corner, 0], max_distance_image_plane[corner, 0],
                                  step_size_0)[:500]
        step_size_1 = (max_distance_image_plane[corner, 1] - min_distance_image_plane[corner, 1]) / 500
        _corner[:, 1] = np.arange(min_distance_image_plane[corner, 1], max_distance_image_plane[corner, 1],
                                  step_size_1)[:500]
        step_size_2 = (max_distance_image_plane[corner, 2] - min_distance_image_plane[corner, 2]) / 500
        _corner[:, 2] = np.arange(min_distance_image_plane[corner, 2], max_distance_image_plane[corner, 2],
                                  step_size_2)[:500]

        rays.append(_corner)

    rays = np.concatenate(rays, axis=0)

    dense_ray_points_world = np.concatenate([rays, min_distance_image_plane, max_distance_image_plane])

    return dense_ray_points_world


def create_centered_box(offset: np.ndarray, box_length: float, box_width: float, box_height: float,
                        points_no: int = 100000) -> np.ndarray:
    if offset is None:
        offset = np.array([0, 0, 0])

    # Define the number of points in each dimension
    n_points_per_dim = int(np.ceil(pow(points_no, 1.0 / 3)))

    # Create a grid of points centered around 0,0,0
    x, y, z = np.meshgrid(np.linspace(-box_width / 2, box_width / 2, n_points_per_dim),
                          np.linspace(-box_length / 2, box_length / 2, n_points_per_dim),
                          np.linspace(-box_height / 2, box_height / 2, n_points_per_dim))

    # add offset
    x, y, z = x + offset[0], y + offset[1], z + offset[2]

    # Reshape the grid of points into a point cloud of shape (N, 3)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    return points


def create_oracle_depth_image(K: np.ndarray, P: np.ndarray, pt_cloud: np.ndarray, image_size: Tuple[int, int],
                              max_depth: int,
                              image_plane_mask: np.ndarray = None, disparity_info: np.ndarray = None) -> np.ndarray:
    """
    Creates a depth image from the given point cloud.

    Args:
        K (numpy.ndarray): The camera intrinsic matrix.
        P (numpy.ndarray): The camera extrinsic matrix.
        point_cloud (np.ndarray): numpy array of shape (N,3) representing 3D coordinates of point cloud.
        image_size (tuple): The size of the image in pixels (width, height).
        max_depth (int): Max distance of objects to consider.
        image_plane_mask: Contains information about pixels where the depth image is not equal to zero.
        disparity_info: Contains the disparity values for each occupied pixel (shape: (N, 4)).

    Returns:
        numpy.ndarray: Depth image of shape image size
    """
    assert (image_plane_mask is None) == (
                disparity_info is None), 'Either set both parameters (image_plane_mask and disparity_info) or set both to None.'
    if image_plane_mask is None and disparity_info is None:
        point_mask, projected_points = is_point_in_image(pt_cloud=pt_cloud, K=K, P=P, image_size=image_size,
                                                         max_distance=max_depth)
    else:
        point_mask = image_plane_mask
        projected_points = disparity_info

    projected_points = projected_points[point_mask, :]
    projected_points[:, :2] = np.floor(projected_points[:, :2])

    width, height = image_size[0], image_size[1]

    # Extract the image coordinates and disparity information
    img_coords = projected_points[:, :2]
    # Round the image coordinates and convert them to integers
    img_coords = np.round(img_coords).astype(np.int32)
    disparity = projected_points[:, 3]

    # Create an empty grayscale image to store the depth map
    depth_img = np.zeros((height, width, 1), dtype=np.uint16)

    # Compute the depth values for each point
    depth_values = 1.0 / disparity

    # Create a dictionary to store the highest depth value for each pixel coordinate
    max_depth = {}

    # Iterate over each projected point
    for i in range(len(projected_points)):
        # Get the image coordinates and depth value for this point
        x, y = int(img_coords[i, 0]), int(img_coords[i, 1])
        depth = depth_values[i]

        # Check if this pixel coordinate has been seen before
        if (x, y) in max_depth:
            # If so, update the maximum depth value if this point is closer
            max_depth[(x, y)] = min(max_depth[(x, y)], depth)
        else:
            # Otherwise, add this pixel coordinate and depth value to the dictionary
            max_depth[(x, y)] = depth

    # Set the depth values in the depth map using the maximum depth values for each pixel coordinate
    for (x, y), depth in max_depth.items():
        depth_img[y, x, 0] = depth

    return depth_img.astype(np.float32)


def fill_sparse_building_representation(depth_img: np.ndarray) -> np.ndarray:
    # Separate x and y coordinates
    y_coords, x_coords, _ = depth_img.shape
    y_coords = np.arange(y_coords, dtype=np.uint32)
    x_coords = np.arange(x_coords, dtype=np.uint32)

    # Initialize arrays to store max and min y values for each unique x value
    max_y_values = np.zeros(x_coords.shape)
    top_y_indices = np.zeros(x_coords.shape)
    min_y_values = np.zeros(x_coords.shape)
    bot_y_indices = np.zeros(x_coords.shape)

    # Loop over unique x values
    for x in x_coords:
        # Find max and min y values for these points
        min_max = np.where(depth_img[:, x] > 0)[0]
        if not np.any(min_max):
            continue

        # Store max and min y values in arrays
        max_y_values[x] = np.max(depth_img[min_max, x])
        top_y_indices[x] = np.max(min_max)
        min_y_values[x] = np.min(depth_img[min_max, x])
        bot_y_indices[x] = np.min(min_max)

    # Interpolate missing values between top and bottom bounds for each column
    for x in x_coords:
        if top_y_indices[x] > bot_y_indices[x]:
            indices = np.arange(bot_y_indices[x], top_y_indices[x] + 1, dtype=np.uint32)
            values = np.interp(indices, [bot_y_indices[x], top_y_indices[x]], [min_y_values[x], max_y_values[x]])
            depth_img[indices, x, 0] = values

    # Interpolate remaining gaps along rows
    for y in y_coords:
        row = depth_img[y, :]
        valid_indices = np.where(row > 0)[0]
        if len(valid_indices) < 2:
            continue
        interp_func = interp1d(valid_indices, row[valid_indices].flatten(), kind='linear', bounds_error=False,
                               fill_value=(row[valid_indices[0]], row[valid_indices[-1]]))
        depth_img[y, :] = interp_func(x_coords)

    return depth_img


def filter_by_oracle_depth(oracle_depth_img: np.ndarray, ref_depth_img: np.ndarray,
                           threshold: float = 0.1) -> np.ndarray:
    """
    Filters out pixel coordinates from depth image 2 that are equal to the depth values in depth image 1 with a given threshold.

    Args:
        oracle_depth_img (np.ndarray): Numpy array of shape (H, W, 1) representing the first depth image.
        ref_depth_img (np.ndarray): Numpy array of shape (H, W, 1) representing the second depth image.
        threshold (float): Threshold to determine if pixel coordinates of ref_depth_img are equal to the depth values in oracle_depth_img.

    Returns:
        np.ndarray: Numpy array of shape (M, 2) representing the pixel coordinates of ref_depth_img that meet the threshold.
    """
    # Ensure both depth maps have the same shape
    assert oracle_depth_img.shape == ref_depth_img.shape

    # Get the coordinates of all non-zero pixels in oracle_depth_img
    occ_indices = np.argwhere(ref_depth_img != 0)

    # Initialize an empty array to store the valid coordinates
    valid_coords = np.empty((0, 2), dtype=int)

    # Loop over each pixel in coord1
    for c1 in occ_indices:
        # Get the corresponding depth value in depth_map2
        depth_value = ref_depth_img[c1[0], c1[1]]

        # If the depth value in depth_map2 is within threshold of the depth value in depth_map1,
        # add the pixel coordinates to the valid_coord2 array
        if abs(depth_value - oracle_depth_img[c1[0], c1[1]]) < threshold:
            valid_coords = np.append(valid_coords, [[c1[0], c1[1]]], axis=0)

    # Return the valid coordinates
    return valid_coords


def create_depth_image_from_array(depth_array: np.ndarray, max_depth: int, file_path: str) -> None:
    """
    Creates a depth image from a depth array.

    Args:
        depth_array (np.ndarray): Numpy array of shape (H, W, 1) representing the depth array.
        max_depth (int): Maximum depth value in the depth array.
        file_path (str): Path to save the depth image.

    Returns:
        np.ndarray: Numpy array of shape (H, W, 3) representing the depth image.
    """
    # Create a copy of the depth array
    depth_img = depth_array.copy()

    # min = 0; max = max_depth
    depth_img = depth_img.clip(0, max_depth)

    # Normalize the depth values to the range [0, 1]
    depth_img = depth_img / max_depth

    # Save depth image where we use RGB format with PIL
    depth_img = (depth_img * 255).astype(np.uint8)

    # Save to disk (PIL)
    depth_img = Image.fromarray(depth_img.squeeze())
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    depth_img.save(file_path)


