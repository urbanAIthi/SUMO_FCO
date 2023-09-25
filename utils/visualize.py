import pickle

import numpy as np
import open3d as o3d
from typing import Dict, List

import torch

from utils.utils_cv import filter_points_by_radius, project_rays_to_3d, create_centered_box, project_circle_to_3d
from utils.create_box import get_object_dimensions, create_box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs.config_simulation import TRAFFIC
import traci
import os


def plot_points_to_vis(vis, points: np.ndarray, color: np.ndarray = None):
    if not isinstance(points, np.ndarray):
        return
    # create point cloud objects
    o3d_points = o3d.geometry.PointCloud()
    # numpy to vector 3d
    o3d_points.points = o3d.utility.Vector3dVector(points[:, :3])
    # set color
    if color is not None:
        o3d_points.paint_uniform_color(color)
    # add to visualization
    vis.add_geometry(o3d_points)


def obj_colors(obj_name: str) -> np.ndarray:
    if obj_name == 'building':
        return np.array([0, 0, 1])
    elif obj_name == 'vehicle':
        return np.array([0.5, 0, 0])
    elif obj_name == 'pedestrian':
        return np.array([0.5, 0.5, 0])
    elif obj_name == 'cyclist':
        return np.array([0.5, 0, 0.5])
    elif obj_name == 'ego':
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 0])


def visualize_frame(detected_objects: Dict[str, List], other_objects: Dict[str, List],
                    obj_pt_clouds: Dict[str, Dict[str, np.ndarray]],
                    cameras: Dict, scan_radius: int, objects_image_plane_mask: Dict[str, Dict[str, np.ndarray]],
                    plot_image_plane_rays: bool = False, plot_scan_radius: bool = False, center: np.ndarray = None):
    if center is None:
        center = np.array([0, 0, 0])

    ego_vehicle_pt_cloud = create_centered_box(offset=[0, -2.25, 0.75], box_length=4.5, box_width=1.8, box_height=1.5)
    # ego_vehicle_pt_cloud = create_centered_box(offset=[0, 0, 0], box_length=1, box_width=1, box_height=1)

    # Visualize using Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Filter object's points by range
    other_objects_pt_cloud_radius_mask = {
        obj_type: {
            fn: filter_points_by_radius(
                pt_cloud=obj_pt_clouds[obj_type][fn], center=center, radius=scan_radius
            ) for fn in other_objects[obj_type]}
        for obj_type in other_objects
    }

    # Only those that are "other objects"
    other_objects_image_plane_mask = {
        fn: objects_image_plane_mask[obj][fn] for obj in objects_image_plane_mask for fn in
        objects_image_plane_mask[obj]
    }

    # plot ego vehicle
    plot_points_to_vis(vis=vis, points=ego_vehicle_pt_cloud, color=obj_colors('ego'))

    # project the image plane rays into the visualization
    if plot_image_plane_rays:
        for cam_pos in cameras:
            K, P = cameras[cam_pos]['K'], cameras[cam_pos]['P']
            img_width, img_height = cameras[cam_pos]['image_width'], cameras[cam_pos]['image_height']
            ray_planes = project_rays_to_3d(K, P, image_size=(img_width, img_height), distance=scan_radius)
            plot_points_to_vis(vis=vis, points=ray_planes, color=np.array([0.4, 0.7, 0.7]))

    if plot_scan_radius:
        # plot scan radius
        scan_radius_points = project_circle_to_3d(radius=scan_radius, center=center)
        plot_points_to_vis(vis=vis, points=scan_radius_points, color=np.array([0, 0, 0]))

    for det_obj in detected_objects:
        for filename in detected_objects[det_obj]:
            pt_cloud = obj_pt_clouds[det_obj][filename]
            plot_points_to_vis(vis=vis, points=pt_cloud, color=np.array([0.2, 0.7, 0.2]))

    for o_obj in other_objects:
        for obj_filename in other_objects[o_obj]:
            pt_cloud = obj_pt_clouds[o_obj][obj_filename]
            radius_mask = other_objects_pt_cloud_radius_mask[o_obj][obj_filename]
            if obj_filename in other_objects_image_plane_mask:
                image_plane_mask = other_objects_image_plane_mask[obj_filename]
            else:
                image_plane_mask = np.zeros(radius_mask.shape, dtype=bool)

            if obj_filename.startswith('building'):
                # we always plot the buildings
                plot_points_to_vis(vis=vis, points=pt_cloud, color=obj_colors('building'))
                continue

            # Plot the object no matter where it is located
            inital_pt_cloud = pt_cloud[np.logical_and(radius_mask == False, image_plane_mask == False), :]
            if np.any(inital_pt_cloud):
                plot_points_to_vis(vis=vis, points=pt_cloud, color=obj_colors(o_obj))

            radius_pt_cloud = pt_cloud[np.logical_and(radius_mask, image_plane_mask == False), :]
            if np.any(radius_pt_cloud):
                plot_points_to_vis(vis=vis, points=pt_cloud, color=obj_colors(o_obj) + np.array([0, 0.5, 0.5]))

            # Assign a color to the objects that are inside the radius (except buildings) and on the image plane
            # On the image plane but not detectable because of occlusion.
            radius_pt_cloud = pt_cloud[np.logical_and(radius_mask, image_plane_mask), :]
            if np.any(radius_pt_cloud):
                plot_points_to_vis(vis=vis, points=pt_cloud, color=obj_colors(o_obj) + np.array([0, 0.5, 0]))

    # Visualize
    vis.run()
    vis.destroy_window()


def visualize_fco(all_vehicles: List[str], all_cyclist: List[str], all_pedestrians: List[str],
                  detected_vehicles: List[str], detected_cyclists: List[str], detected_pedestrians: List[str],
                  fco_list: List[str], center: List[str], show=False, save=False, save_path='visualization'):
    """
    Visualize the objects that are detected and not detected in the current timestamp
    :param all_vehicles: list of all vehicles (within the detection radius) in the current timestamp
    :param all_cyclist: list of all cyclists (within the detection radius) in the current timestamp
    :param all_pedestrians: list of all pedestrians (within the detection radius) in the current timestamp
    :param detected_vehicles: list of all detected vehicles in the current timestamp
    :param detected_cyclists: list of all detected cyclists in the current timestamp
    :param detected_pedestrians: list of all detected pedestrians in the current timestamp
    :param fco_list: list of the floating car observers in the current timestamp

    shows and saves the visualization
    """
    fig, ax = plt.subplots()
    # append the fco list to all vehicles to plot them
    all_vehicles.extend(fco_list)
    for box in all_vehicles:
        color = get_color(detected_vehicles, fco_list, box)
        # get the vehicle type with traci to create the box with correct dimensions
        width, length, _ = get_object_dimensions(box)

        # Create a polygon
        polygon = create_box(width, length,
                             traci.vehicle.getPosition(box)[0], traci.vehicle.getPosition(box)[1],
                             traci.vehicle.getAngle(box), center)
        # Add the polygon to the axis
        polygon.set_facecolor(color)
        ax.add_patch(polygon)

    for box in all_cyclist:
        color = get_color(detected_cyclists, fco_list, box)

        width, length, _ = get_object_dimensions(box)

        # Create a polygon
        polygon = create_box(width, length,
                             traci.vehicle.getPosition(box)[0], traci.vehicle.getPosition(box)[1],
                             traci.vehicle.getAngle(box), center)
        # Add the polygon to the axis
        polygon.set_facecolor(color)
        ax.add_patch(polygon)

    for box in all_pedestrians:
        # Create a polygon
        color = get_color(detected_pedestrians, fco_list, box)
        polygon = create_box(TRAFFIC['PERSON']['WIDTH'], TRAFFIC['PERSON']['LENGTH'],
                             traci.person.getPosition(box)[0], traci.person.getPosition(box)[1],
                             traci.person.getAngle(box), center)
        # Add the polygon to the axis
        polygon.set_facecolor(color)
        ax.add_patch(polygon)
    plt.xlim( - 100,  + 100)
    plt.ylim( - 100,  + 100)
    ax.set_aspect('equal')
    if show:
        plt.show()


    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{int(traci.simulation.getTime())}.png'))

    plt.close()


def get_color(detection_list: List[str], fco_list: List[str], id: str) -> str:
    if id in fco_list:
        color = 'yellow'
    elif id in detection_list:
        color = 'green'
    else:
        color = 'red'
    return color


def get_data_sample(dataset_path: str) -> (torch.Tensor, torch.Tensor):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    sample = dataset.sample(n=1).to_dict()
    img = list(sample['plot'].values())[0]
    vector = torch.tensor(list(sample['vector'].values())[0])
    return img, vector
