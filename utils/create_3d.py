import os

from numpy import ndarray
from sumolib import checkBinary
import traci
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
from sumo_sim.extract_poly_cords import filter_buildings, \
    extrude_points, interpolate_points, extract_building_coordinates, get_centroid
from configs.config_simulation import TRAFFIC, BUILDINGS
from configs.config_dataset import DATASET_GENERAL
from utils.mesh_to_pointcloud import mesh_to_pc
from typing import Tuple, Dict


def get_all_vehicles(path_to_xml: str) -> list:
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    vehicle_ids = [vehicle.get('id') for vehicle in root.iter('vehicle')]
    return vehicle_ids


def create_3d(ego, raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck,
              raw_bike, raw_person, raw_buildings, building_coordinates, distance,
              save=False, visualize = False):
    close = get_close(ego, distance)
    close_pedestrians = get_close_pedestrians(ego, distance)
    close_points = dict()
    ego_info = [traci.vehicle.getPosition(ego)[0], traci.vehicle.getPosition(ego)[1], traci.vehicle.getAngle(ego)]
    for c in close:
        if traci.vehicle.getVehicleClass(c) == 'bicycle':
            original_car = create_positioned_car(raw_bike, close[c])
            close_points[f'cyclist_{c}'] = relocate(original_car, ego_info)
        elif traci.vehicle.getVehicleClass(c) == 'delivery':
            original_car = create_positioned_car(raw_delivery_car, close[c])
            close_points[f'vehicle_{c}'] = relocate(original_car, ego_info)
        elif traci.vehicle.getVehicleClass(c) == 'truck':
            original_car = create_positioned_car(raw_truck, close[c])
            close_points[f'vehicle_{c}'] = relocate(original_car, ego_info)
        elif traci.vehicle.getVehicleClass(c) == 'bus':
            original_car = create_positioned_car(raw_bus, close[c])
            close_points[f'vehicle_{c}'] = relocate(original_car, ego_info)
        elif traci.vehicle.getVehicleClass(c) == 'passenger':
            if traci.vehicle.getShapeClass(c) == 'passenger/van':
                original_car = create_positioned_car(raw_large_car, close[c])
                close_points[f'vehicle_{c}'] = relocate(original_car, ego_info)
            else:
                original_car = create_positioned_car(raw_small_car, close[c])
                close_points[f'vehicle_{c}'] = relocate(original_car, ego_info)
        else:
            raise ValueError('Vehicle class not found')
    for p in close_pedestrians:
        original_pedestrian = create_positioned_car(raw_person, close_pedestrians[p])
        close_points[f'pedestrian_{p}'] = relocate(original_pedestrian, ego_info)
    close_buildings = list(filter_buildings(building_coordinates, ego_info[0], ego_info[1], 100).keys())
    for b in raw_buildings:
        if b in close_buildings:
            center_point = get_centroid(building_coordinates[b])
            original_building = create_positioned_building(raw_buildings[b], [0, 0, 0], center_point[0],
                                                           center_point[1])
            close_points[f'building_{b}'] = relocate(original_building, ego_info)
    # show the 3D points
    if visualize:
        points = np.vstack(close_points.values())
        points = np.array([points[:, 0], points[:, 1], points[:, 2]]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coord_frame])
    save_dict_poitns(close_points, ego)
    if save:
        save_dict_poitns(close_points, ego, path='test_3d')
    return close_points


def relocate(points, info):
    x_new = info[0]
    y_new = info[1]
    theta = np.radians(info[2])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Create the rotation matrix
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    # Define the translation vector
    t = np.array([x_new, y_new])

    # Apply the transformation
    points[:, :2] = np.dot(points[:, :2] - t, R.T)

    # Keep the third dimension unchanged
    points[:, 2] = points[:, 2]

    return points


def create_positioned_car(raw_car, info):
    x, y, angle = info
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Perform rotation and translation in a single step

    raw_car_arr = np.array(raw_car)
    points = np.empty_like(raw_car)
    points[:, 0] = raw_car_arr[:, 0] * cos_theta - raw_car_arr[:, 1] * sin_theta + x
    points[:, 1] = raw_car_arr[:, 0] * sin_theta + raw_car_arr[:, 1] * cos_theta + y
    points[:, 2] = raw_car_arr[:, 2]

    return points


def create_positioned_building(raw_building, info, p_x, p_y):
    x, y, angle = info
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Translate the points so that (p_x, p_y) becomes the origin
    translated_points = raw_building - np.array([p_x, p_y, 0])

    # Perform rotation around the origin
    rotated_points = np.empty_like(translated_points)
    rotated_points[:, 0] = translated_points[:, 0] * cos_theta - translated_points[:, 1] * sin_theta
    rotated_points[:, 1] = translated_points[:, 0] * sin_theta + translated_points[:, 1] * cos_theta
    rotated_points[:, 2] = translated_points[:, 2]

    # Translate the points back to their original position and apply the (x, y) translation
    points = rotated_points + np.array([p_x - x, p_y - y, 0])

    return points


def save_dict_poitns(points, ego, path=None):
    if path is None:
        dir = os.path.join('tmp_3d', f'{ego}_{int(traci.simulation.getTime())}')
    else:
        dir = os.path.join(path, f'{ego}_{int(traci.simulation.getTime())}')
    os.makedirs(dir, exist_ok=False)
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[p])
        o3d.io.write_point_cloud(os.path.join(dir, f'{p}.ply'), pcd)


def get_close(ego, distance):
    vehicles = traci.vehicle.getIDList()
    ego_pos = traci.vehicle.getPosition(ego)
    ego_angle = traci.vehicle.getAngle(ego)
    close = {}
    for v in vehicles:
        if v != ego:
            pos = traci.vehicle.getPosition(v)
            dist = np.sqrt((pos[0] - ego_pos[0]) ** 2 + (pos[1] - ego_pos[1]) ** 2)
            if dist < distance:
                close[v] = [pos[0], pos[1], -traci.vehicle.getAngle(v)]
                # print(f'vehcile: {v}, distance: {dist}, angle: {traci.vehicle.getAngle(v)}, angles: {ego_angle - traci.vehicle.getAngle(v)}, vector: {(pos[0] - ego_pos[0], pos[1] - ego_pos[1])}')
    return close


def get_close_pedestrians(ego, distance):
    pedestrians = traci.person.getIDList()
    ego_pos = traci.vehicle.getPosition(ego)
    close = {}
    for p in pedestrians:
        pos = traci.person.getPosition(p)
        dist = np.sqrt((pos[0] - ego_pos[0]) ** 2 + (pos[1] - ego_pos[1]) ** 2)
        if dist < distance:
            close[p] = [pos[0], pos[1], -traci.person.getAngle(p)]
    return close



def create_raw_box_representation(b: float, l: float, h: float, density: int, vis: bool = False, ) -> np.ndarray:
    """
    Creates boxes as simple representation for the traffic participants defined by their width, length and height.

    Args:
        w (float): The width of the box.
        l (float): The length of the box.
        h (float): The height of the box.
        vis (bool, optional): Flag to visualize the box representation. Defaults to False.

    Returns:
        np.ndarray: Numpy array containing the raw point cloud representation of the box.

    """
    # Define the half-length and half-height of the rectangle
    half_l = l / 2
    half_b = b / 2

    # Calculate the vertex points
    vertices = np.array([[-half_b, 0], [-half_b, -l], [half_b, -l], [half_b, 0]])
    # Translate the vertices to the center point
    org_vertices = vertices

    # Find the minimum and maximum x and y coordinates
    x_min, y_min = np.min(org_vertices, axis=0)
    x_max, y_max = np.max(org_vertices, axis=0)
    x_distance = x_max - x_min
    y_distance = y_max - y_min
    # Create arrays of x and y coordinates using linspace
    x = np.linspace(x_min, x_max, int(x_distance * density))
    y = np.linspace(y_min, y_max, int(y_distance * density))
    z = np.linspace(h, h, 1)

    # Create a grid of x and y coordinates using meshgrid
    X, Y, Z = np.meshgrid(x, y, z)

    # Stack X and Y horizontally to create an array of points
    points_ = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    points = list()
    for i in range(4):
        start = vertices[i]
        end = vertices[(i + 1) % 4]
        distance = max(int(np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)), 1)
        num_points = distance * density
        x_vals = np.linspace(start[0], end[0], num_points)
        # print(x_vals)
        y_vals = np.linspace(start[1], end[1], num_points)
        # print(y_vals)
        z_vals = np.zeros(num_points)
        # print(z_vals)
        points.append(np.stack((x_vals, y_vals, z_vals), axis=-1))
    points = np.vstack(points)

    # Create additional points in the z direction
    z_vals = np.linspace(0, h, int(h * density))
    points = np.repeat(points, int(h * density), axis=0)
    z_vals = np.tile(z_vals, int(len(points) / len(z_vals)))
    points[:, 2] = z_vals
    points = np.vstack([points, points_])

    if vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

    return points


def create_all_raw_participants() -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Generate raw point clouds for the different traffic participants depending on the representation type.

    Returns:
        Tuple: A tuple containing the raw point cloud or box representations of small cars, large cars,
               delivery vehicles, buses, trucks, bikes, and pedestrians.
    """
    if TRAFFIC['VEHICLE_REPRESENTATION']['SMALL_CAR'] == 'box':
        raw_small_car = create_raw_box_representation(
                                       density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                       l=TRAFFIC["SMALL_CAR"]["LENGTH"], b=TRAFFIC["SMALL_CAR"]["WIDTH"],
                                       h=TRAFFIC["SMALL_CAR"]["HEIGHT"])
    elif TRAFFIC['VEHICLE_REPRESENTATION']['SMALL_CAR'] == 'mesh':
        raw_small_car = mesh_to_pc(TRAFFIC['MESH_PATH']['SMALL_CAR'], DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                                   TRAFFIC['SMALL_CAR']['LENGTH'], [90, 0, 180])
        raw_small_car = np.array(raw_small_car.points)

    else:
        raise ValueError
    if TRAFFIC['VEHICLE_REPRESENTATION']['LARGE_CAR'] == 'box':
        raw_large_car = create_raw_box_representation(
                                       density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                       l=TRAFFIC["LARGE_CAR"]["LENGTH"], b=TRAFFIC["LARGE_CAR"]["WIDTH"],
                                       h=TRAFFIC["LARGE_CAR"]["HEIGHT"])
    elif TRAFFIC['VEHICLE_REPRESENTATION']['LARGE_CAR'] == 'mesh':
        raw_large_car = mesh_to_pc(TRAFFIC['MESH_PATH']['LARGE_CAR'], DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                                   TRAFFIC['LARGE_CAR']['LENGTH'], [90, 0, 180])
        raw_large_car = np.array(raw_large_car.points)
    else:
        raise ValueError

    if TRAFFIC['VEHICLE_REPRESENTATION']['DELIVERY'] == 'box':
        raw_delivery_car = create_raw_box_representation(
                                          density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                          l=TRAFFIC["DELIVERY"]["LENGTH"], b=TRAFFIC["DELIVERY"]["WIDTH"],
                                          h=TRAFFIC["DELIVERY"]["HEIGHT"])

    elif TRAFFIC['VEHICLE_REPRESENTATION']['DELIVERY'] == 'mesh':
        raw_delivery_car = mesh_to_pc(TRAFFIC['MESH_PATH']['DELIVERY'],
                                      DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                                      TRAFFIC['DELIVERY']['LENGTH'], [90, 0, 180])
        raw_delivery_car = np.array(raw_delivery_car.points)
    else:
        raise ValueError

    if TRAFFIC['VEHICLE_REPRESENTATION']['BUS'] == 'box':
        raw_bus = create_raw_box_representation(
                                 density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                 l=TRAFFIC["BUS"]["LENGTH"], b=TRAFFIC["BUS"]["WIDTH"],
                                 h=TRAFFIC["BUS"]["HEIGHT"])
    elif TRAFFIC['VEHICLE_REPRESENTATION']['BUS'] == 'mesh':
        raw_bus = mesh_to_pc(TRAFFIC['MESH_PATH']['BUS'], DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                             TRAFFIC['BUS']['LENGTH'], [90, 0, 180])
        raw_bus = np.array(raw_bus.points)
    else:
        raise ValueError

    if TRAFFIC['VEHICLE_REPRESENTATION']['TRUCK'] == 'box':
        raw_truck = create_raw_box_representation(
                                   density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                   l=TRAFFIC["TRUCK"]["LENGTH"], b=TRAFFIC["TRUCK"]["WIDTH"],
                                   h=TRAFFIC["TRUCK"]["HEIGHT"])
    elif TRAFFIC['VEHICLE_REPRESENTATION']['TRUCK'] == 'mesh':
        raw_truck = mesh_to_pc(TRAFFIC['MESH_PATH']['TRUCK'], DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                               TRAFFIC['TRUCK']['LENGTH'], [90, 0, 180])
        raw_truck = np.array(raw_truck.points)
    else:
        raise ValueError

    if TRAFFIC['VEHICLE_REPRESENTATION']['BIKE'] == 'box':
        raw_bike = create_raw_box_representation(
                                  density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                  l=TRAFFIC["BIKE"]["LENGTH"], b=TRAFFIC["BIKE"]["WIDTH"],
                                  h=TRAFFIC["BIKE"]["HEIGHT"])
    elif TRAFFIC['VEHICLE_REPRESENTATION']['BIKE'] == 'mesh':
        raw_bike = mesh_to_pc(TRAFFIC['MESH_PATH']['BIKE'], DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                              TRAFFIC['BIKE']['LENGTH'], [270, 0, 90])
        raw_bike = np.array(raw_bike.points)
    else:
        raise ValueError
    if TRAFFIC['VEHICLE_REPRESENTATION']['PERSON'] == 'box':
        raw_person = create_raw_box_representation(
                                    density=DATASET_GENERAL["TRAFFIC_PARTICIPANT_DENSITY"],
                                    l=TRAFFIC["PERSON"]["LENGTH"], b=TRAFFIC["PERSON"]["WIDTH"],
                                    h=TRAFFIC["PERSON"]["HEIGHT"])
    elif TRAFFIC['VEHICLE_REPRESENTATION']['PERSON'] == 'mesh':
        raw_person = mesh_to_pc(TRAFFIC['MESH_PATH']['PERSON'], DATASET_GENERAL['TRAFFIC_PARTICIPANT_DENSITY'],
                                TRAFFIC['PERSON']['HEIGHT'],
                                [90, 0, 180])  # use height here since it is scaled on the largest dimension
        raw_person = np.array(raw_person.points)
    else:
        raise ValueError

    return raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck, raw_bike, raw_person


def create_all_raw_buildings() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Create raw representations of buildings.

    Returns:
        Tuple: A tuple containing dictionaries of raw 3D buildings and raw building coordinates.
    """
    raw_buildings = extract_building_coordinates(os.path.join('sumo_sim', BUILDINGS["POLY_FILE"]))
    raw_3d_buildings = {}

    for building in raw_buildings:
        interpolated_points = interpolate_points(raw_buildings[building], density=BUILDINGS["DENSITY"])
        points_3d = extrude_points(interpolated_points, z_start=0, z_end=BUILDINGS["HEIGHT"], density=BUILDINGS["DENSITY"])
        raw_3d_buildings[building] = points_3d

    return raw_3d_buildings, raw_buildings

if __name__ == "__main__":
    create_raw_box_representation(1.5, 5, 2, 100, vis=True)
