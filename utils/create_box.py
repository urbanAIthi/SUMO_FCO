import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
from PIL import Image
import torch
import os
from configs.config_simulation import TRAFFIC
import traci
from sumo_sim.extract_poly_cords import filter_buildings
from typing import List, Tuple
import time
import uuid


def create_box(width, length, x_pos, y_pos, angle, ego_info):
    ego_x, ego_y, ego_angle = ego_info

    # Calculate center_point
    center_point = np.array([0, length / 2])

    # Define vertices of the box
    vertices = np.array([
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, -length],
        [-width / 2, -length]
    ])

    # Calculate angle in radians and adjust with ego's angle
    angle_rad = np.radians(-angle)

    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Rotate vertices using rotation matrix
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    rotated_center_point = np.dot(center_point, rotation_matrix.T)

    # Translate rotated vertices by relative position of the box to the ego vehicle
    global_vertices = rotated_vertices + np.array([x_pos, y_pos])
    #print(f'global_vertices: {global_vertices}')

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # rotation matrix

    t = np.array([ego_x, ego_y])  # translation vector

    # Apply transformation
    global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

    # Log the final position of the vertices
    #print(f'global_vertices: {global_vertices}')

    # Create a polygon using the final vertices
    polygon = patches.Polygon(global_vertices, closed=True, fill=True, edgecolor='black', facecolor='black')

    return polygon


def create_building(global_vertices, ego_info):
    ego_x, ego_y, ego_angle = ego_info

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)

    # Create the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Define the translation vector
    t = np.array([ego_x, ego_y])

    # Convert global_vertices to numpy array for operations
    global_vertices = np.array(global_vertices)

    # Apply the transformation
    global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

    # Create a polygon using the final vertices
    polygon = patches.Polygon(global_vertices, closed=True, fill=True, edgecolor='black', facecolor='black')

    return polygon

def get_object_dimensions(vehicle_id):
    vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)
    if vehicle_type == 'passenger':
        if vehicle_type == 'passenger/van':
            width = TRAFFIC['LARGE_CAR']['WIDTH']
            length = TRAFFIC['LARGE_CAR']['LENGTH']
        else:
            width = TRAFFIC['SMALL_CAR']['WIDTH']
            length = TRAFFIC['SMALL_CAR']['LENGTH']
    elif vehicle_type == 'truck':
        width = TRAFFIC['TRUCK']['WIDTH']
        length = TRAFFIC['TRUCK']['LENGTH']
    elif vehicle_type == 'bus':
        width = TRAFFIC['BUS']['WIDTH']
        length = TRAFFIC['BUS']['LENGTH']
    elif vehicle_type == 'bicycle':
        width = TRAFFIC['BIKE']['WIDTH']
        length = TRAFFIC['BIKE']['LENGTH']
    elif vehicle_type == 'delivery':
        width = TRAFFIC['DELIVERY']['WIDTH']
        length = TRAFFIC['DELIVERY']['LENGTH']
    else:
        raise ValueError(f'Vehicle type {vehicle_type} not recognized')

    return width, length, None

def plot_vehicles_buildings(vehicles, pedestrians, buildings, ego_info, radius):
    fig, ax = plt.subplots()
    for box in vehicles:
        #get the vehicle type with traci to create the box with correct dimensions
        width, length, _ = get_object_dimensions(box)

        # Create a polygon
        polygon = create_box(width, length,
                             vehicles[box][0], vehicles[box][1],
                             vehicles[box][2], ego_info)
        # Add the polygon to the axis
        ax.add_patch(polygon)

    for box in pedestrians:
        # Create a polygon
        polygon = create_box(TRAFFIC['PERSON']['WIDTH'], TRAFFIC['PERSON']['LENGTH'],
                             pedestrians[box][0], pedestrians[box][1],
                             pedestrians[box][2], ego_info)
        # Add the polygon to the axis
        ax.add_patch(polygon)

    for building in buildings:
        polygon = create_building(buildings[building], ego_info)
        ax.add_patch(polygon)

        # Set the limits of the plot based on the coordinates of the points
        # Adjust the values according to your needs
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Hide the axis
    ax.axis('off')
    # Show the plot
    # create tmp folder if it does not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    filename = f'image_{uuid.uuid4().hex}.jpg'
    plt.savefig(os.path.join('tmp', filename), bbox_inches='tight', pad_inches=0)
    #print(f'save time: {time.time() - t}')

    plt.close(fig)

    return filename

def plot_boxes(vehicles: dict, radius: int, ego_info: List[float], target_size: int = 400) -> torch.Tensor:
    # Create a figure and axis
    fig, ax = plt.subplots()

    for box in vehicles:
        # Create a polygon
        polygon = create_box(vehicles[box]['width'], vehicles[box]['length'],
                             vehicles[box]['position'][0], vehicles[box]['position'][1],
                             vehicles[box]['angle'], ego_info)
        # Add the polygon to the axis
        ax.add_patch(polygon)

    # Set the limits of the plot based on the coordinates of the points
    # Adjust the values according to your needs
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Hide the axis
    ax.axis('off')

    # Show the plot
    # plt.show()
    plt.savefig('tmp/tmp.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Load the saved image
    image = Image.open('tmp/tmp.png')

    # Convert the image to grayscale (single channel)
    gray_image = image.convert('L')

    # Resize the image to 400x400
    resized_image = gray_image.resize((target_size, target_size))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Convert the numpy array to a Torch tensor
    tensor_image = torch.from_numpy(image_array)

    # Add an additional dimension to represent the single channel
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def create_nn_input(ego: str, vehicles: List[str], pedestrians: List[str], ego_pos: List[float], radius: float, raw_buildings, vehicle_dict = None, pedestrians_dict = None) -> Tuple[dict, dict]:
    t = time.time()
    radius_vehicles = {}
    radius_pedestrians = {}
    for v in vehicles:
        if vehicle_dict is None:
            v_pos = traci.vehicle.getPosition(v)
            v_angle = traci.vehicle.getAngle(v)
        else:
            v_pos = vehicle_dict[v]['position']
            v_angle = vehicle_dict[v]['angle']
        # check if i vehicle is within radius
        if np.linalg.norm(np.array(v_pos[0:1]) - np.array(ego_pos[0:1])) < radius:
            radius_vehicles[v] = [v_pos[0], v_pos[1], v_angle]
    for p in pedestrians:
        if pedestrians_dict is None:
            p_pos = traci.person.getPosition(p)
            p_angle = traci.person.getAngle(p)
        else:
            p_pos = pedestrians_dict[p]['position']
            p_angle = pedestrians_dict[p]['angle']
        # check if i vehicle is within radius
        if np.linalg.norm(np.array(p_pos) - np.array(ego_pos)) < radius:
            radius_pedestrians[p] = [p_pos[0], p_pos[1], p_angle]
    # get the relevant buildings
    close_buildings = list(filter_buildings(raw_buildings, ego_pos[0], ego_pos[1], 100))
    c_buildings = {}
    for b in raw_buildings:
        if b in close_buildings:
            c_buildings[b] = raw_buildings[b]
    tmp_filename = plot_vehicles_buildings(radius_vehicles, radius_pedestrians, c_buildings,
                            [ego_pos[0], ego_pos[1], ego_pos[2]], radius)

    return radius_vehicles, radius_pedestrians, tmp_filename

def get_tmp_image_tensor(tmp_filename: str = 'image.jpg', image_size: int = 400):
    img = Image.open(f'tmp/{tmp_filename}')
    img = img.convert('L')
    img = img.resize((image_size, image_size))
    tensor = torch.from_numpy(np.array(img))
    tensor = tensor.unsqueeze(0)
    # delete the tmp file
    os.remove(f'tmp/{tmp_filename}')
    return tensor
