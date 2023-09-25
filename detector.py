import os
import pickle
import time
from typing import List, Tuple, Any, Union
import math

import numpy
import pandas as pd

import torch
import traci
from PIL import Image
from matplotlib import pyplot as plt

from configs.config_cv import DEFAULT, CAMERAS
from utils.detect_cv import run_computer_vision_approach, split_detected_from_undetected_objects
from utils.utils_cv import create_camera_matrices_from_config
from vit_pytorch import ViT
import importlib
import PIL

from utils.detect_cv import *
from utils.file_utils import delete_all_items_in_dir
from utils.create_3d import create_3d, create_all_raw_participants, create_all_raw_buildings
from utils.create_box import plot_boxes, get_object_dimensions, create_nn_input, get_tmp_image_tensor
from utils.create_cnn import CustomResNet
from utils.visualize import visualize_frame


class Detector:
    def __init__(self, mode: str, model_path: Union[None, str] = None, camera_settings=None):
        assert(mode in ['cv', 'nn'])
        self.mode = mode  # either cv or nn
        # import the dict with the model settings from model_path
        self.model_path = model_path  # path to the trained model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.camera_settings = camera_settings
        self._3d_distance = 55
        if mode == 'cv':
            self.raw_small_car, self.raw_large_car, self.raw_delivery_car, \
            self.raw_bus, self.raw_truck, self.raw_bike, self.raw_person \
            = create_all_raw_participants()
            self.raw_3d_buildings, self.raw_buildings = create_all_raw_buildings()


        if self.mode == 'nn':
            if self.model_path is None:
                raise ValueError('path to trained model needs to be specified')
            NETWORK_CONFIG = importlib.import_module(f"models.{model_path}.configs.config_networks").NETWORK_CONFIG
            VIT_CONFIG = importlib.import_module(f"models.{model_path}.configs.config_networks").VIT_CONFIG
            RESNET_CONFIG = importlib.import_module(f"models.{model_path}.configs.config_networks").RESNET_CONFIG
            DATASET_GENERAL = importlib.import_module(f"models.{model_path}.configs.config_dataset").DATASET_GENERAL
            if NETWORK_CONFIG['NETWORK_TYPE'] == 'ViT':
                self.model = ViT(**VIT_CONFIG, image_size=DATASET_GENERAL['BEV_IMAGE_SIZE'])
                self.model.load_state_dict(torch.load(os.path.join('models', self.model_path, 'model_state_dict.pt'), map_location=self.device))
            elif NETWORK_CONFIG['NETWORK_TYPE'] == 'RESNET':
                self.model = CustomResNet(**RESNET_CONFIG, image_size=DATASET_GENERAL['BEV_IMAGE_SIZE'])
                self.model.load_state_dict(torch.load(os.path.join('models', self.model_path, 'model_state_dict.pt'), map_location=self.device))
            else:
                raise NotImplementedError
            self.model.eval()
            self.model.to(self.device)
            self.image_size = DATASET_GENERAL['BEV_IMAGE_SIZE']

    def detect(self, vehicle_id: str, vehicle_dict: dict = None, pedestrians_dict: dict = None,):
        if self.mode == 'cv':
            return self._detect_cv(vehicle_id)
        elif self.mode == 'nn':
            # print(vehicle_dict)
            return self._detect_nn(vehicle_id, vehicle_dict = vehicle_dict, pedestrians_dict = pedestrians_dict)
        else:
            raise NotImplementedError

    def _detect_cv(self, vehicle_id):
        points = create_3d(vehicle_id, self.raw_small_car, self.raw_large_car, self.raw_delivery_car, self.raw_bus,
                           self.raw_truck,
                           self.raw_bike, self.raw_person, self.raw_3d_buildings, self.raw_buildings,
                           self._3d_distance, save=False)
        total_points = 0
        for obj in points:
            total_points += points[obj].shape[0]
        all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, \
            detected_cyclists, detected_pedestrians = self._execute_cv()

        return all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians

    def _execute_cv(self):

        ply_dir = 'tmp_3d'

        # load config
        default_settings = DEFAULT
        radius = default_settings['detection_range']
        objects_to_detect = default_settings['objects_to_detect']
        save_depth_image_dir = default_settings['save_depth_image_dir']

        cameras = create_camera_matrices_from_config(CAMERAS)

        visualize = False

        # Iterates all frames from ply_dir
        for parent_folder, image_plane_objects_detection_states, image_plane_objects_detection_occupancy, obj_pt_clouds, objects_image_masks, objects_image_disparity in run_computer_vision_approach(
                ply_dir=ply_dir,
                cameras=cameras,
                scan_radius=radius,
                objects_to_detect=objects_to_detect,
                save_depth_image_dir=save_depth_image_dir
        ):
            all_objects_detection_state = {cam_pos: {obj_type: {
                filename: False for filename in obj_pt_clouds[obj_type] if filename.startswith(obj_type)
            }
                for obj_type in objects_to_detect if obj_type in obj_pt_clouds
            } for cam_pos in image_plane_objects_detection_states
            }

            for cam_pos in image_plane_objects_detection_states:
                all_objects_detection_state[cam_pos].update(image_plane_objects_detection_states[cam_pos])

            detected_objects, other_objects = split_detected_from_undetected_objects(
                image_plane_objects_detection_states=image_plane_objects_detection_states,
                obj_pt_clouds=obj_pt_clouds,
                cameras=cameras,
            )
            # print(f'detected_objects: {detected_objects}')
            # print(f'other_objects: {other_objects}')

            # run the detection visualization
            if visualize:
                visualize_frame(detected_objects, other_objects, obj_pt_clouds, cameras, radius,
                                objects_image_masks,
                                plot_image_plane_rays=True, plot_scan_radius=True)
            # Add the detected vehicles to the data
            delete_all_items_in_dir('tmp_3d')
            all_vehicles = detected_objects.get('vehicle', []) + other_objects.get('vehicle', [])
            all_vehicles = [vehicle[vehicle.find('_') + 1: vehicle.rfind('.')] for vehicle in all_vehicles]
            detected_vehicles = detected_objects.get('vehicle', [])
            detected_vehicles = [vehicle[vehicle.find('_') + 1: vehicle.rfind('.')] for vehicle in detected_vehicles]

            # Add cyclists in the scene to the data
            all_cyclists = detected_objects.get('cyclist', []) + other_objects.get('cyclist', [])
            all_cyclists = [cyclist[cyclist.find('_') + 1: cyclist.rfind('.')] for cyclist in all_cyclists]
            detected_cyclists = detected_objects.get('cyclist', [])
            detected_cyclists = [cyclist[cyclist.find('_') + 1: cyclist.rfind('.')] for cyclist in detected_cyclists]

            # Add pedestrians in the scene to the data
            all_pedestrians = detected_objects.get('pedestrian', []) + other_objects.get('pedestrian', [])
            all_pedestrians = [pedestrian[pedestrian.find('_') + 1: pedestrian.rfind('.')] for pedestrian in
                               all_pedestrians]
            detected_pedestrians = detected_objects.get('pedestrian', [])
            detected_pedestrians = [pedestrian[pedestrian.find('_') + 1: pedestrian.rfind('.')] for pedestrian in
                                    detected_pedestrians]

            return all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians, occupancy_vehicles, occupancy_cyclists, occupancy_pedestrians

    def _detect_nn(self, vehicle_id: str, image_type: str = 'box', det_radius: int = 50, vehicle_dict: bool = None, pedestrians_dict: bool = None) -> Tuple[
        List[Any], List[Any], List[Any], List[str], List[str], List[str]]:
        if vehicle_dict is None:
            vehicles = traci.vehicle.getIDList()
            ego_pos = list(traci.vehicle.getPosition(vehicle_id))
            # append the angle of the vehicle to the ego_pos tuple
            ego_pos.append(traci.vehicle.getAngle(vehicle_id))
        else:
            vehicles = vehicle_dict
            ego_pos = list(vehicles[vehicle_id]['pos'])
            ego_pos.append(vehicles[vehicle_id]['angle'])
        if pedestrians_dict is None:
            pedestrians = traci.person.getIDList()
        else:
            pedestrians = pedestrians_dict
        # create image of current fco and return the image tensor
        current_data = {}
        if image_type == 'box':
            radius_vehicles, radius_pedestrians, tmp_filename = create_nn_input(vehicle_id, vehicles, pedestrians, ego_pos,
                                                                  det_radius, self.raw_buildings)
            image_tensor = get_tmp_image_tensor(tmp_filename, self.image_size)
        else:
            raise NotImplementedError
        for vehicle in radius_vehicles:
            if vehicle_dict is None:
                pos = traci.vehicle.getPosition(vehicle)
                type = traci.vehicle.getTypeID(vehicle)
            else:
                pos = vehicles[vehicle]['pos']
                type = vehicles[vehicle]['type']
            current_data[vehicle] = {'type': type,
                                     'vector': [pos[0] - ego_pos[0],
                                                pos[1] - ego_pos[1]],
                                     'plot': image_tensor}
        for pedestrian in radius_pedestrians:
            if pedestrians_dict is None:
                pos = traci.person.getPosition(pedestrian)
            else:
                pos = pedestrians[pedestrian]['pos']
            current_data[pedestrian] = {'type': 'pedestrian',
                                        'vector': [pos[0] - ego_pos[0],
                                                   pos[1] - ego_pos[1]],
                                        'plot': image_tensor}

        detected_objects = self._execute_nn(current_data)
        # assign all objects and the detected objects to the corresponding categories
        vehicle_objects = ['small_car', 'large_car', 'bus', 'truck', 'delivery']
        all_vehicles = [vehicle for vehicle in radius_vehicles if current_data[vehicle]['type'] in vehicle_objects]
        detected_vehicles = [vehicle for vehicle in detected_objects if
                             vehicle in radius_vehicles and current_data[vehicle]['type'] in vehicle_objects]
        all_cyclists = [vehicle for vehicle in radius_vehicles if current_data[vehicle]['type'] == 'cyclist']
        detected_cyclists = [vehicle for vehicle in detected_objects if
                             vehicle in radius_vehicles and current_data[vehicle]['type'] == 'cyclist']
        all_pedestrians = [pedestrian for pedestrian in radius_pedestrians]
        detected_pedestrians = [pedestrian for pedestrian in detected_objects if pedestrian in radius_pedestrians]
        #print(all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians)
        return all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians

    def _execute_nn(self, data_dict: dict) -> List[str]:
        images = torch.stack([entry['plot'] for entry in data_dict.values()]).float().to(self.device)
        vectors = torch.tensor([entry['vector'] for entry in data_dict.values()]).float().to(self.device)
        keys = [key for key in data_dict.keys()]
        with torch.no_grad():
            outputs = self.model(images, vectors)
        # map outputs to 0 or 1 not detected or detected
        detections = torch.where(outputs.cpu() >= 0.5, torch.tensor(1), torch.tensor(0))
        # get the keys of the detected objects

        detected_objects = [keys[i] for i, detection in enumerate(detections) if detection == 1]
        return detected_objects


class DatasetGenerator:
    def __init__(self, filename: str, raw_buildings: np.array, bev_image_size: int,
                 bev_image_type: str, radius: int, gui: bool,
                  show_3d: bool = False, show_bev: bool = False):
        self.data = {}
        self.num_detections = 1
        self.v_appended = list()
        self.filename = filename
        self.raw_buildings = raw_buildings
        self.show_3D = show_3d
        self.bev_image_size = bev_image_size
        self.bev_image_type = bev_image_type
        self.radius = radius
        self.gui = gui
        self.show_bev = show_bev
        self.tmp_filename = 'image'

    def get_data(self, ego):
        # Store the current screenshot to the objects that were in the scene
        for v in self.v_appended:
            self._store_old_image(v)

        # Reset the list to get the new objects that are in the scene
        self.v_appended = list()

        # Remove the old screenshot
        if os.path.exists('tmp/image.jpg'):
            os.remove('tmp/image.jpg')
        
        if os.path.exists(f'tmp/{self.tmp_filename}'):
            os.remove(f'tmp/{self.tmp_filename}')

        # Get current simulation information
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        simtime = traci.simulation.getTime()
        ego_pos = [traci.vehicle.getPosition(ego)[0], traci.vehicle.getPosition(ego)[1], traci.vehicle.getAngle(ego)]

        # Track the ego vehicle and take screenshot (image gets saved after next simulation.step())
        if self.gui:
            traci.gui.setOffset('View #0', ego_pos[0], ego_pos[1])
            traci.gui.setZoom('View #0', 750)
        if self.bev_image_type == "screenshot":
            traci.gui.screenshot('View #0', os.path.join('tmp', f'image.jpg'))

        elif self.bev_image_type == "box":
            # get the vehicles within the 3d detection range and extract their position information
            _, _, self.tmp_filename = create_nn_input(ego, vehicles, pedestrians, ego_pos, self.radius, self.raw_buildings)

        else:
            raise NotImplementedError

        # Detect the objects with the 3D CV approach
        t = time.time()
        ply_dir = 'tmp_3d'

        # load config
        default_settings = DEFAULT
        radius = default_settings['detection_range']
        objects_to_detect = default_settings['objects_to_detect']
        save_depth_image_dir = default_settings['save_depth_image_dir']

        cameras = create_camera_matrices_from_config(CAMERAS)
        print('starting cv approach')
        # Iterates all frames from ply_dir
        for parent_folder, image_plane_objects_detection_states, image_plane_objects_detection_occupancy, obj_pt_clouds, objects_image_masks, objects_image_disparity in run_computer_vision_approach(
                ply_dir=ply_dir,
                cameras=cameras,
                scan_radius=radius,
                objects_to_detect=objects_to_detect,
                save_depth_image_dir=save_depth_image_dir
        ):  
            print(image_plane_objects_detection_states)
            all_objects_detection_state = {cam_pos: {obj_type: {
                filename: False for filename in obj_pt_clouds[obj_type] if filename.startswith(obj_type)
            }
                for obj_type in objects_to_detect if obj_type in obj_pt_clouds
            } for cam_pos in image_plane_objects_detection_states
            }

            for cam_pos in image_plane_objects_detection_states:
                all_objects_detection_state[cam_pos].update(image_plane_objects_detection_states[cam_pos])

            detected_objects, other_objects = split_detected_from_undetected_objects(
                image_plane_objects_detection_states=image_plane_objects_detection_states,
                obj_pt_clouds=obj_pt_clouds,
                cameras=cameras,
            )
            occupancy_objects = get_object_max_occupancy(image_plane_objects_detection_occupancy, obj_pt_clouds, ego, int(simtime))

            # run the detection visualization
            if self.show_3D:
                visualize_frame(detected_objects, other_objects, obj_pt_clouds, cameras, radius,
                                objects_image_masks,
                                plot_image_plane_rays=True, plot_scan_radius=True)
        'objects detection done'
        # Add the detected vehicles to the data
        delete_all_items_in_dir('tmp_3d')
        all_vehicles = detected_objects.get('vehicle', []) + other_objects.get('vehicle', [])
        print(f'in_scene_v: {all_vehicles}')
        print(f'detected_v: {detected_objects.get("vehicle", [])}')
        for v in all_vehicles:
            vehicle = v[v.find('vehicle_') + len('vehicle_'):v.find('.ply')]  # extract vehicle id from filename
            v_name = f'ego_{ego}_vehicle_{vehicle}_{int(simtime)}'
            self.data[v_name] = {}
            vector = [traci.vehicle.getPosition(vehicle)[0] - ego_pos[0],
                                           traci.vehicle.getPosition(vehicle)[1] - ego_pos[1]]
            if self.bev_image_type != "screenshot":
                # rotate the vector to the ego vehicle's orientation
                ego_angle = ego_pos[2]
                theta = math.radians(ego_angle) # convert to radians (sumo uses degrees in clockwise direction)
                vector = [vector[0] * math.cos(theta) - vector[1] * math.sin(theta),
                            vector[0] * math.sin(theta) + vector[1] * math.cos(theta)]
            self.data[v_name]['vector'] = vector
            self.data[v_name]['detected'] = 1 if v in detected_objects['vehicle'] else 0
            self.v_appended.append(v_name)

        # Add cyclists in the scene to the data
        all_cyclists = detected_objects.get('cyclist', []) + other_objects.get('cyclist', [])
        print(f'in_scene_c: {all_cyclists}')
        print(f'detected_c: {detected_objects.get("cyclist", [])}')
        for c in all_cyclists:
            cyclist = c[c.find('cyclist_') + len('cyclist_'):c.find('.ply')]
            c_name = f'ego_{ego}_cyclist_{cyclist}_{int(simtime)}'
            self.data[c_name] = {}
            self.data[c_name]['vector'] = [traci.vehicle.getPosition(cyclist)[0] - ego_pos[0],
                                           traci.vehicle.getPosition(cyclist)[1] - ego_pos[1]]
            self.data[c_name]['detected'] = 1 if c in detected_objects['cyclist'] else 0
            self.v_appended.append(c_name)

        # Add pedestrians in the scene to the data
        all_pedestrians = detected_objects.get('pedestrian', []) + other_objects.get('pedestrian', [])
        #print(f'in_scene_p: {all_pedestrians}')
        #print(f'detected_p: {detected_objects.get("pedestrian", [])}')
        for p in all_pedestrians:
            pedestrian = p[p.find('pedestrian_') + len('pedestrian_'):p.find('.ply')]
            p_name = f'ego_{ego}_pedestrian_{pedestrian}_{int(simtime)}'
            self.data[p_name] = {}
            self.data[p_name]['vector'] = [traci.person.getPosition(pedestrian)[0] - ego_pos[0],
                                           traci.person.getPosition(pedestrian)[1] - ego_pos[1]]
            self.data[p_name]['detected'] = 1 if p in detected_objects['pedestrian'] else 0
            self.v_appended.append(p_name)
        # add the occupation information
        for obj in self.v_appended:
            if obj in occupancy_objects:
                self.data[obj]['occupancy'] = occupancy_objects[obj]
            else: 
                self.data[obj]['occupancy'] = 0
            if self.data[obj]['occupancy'] == 0 and self.data[obj]['detected'] == 1:
                raise ValueError(f'occupancy is 0 but detected is 1 for {obj}')
        print(f'3d detection took {time.time() - t} seconds')


    def _get_closest(self, vehicles, ego, ego_pos):
        distances = {}
        for v in vehicles:
            if v != ego:
                v_pos = traci.vehicle.getPosition(v)
                distance = ((v_pos[0] - ego_pos[0]) ** 2 + (v_pos[1] - ego_pos[1]) ** 2) ** 0.5
                distances[v] = distance
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        closest = sorted_distances[:self.num_detections]
        return closest

    def _in_image(self, ego: str, close_vehicle: tuple) -> bool:
        b = traci.gui.getBoundary('View #0')  # get the boundary of the view
        # make boundary to be a rectangle
        height = b[1][1] - b[0][1]  # calculate the height of the rectangle
        width = b[1][0] - b[0][0]  # calculate the width of the rectangle
        b = [[b[0][0] + (width - height) / 2, b[0][1]], [b[1][0] - (width - height) / 2, b[1][1]]]
        p = traci.vehicle.getPosition(close_vehicle)
        if b[0][0] < p[0] < b[1][0] and b[0][1] < p[1] < b[1][1]:
            return True
        else:
            return False

    def _store_old_image(self, v):
        # check if img exists if it does not exist wait for it 
        #while not os.path.exists('tmp/image.jpg'):
        #    print('waiting for image')
        #    time.sleep(0.1)
        #img = Image.open('tmp/image.jpg')  # Open the image
        retries = 5
        delay = 0.5
        for attempt in range(retries):
            try:
                img_path = os.path.join('tmp', self.tmp_filename)
                img = Image.open(img_path)
                # Continue processing
                break
            except PIL.UnidentifiedImageError:
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait for a while before retrying
                else:
                    raise
        if self.bev_image_type == 'screenshot':
            # Get the dimensions of the original image
            width, height = img.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img = cropped_img.crop((cropped_img.width // 2 - 500, cropped_img.height // 2 - 500,
                                            cropped_img.width // 2 + 500, cropped_img.height // 2 + 500))
            cropped_img = cropped_img.resize((self.bev_image_size, self.bev_image_size))
            # show the image
            if self.show_bev:
                cropped_img.show()
            tensor = torch.from_numpy(np.array(cropped_img))  # Convert the image to a numpy array
            tensor = tensor.permute(2, 0, 1)  # Permute the dimensions to match the expected format for PyTorch tensors
            # print(self.data[f'{ego}_{int(simtime)-1}'])
            self.data[v]['image'] = tensor
        elif self.bev_image_type == 'box':
            #convert image to torch tensor with one channel
            img = img.convert('L')
            img = img.resize((self.bev_image_size, self.bev_image_size))

            ''' show the vector on the image
            x_center, y_center = img.size[0]/2, img.size[1]/2
            mp_ratio = 0.25
            vector = self.data[v]['vector']
            scaled_vector = vector/mp_ratio
            end_x, end_y = (x_center + scaled_vector[0]), (y_center - scaled_vector[1])
            draw = PIL.ImageDraw.Draw(img)
            draw.line((x_center, y_center, end_x, end_y), fill="red", width=3)
            img.save(f'img_rec_arr_{v}.jpg') 
            '''


            tensor = torch.from_numpy(np.array(img))  # Convert the image to a numpy array
            tensor = tensor.unsqueeze(0)
            self.data[v]['plot'] = tensor
            # plot the tensor to check if it is correct
            plt.imshow(tensor.squeeze(0), cmap='gray')
            if self.show_bev:
                plt.show()


    def store_data(self, f):
        # convert dictionary to pandas dataframe
        df = pd.DataFrame(self.data)
        df = df.T
        filename = self.filename
        filename = os.path.join('data', filename, f'{filename}_{f}.pkl')

        # check if the file already exists
        if os.path.isfile(filename):
            # open the file in append mode
            with open(filename, "ab") as file:
                pickle.dump(df, file)


        else:
            # save the dataframe to a new file
            with open(filename, 'wb') as file:
                pickle.dump(df, file)

    def store_data_csv(self):
        # convert dictionary to pandas dataframe
        df = pd.DataFrame(self.data)
        df = df.T

        # check if the file already exists
        if os.path.isfile(self.filename):
            # open the file in append mode
            with open(self.filename, "a") as file:
                df.to_csv(file, header=False, index=False)
        else:
            # create the file and write the header and data
            with open(self.filename, "w") as file:
                df.to_csv(file, header=True, index=False)
