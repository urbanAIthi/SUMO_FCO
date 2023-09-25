import traci
from PIL import Image
import torch
import pandas as pd
from utils.detect_cv import *
from config import DEFAULT, CAMERAS
#from train_det import draw_circle_at_position
import time
from utils.create_3d import create_3d, create_all_raw_participants, create_all_raw_buildings
from generate_dataset import configure_sumo, get_all_vehicles
from utils.file_utils import delete_all_items_in_dir
from utils.create_cnn import create_cnn
from vit_pytorch import ViT
from torch.utils.data import Dataset


def detect():
    ply_dir = '../tmp_3d'

    # load config
    default_settings = DEFAULT
    radius = default_settings['detection_range']
    objects_to_detect = default_settings['objects_to_detect']
    save_depth_image_dir = default_settings['save_depth_image_dir']

    cameras = create_camera_matrices_from_config(CAMERAS)

    # Iterates all frames from ply_dir
    for parent_folder, image_plane_objects_detection_states, obj_pt_clouds, objects_image_masks, objects_image_disparity in run_computer_vision_approach(
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

        # run the detection visualization
        if False:
            visualize_frame(detected_objects, other_objects, obj_pt_clouds, cameras, radius,
                            objects_image_masks,
                            plot_image_plane_rays=True, plot_scan_radius=True)
        traci.simulationStep()
        # Add the detected vehicles to the data
        delete_all_items_in_dir('tmp_3d')
        all_vehicles = detected_objects.get('vehicle', []) + other_objects.get('vehicle', [])
        print(f'in_scene_v: {all_vehicles}')
        print(f'detected_v: {detected_objects.get("vehicle", [])}')

        # Add cyclists in the scene to the data
        all_cyclists = detected_objects.get('cyclist', []) + other_objects.get('cyclist', [])
        print(f'in_scene_c: {all_cyclists}')
        print(f'detected_c: {detected_objects.get("cyclist", [])}')

        # Add pedestrians in the scene to the data
        all_pedestrians = detected_objects.get('pedestrian', []) + other_objects.get('pedestrian', [])
        print(f'in_scene_p: {all_pedestrians}')
        print(f'detected_p: {detected_objects.get("pedestrian", [])}')

        return all_vehicles, all_cyclists, all_pedestrians, detected_objects.get("vehicle", []), detected_objects.get("cyclist", []), detected_objects.get("pedestrian", [])

class Data:
    def __init__(self):
        self.data = {}
        self.num_detections = 1
        self.v_appended = list()

    def _store_old_image(self, v):
        img = Image.open('image.jpg')  # Open the image
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
        cropped_img = cropped_img.resize((400, 400))
        tensor = torch.from_numpy(np.array(cropped_img))  # Convert the image to a numpy array
        tensor = tensor.permute(2, 0, 1)  # Permute the dimensions to match the expected format for PyTorch tensors
        # print(self.data[f'{ego}_{int(simtime)-1}'])
        self.data[v]['image'] = tensor

    def get_data(self, ego):
        #if os.path.exists('image.jpg'):
         #   os.remove('image.jpg')
        ego_pos = traci.vehicle.getPosition(ego)
        vehicles = traci.vehicle.getIDList()
        for v in vehicles:
            if v != ego:
                pos = traci.vehicle.getPosition(v)
                dist = np.sqrt((pos[0] - ego_pos[0]) ** 2 + (pos[1] - ego_pos[1]) ** 2)
                if dist < 55:
                    self.data[v] = {}
                    self.data[v]['vector'] = torch.tensor([pos[0] - ego_pos[0],
                                           pos[1] - ego_pos[1]])
        pedestrians = traci.person.getIDList()
        for p in pedestrians:
            pos = traci.person.getPosition(p)
            dist = np.sqrt((pos[0] - ego_pos[0]) ** 2 + (pos[1] - ego_pos[1]) ** 2)
            if dist < 55:
                self.data[p] = {}
                self.data[p]['vector'] = torch.tensor([pos[0] - ego_pos[0],
                                                       pos[1] - ego_pos[1]])
        traci.gui.screenshot('View #0', 'image.jpg')
        traci.simulationStep()
        for v in self.data:
            self._store_old_image(v)
        return self.data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        k = list(sample.keys())[0]
        image = sample[k]['image'].type(torch.float32)
        vector = sample[k]['vector'].type(torch.float32)
        return image, vector



if __name__ == "__main__":
    method = "cv" # cv or nn
    model = "resnet"
    model_path = None
    traffic = 'multimodal'
    cameras = '1'
    poly_file = 'buildings_a9.poly.xml'

    if model == 'vit':
        model = ViT(image_size=400, patch_size=50, num_classes=1, channels=3, mlp_dim=2048, dropout=0.5, emb_dropout=0, dim=512, depth = 6, heads=8)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
    if model == 'resnet':
        model = create_cnn()
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model.load_state_dict(torch.load("best_valacc_model.pth"))
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    t_nn = time.time()
    img = torch.randn(10,3,400,400)
    vec = torch.randn(10,2)
    out = model(img.to(device),vec.to(device))
    t_nn = time.time() - t_nn
    print(t_nn)
    d = Data()

    _3d_distance= 55
    if method == 'cv':
        ###### 3D data ######
        print('creating cars')
        raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck, raw_bike, raw_person = create_all_raw_participants()
        print('creating buildings')
        raw_3d_buildings, raw_buildings = create_all_raw_buildings()
        print('starting simulation')
    route_file = 'a9-thi-route.rou.xml'
    all_vehicles = get_all_vehicles(os.path.join('../sumo_sim', route_file))
    print('starting simulation')
    test_episodes = [503]
    test_dict = dict()
    for i in test_episodes:
        print('____________________')
        print(i)
        config_file = 'config_a9-thi.sumocfg'
        sumo_cmd = configure_sumo(config_file, True, 3600)
        ego = all_vehicles[i]
        traci.start(sumo_cmd)
        step = 0
        showed_up = False
        while step < 1000:
            traci.simulationStep()
            if ego in traci.vehicle.getIDList():
                traci.gui.track(ego)
                traci.gui.setZoom('View #0', 750)
                showed_up = True
                current_dict = f'{ego}_{int(traci.simulation.getTime())}'
                test_dict[current_dict] = dict()
                if method == "cv":
                    t_3d = time.time()
                    points = create_3d(ego, raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck,
                              raw_bike, raw_person, raw_3d_buildings, raw_buildings,
                              _3d_distance, save=False)
                    t_3d = time.time() - t_3d
                    test_dict[current_dict]['duration_3d'] = t_3d
                    total_points = 0
                    for obj in points:
                        total_points += points[obj].shape[0]
                    test_dict[current_dict]['num_points'] = total_points
                    test_dict[current_dict]['num_objects'] = len(points)
                    t_detection = time.time()
                    all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians = detect()
                    t_detection = time.time() - t_detection
                    test_dict[current_dict]['duration_detection'] = t_detection
                    test_dict[current_dict]['num_vehicles'] = len(all_vehicles)
                    test_dict[current_dict]['num_cyclists'] = len(all_cyclists)
                    test_dict[current_dict]['num_pedestrians'] = len(all_pedestrians)
                    test_dict[current_dict]['num_detected_vehicles'] = len(detected_vehicles)
                    test_dict[current_dict]['num_detected_cyclists'] = len(detected_cyclists)
                    test_dict[current_dict]['num_detected_pedestrians'] = len(detected_pedestrians)
                    test_dict[current_dict]['detected_vehicles'] = detected_vehicles
                    test_dict[current_dict]['detected_cyclists'] = detected_cyclists
                    test_dict[current_dict]['detected_pedestrians'] = detected_pedestrians
                if method == "nn":
                    t_data = time.time()
                    dataset = d.get_data(ego)
                    t_data = time.time() - t_data
                    print(f't_data: {t_data}')
                    test_dict[current_dict]['duration_data'] = t_data
                    test_dict[current_dict]['num_objects'] = len(dataset)
                    print(f'dataset: {len(dataset)}')
                    #device = torch.device('cpu')  # Set the device to CPU
                    t_nn = time.time()
                    # Initialize lists for images and vectors
                    image_list = []
                    vector_list = []

                    for key in dataset:
                        image = dataset[key]['image'].unsqueeze(dim=0)
                        vector = dataset[key]['vector'].unsqueeze(dim=0)
                        # Convert the tensors to the same data type
                        image = image.type(torch.float32)
                        vector = vector.type(torch.float32)
                        # Append the images and vectors to their respective lists
                        image_list.append(image)
                        vector_list.append(vector)

                    # Concatenate the lists into tensors
                    image_batch = torch.cat(image_list, dim=0).to(device)
                    print(image_batch.shape)
                    vector_batch = torch.cat(vector_list, dim=0).to(device)

                    # Pass the batch through the model
                    output = model(image_batch, vector_batch)
                    t_nn = time.time() - t_nn
                    print(f't_nn: {t_nn}')
                    test_dict[current_dict]['duration_nn'] = t_nn
                    d.data = {}


            elif showed_up == True:
                traci.close()
                break

        try:
            traci.close()
        except:
            pass
    test_df = pd.DataFrame.from_dict(test_dict).T
    if method == "cv":
        print(f'duration_3d_mean: {test_df.duration_3d.mean()} duration_detection_mean: {test_df.duration_detection.mean()}')
        test_df.to_csv(f'test_dict_{method}_{cameras}_{traffic}.csv')
    if method == "nn":
        print(f'duration_data_mean: {test_df.duration_data.mean()} duration_nn_mean: {test_df.duration_nn.mean()}')
        test_df.to_csv(f'test_dict_{method}_{model}_{traffic}.csv')