import traci
import time
import sys
from utils.detect_cv import *
from configs.config_simulation import TRAFFIC
from configs.config_dataset import DATASET_GENERAL, DATASET_INTERSECTION
from configs.config_cv import DEFAULT as DEFAULT_CV
from utils.sumo_utils import configure_sumo, update_modal_split, update_sumocfg
from utils.create_3d import create_3d, create_all_raw_participants, create_all_raw_buildings
from detector import DatasetGenerator
import logging
import shutil
import subprocess

"""
This script has the goal to create a dataset for a specific intersection only in order
in order to investigate if the nn approach achieves higher accuracies if the variability in the 
dataset is lower. This script uses the follwing config files: 
    - config_simulation.py
    - config_dataset.py 
    - config_cv.py
"""

if __name__ == '__main__':
    filename = DATASET_GENERAL['DATASET_FILENAME']
    net_file = DATASET_GENERAL['NET_FILE']
    config_file = DATASET_GENERAL['CONFIG_FILE']
    _3d_distance = DATASET_GENERAL['_3D_DISTANCE']
    image_type = DATASET_GENERAL['TOP_VIEW_IMAGETYPE']
    image_size = DATASET_GENERAL['BEV_IMAGE_SIZE']
    gui = DATASET_GENERAL['GUI']
    intersection = DATASET_INTERSECTION['INTERSECTION']
    intersection_radius = DATASET_INTERSECTION['INTERSECTION_RADIUS']
    detection_radius = DEFAULT_CV['detection_range']
    if image_type == "screenshot" and gui == False:
        raise ValueError('To take screenshots, set gui needs to be running')
    if os.path.isdir(os.path.join('data', filename)):
        raise ValueError('The filename already exists')
    
    # create the folder with the filename
    print(filename)
    path = os.path.join('data', filename)
    print(path)
    os.makedirs(path)
    logger_filename = os.path.join(path, 'log_dataset.log')
    logging.basicConfig(filename=logger_filename, level=logging.INFO)
    logging.info('starting simulation at ' + time.strftime("%H:%M:%S", time.localtime()))
    # copy the config folder to the filename folder
    shutil.copytree('configs', os.path.join('data', filename, 'configs'))
    #delete the config_FCO file in the configs folder
    os.remove(os.path.join('data', filename, 'configs', 'config_FCO.py'))
    # delete the config_networks file in the configs folder
    os.remove(os.path.join('data', filename, 'configs', 'config_networks.py'))

    # get the current git hash
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    logging.info('dataset generated with git hash ' + git_hash)
    start_time = time.time()
    update_sumocfg(os.path.join('sumo_sim', DATASET_GENERAL['CONFIG_FILE']),
                   DATASET_GENERAL['NET_FILE'],
                   [DATASET_GENERAL['VEHICLE_ROUTE_FILE'],
                    DATASET_GENERAL['BIKE_ROUTE_FILE'], DATASET_GENERAL['PEDESTRIAN_ROUTE_FILE']],
                   DATASET_GENERAL['START_TIME'], DATASET_GENERAL['END_TIME'])
    update_modal_split(os.path.join('sumo_sim', DATASET_GENERAL['VEHICLE_ROUTE_FILE']), TRAFFIC['MODAL_SPLIT'])
    logging.info('creating participants')
    raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck, raw_bike, raw_person = create_all_raw_participants()
    logging.info('creating buildings')
    raw_3d_buildings, raw_buildings = create_all_raw_buildings()
    logging.info('starting simulation')
    sumo_cmd = configure_sumo(config_file, gui, 3600)
    traci.start(sumo_cmd)
    intersection_center = traci.junction.getPosition(intersection)
    for chunk in range(DATASET_GENERAL['NUM_CHUNKS']):
        d = DatasetGenerator(filename, raw_buildings, image_size, image_type, detection_radius, gui)
        while len(d.data) < DATASET_GENERAL['CHUNK_SIZE']:
            logging.info(f'chunk {chunk} currently has {len(d.data)} samples')
            traci.simulation.step()
            vehicles = traci.vehicle.getIDList()
            intersection_vehicles = list()
            for v in vehicles:
                v_pos = traci.vehicle.getPosition(v)
                if np.linalg.norm(np.array(v_pos) - np.array(intersection_center)) < intersection_radius:
                    intersection_vehicles.append(v)
            for v in intersection_vehicles:
                ego = v
                if traci.vehicle.getVehicleClass(ego) == 'passenger':
                    create_3d(ego, raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck,
                              raw_bike, raw_person, raw_3d_buildings, raw_buildings,
                              _3d_distance, save=False)
                    d.get_data(ego)

        d.store_data(chunk)
    traci.close()
    logging.info('simulation finished at ' + time.strftime("%H:%M:%S", time.localtime()))
    logging.info('simulation took ' + str((time.time() - start_time)/3600) + ' hours')
    sys.exit()
