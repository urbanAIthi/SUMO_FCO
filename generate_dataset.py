import time

from utils.detect_cv import *
from configs.config_simulation import TRAFFIC
from configs.config_dataset import DATASET_GENERAL, DATASET_NETWORK
from configs.config_cv import DEFAULT as DEFAULT_CV
from utils.sumo_utils import update_modal_split, update_sumocfg, get_all_vehicles, configure_sumo, get_index_after_start
from detector import DatasetGenerator
from utils.create_3d import create_3d, create_all_raw_participants, create_all_raw_buildings
import traci
import shutil
import sys
import logging
import subprocess

"""
This scripts generates a dataset for training of the neural network approach. It uses the CV appraoch to detect the
objects that can be detected by the ego vehicle together with the corresponding BEV. This file uses the follwing configs config files:
    - config_simulation.py
    - config_dataset.py 
    - config_cv.py
"""


if __name__ == '__main__':
    # import the necessary configs
    filename = DATASET_GENERAL['DATASET_FILENAME']
    net_file = DATASET_GENERAL['NET_FILE']
    config_file = DATASET_GENERAL['CONFIG_FILE']
    _3d_distance = DATASET_GENERAL['_3D_DISTANCE']
    image_type = DATASET_GENERAL['TOP_VIEW_IMAGETYPE']
    image_size = DATASET_GENERAL['BEV_IMAGE_SIZE']
    gui = DATASET_GENERAL['GUI']
    detection_radius = DEFAULT_CV['detection_range']
    if image_type == "screenshot" and gui == False:
        raise ValueError('To take screenshots, set gui needs to be running')
    # check if the filename already exists as a folder under the data directory
    if os.path.isdir(os.path.join('data', filename)):
        raise ValueError('The filename already exists')
    # create the folder with the filename
    os.mkdir(os.path.join('data', filename))
    # copy the config folder to the filename folder
    shutil.copytree('configs', os.path.join('data', filename, 'configs'))
    logging.basicConfig(filename=os.path.join('data', filename, 'log.log'), level=logging.INFO)
    logging.info('starting dataset generation at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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
    all_vehicles = get_all_vehicles(os.path.join('sumo_sim', DATASET_GENERAL['VEHICLE_ROUTE_FILE']))
    first_index = get_index_after_start(DATASET_GENERAL['START_TIME'], os.path.join('sumo_sim', DATASET_GENERAL['VEHICLE_ROUTE_FILE']))
    logging.info(f'first index with start time {DATASET_GENERAL["START_TIME"]}: {first_index}')
    logging.info('creating participants')
    raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck, raw_bike, raw_person = \
        create_all_raw_participants()
    logging.info('creating buildings')
    raw_3d_buildings, raw_buildings = create_all_raw_buildings()
    logging.info('starting simulation')
    chunk_number = 0
    ego_index = 0
    # run the simulation till the numb
    d = DatasetGenerator(filename, raw_buildings, image_size, image_type, detection_radius, gui)
    while True:
        logging.info(f'running simulation with vehicle {ego_index} as ego')
        shutil.rmtree('tmp') if os.path.exists('tmp') else None  # remove the tmp folder
        os.makedirs('tmp', exist_ok=True)
        sumo_cmd = configure_sumo(config_file, gui, 3600)
        ego = all_vehicles[ego_index + DATASET_NETWORK['VEHICLE_OFFSET'] + first_index]  # add offset to allow for warmup
        traci.start(sumo_cmd)
        step = 0
        showed_up = False
        d.v_appended = list()
        while step < DATASET_NETWORK['MAX_STEPS']:  # avoid that vehicle is stuck
            traci.simulation.step()
            for person_id in traci.person.getIDList():
                waiting_time = traci.person.getWaitingTime(person_id)
                if waiting_time > 60:  # avoid that person is stuck
                    traci.person.remove(person_id)
            if ego in traci.vehicle.getIDList():
                # only generate data if the ego vehicle is a passenger vehicle
                if traci.vehicle.getVehicleClass(ego) != 'passenger':
                    break
                showed_up = True
                # focus on the ego if gui is running
                if gui:
                    traci.gui.trackVehicle('View #0', ego)
                    traci.gui.setZoom('View #0', 250)
                create_3d(ego, raw_small_car, raw_large_car, raw_delivery_car, raw_bus, raw_truck,
                          raw_bike, raw_person, raw_3d_buildings, raw_buildings,
                          _3d_distance, save=False)
                d.get_data(ego)  # generate a dataset entry
                logging.info(f'the dataset currently has {len(d.data)} elements')
                if len(d.data) > DATASET_GENERAL['CHUNK_SIZE']:
                    d.store_data(chunk_number)
                    traci.close()
                    chunk_number += 1
                    logging.info(f'finished recording {chunk_number}')
                    if chunk_number == DATASET_GENERAL['NUM_CHUNKS']:
                        sys.exit()
                    else:
                        d = DatasetGenerator(filename, raw_buildings)
                        break

            elif showed_up == True:
                traci.close()

                break
        ego_index += 1
        try:
            traci.close()
        except:
            pass

