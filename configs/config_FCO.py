import numpy as np


MULTI_FCO_INTERSECTION = {
    'FILENAME': 'i3040_newvector',
    'OBSERVED_INTERSECTIONS': ['cluster_25579770_2633530003_2633530004_2633530005'],
    'FCO_PENETRATION_RATE': 0.2,
    'SUMO_FILE' : '24h_sim.sumocfg',
    'MODAL_SPLIT': dict(small_car=1, large_car=0, delivery=0, bus=0, truck=0),
    'ROU_FILE' : "motorized_routes_2020-09-16_24h_dist.rou.xml",
    'PEDESTRIANS_PATH' : None,
    'BIKES_PATH' : None,
    'SHOW_GUI' : False,
    'PLOT_DATA' : False,
    'START_TIME': 3600 * 12, 
    'END_TIME': 3600 * 16, 
    'NUM_CHUNKS': 12, 
    'CHUNK_SIZE': 1000,
    'RADIUS': 75, # radius aroud the intersection center in which the vehicles are considered
    'IMAGE_SIZE': 400, # size of the image in pixels for the fco dataset
    'DETECTION_METHOD': 'nn', # nn or cv
    'MODEL_PATH': 'ViT_ing_plot_vehicles_box_i3040_120000', # path to trained nn
    'CAMERA_SETTING': None # None for default setting
}