# import numpy as np
#
#
# VEHICLE = dict(
#     min_depth_image_occupancy_percentage=0.0009,     # number that represents the minimum required percentage of pixels occupied by a vehicle
# )
#
# PEDESTRIAN = dict(
#     min_depth_image_occupancy_percentage=0.0009,     # number that represents the minimum required percentage of pixels occupied by a pedestrian
# )
#
# CYCLIST = dict(
#     min_depth_image_occupancy_percentage=0.0009,     # number that represents the minimum required percentage of pixels occupied by a cyclist
# )
#
#
# DEFAULT = dict(
#     detection_range=50,                             # meters
#     objects_to_detect={'vehicle': VEHICLE,          # dict of objects to detect
#                        'pedestrian': PEDESTRIAN,
#                        'cyclist': CYCLIST},
#     save_depth_image_dir=False#'./depth_images',          # save depth images
# )
#
# CAMERAS = dict(
#     FRONT_CAMERA = dict(
#         ACTIVE = 1,
#         FOV=100,                                        # degree
#         IMAGE_WIDTH=640,                                # pixels
#         IMAGE_HEIGHT=480,                               # pixels
#         ROTATION_ANGLE=np.array([0,0,0]),               # degrees (x, y, z) - camera coordinate system
#         TRANSLATION_OFFSET=np.array([0,-1.5,0])         # meters (x, y, z) - camera coordinate system (translation offset)
#     ),
#
#     BACK_CAMERA = dict(
#         ACTIVE = 1,
#         FOV=100,                                        # degree
#         IMAGE_WIDTH=640,                                # pixels
#         IMAGE_HEIGHT=480,                               # pixels
#         ROTATION_ANGLE=np.array([0,180,0]),             # degrees (x, y, z) - camera coordinate system
#         TRANSLATION_OFFSET=np.array([0,-1.5,4.5]),      # meters (x, y, z) - camera coordinate system (translation offset)
#     ),
#
#     LEFT_CAMERA = dict(
#         ACTIVE = 1,
#         FOV=100,                                        # degree
#         IMAGE_WIDTH=640,                                # pixels
#         IMAGE_HEIGHT=480,                               # pixels
#         ROTATION_ANGLE=np.array([0,90,0]),              # degrees (x, y, z) - camera coordinate system
#         TRANSLATION_OFFSET=np.array([-2.25,-1.5,0.9]),  # meters (x, y, z) - camera coordinate system (translation offset)
#     ),
#
#     RIGHT_CAMERA = dict(
#         ACTIVE = 1,
#         FOV=100,                                        # degree
#         IMAGE_WIDTH=640,                                # pixels
#         IMAGE_HEIGHT=480,                               # pixels
#         ROTATION_ANGLE=np.array([0,-90,0]),             # degrees (x, y, z) - camera coordinate system
#         TRANSLATION_OFFSET=np.array([2.25,-1.5, 0.9]),  # meters (x, y, z) - camera coordinate system (translation offset)
#     )
# )
#
#
# DATASET_GENERATION = dict(
#     DELETE_EXISTING_DATASET=False,  # delete existing dataset
#     DATASET_FILENAME='test',  # dataset filename
#     TOP_VIEW_IMAGETYPE='box',  # top view image type (plot or screenshot)
#     GUI = True, # show GUI
#     SAVE_IMAGES=False,  # save images for test case
#     PEDESTRIAN_ROUTE_FILE='pedestrians.rou.xml',  # pedestrian route file
#     BIKE_ROUTE_FILE='bikes.rou.xml',  # bike route file
#     VEHICLE_ROUTE_FILE='multimodal.rou.xml',  # vehicle route file
#     NET_FILE='a9-thi.net.xml',  # net file
#     CONFIG_FILE='config_a9-thi.sumocfg',  # config file
#     LABEL_TYPE='3d',  # label type (closest or 3D)
#     _3D_DISTANCE=55,  # 3D distance where objects are created
#     TRAFFIC_PARTICIPANT_DENSITY=75,  # traffic participant density of the point cloud
#     VEHICLE_OFFSET=500,  # offset for the vehicle IDs
#     MAX_STEPS=1000,  # maximum number of steps for the simulation for each vehicle
#     CHUNK_SIZE=10000, # number of individual data entries for ech chunk
#     NUM_CHUNKS=12 # number of chunks --> Total size of the recorded dataset = CHUNK_SIZE * NUM_CHUNKS
# )
#
# TRAFFIC = dict(
#     MODAL_SPLIT=dict(small_car=0.6, large_car=0.2, delivery=0.1, bus=0.05, truck=0.05),
#     # modal split for the vehicles [small car, large car, delivery, bus, truck]
#     VEHICLE_REPRESENTATION=dict(SMALL_CAR='box', LARGE_CAR='mesh', DELIVERY='mesh',
#                                 BUS='mesh', TRUCK='mesh', PERSON='mesh', BIKE='mesh'),
#     MESH_PATH = dict(SMALL_CAR='meshes/car.obj', LARGE_CAR='meshes/suv.obj', DELIVERY='meshes/van.obj',
#                                 BUS='meshes/bus.obj', TRUCK='meshes/truck.obj',
#                      PERSON='meshes/pedestrian1.obj', BIKE='meshes/cyclist.stl'),
#     SMALL_CAR=dict(
#         LENGTH=4.5,  # length of the vehicle in meters
#         WIDTH=1.8,  # width of the vehicle in meters
#         HEIGHT=1.5),  # height of the vehicle in meters
#     LARGE_CAR=dict(
#         LENGTH=4.7,  # length of the vehicle in meters
#         WIDTH=1.9,  # width of the vehicle in meters
#         HEIGHT=1.8),  # height of the vehicle in meters
#     DELIVERY=dict(
#         LENGTH=6,  # length of the vehicle in meters
#         WIDTH=2,  # width of the vehicle in meters
#         HEIGHT=2.8),  # height of the vehicle in meters
#     BUS=dict(
#         LENGTH=12,  # length of the vehicle in meters
#         WIDTH=2.5,  # width of the vehicle in meters
#         HEIGHT=3.5),  # height of the vehicle in meters
#     TRUCK=dict(
#         LENGTH=12,  # length of the vehicle in meters
#         WIDTH=2.5,  # width of the vehicle in meters
#         HEIGHT=3.5),  # height of the vehicle in meters
#     PERSON=dict(
#         LENGTH=0.5,  # length of the pedestrian in meters
#         WIDTH=0.5,  # width of the pedestrian in meters
#         HEIGHT=1.8),  # height of the pedestrian in meters
#     BIKE=dict(
#         LENGTH=1.5,  # length of the bike in meters
#         WIDTH=0.5,  # width of the bike in meters
#         HEIGHT=1.5),  # height of the bike in meters
# )
#
# BUILDINGS = dict(
#     BUILDING_REPRESENTATION='polygons',  # building representation (mesh or polygons)
#     POLY_FILE='buildings_a9.poly.xml',  # poly file
#     MESH_PATH='meshes/building.obj',  # path to the building mesh
#     DENSITY = 10, # density of the buildings point cloud
#     HEIGHT=5,  # height of the building in meters if polygons are used
# )
#
# VIT_CONFIG = dict(
#     # Image size. If you have rectangular images, make sure your image size is the maximum of the width and height.
#     image_size=500,
#
#     # Number of patches. image_size must be divisible by patch_size.
#     # The number of patches is:  n = (image_size // patch_size) ** 2 and n must be greater than 16.
#     patch_size=50,
#
#     # Number of classes to classify.
#     num_classes=1,
#
#     # Last dimension of output tensor after linear transformation nn.Linear(..., dim).
#     dim=512,
#
#     # Number of Transformer blocks.
#     depth=6,
#
#     # Number of heads in Multi-head Attention layer.
#     heads=8,
#
#     # Dimension of the MLP (FeedForward) layer.
#     mlp_dim=2048,
#
#     # Number of image's channels.
#     channels=1,
#
#     # Dropout rate (float between [0, 1], default 0).
#     dropout=0,
#
#     # Embedding dropout rate (float between [0, 1], default 0).
#     emb_dropout=0,
#
#     # Pooling type: either 'cls_token' pooling or 'mean' pooling.
#     pool='cls',
# )
#
# FCO_config = dict(
#     method= 'CV', # CV of NN
#     trained_NN_model = 'xxx', # path to trained NN model
# )

