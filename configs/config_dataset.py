"""
Specify parameters for the general dataset generation
"""
DATASET_GENERAL = dict(
    DATASET_FILENAME='test',  # dataset filename
    TOP_VIEW_IMAGETYPE='box',  # top view image type (box [recommended] or screenshot)
    BEV_IMAGE_SIZE = 400, # size of the BEV image
    GUI = False, # show GUI
    SAVE_IMAGES=False,  # save images for test case
    PEDESTRIAN_ROUTE_FILE=None,  # pedestrian route file
    BIKE_ROUTE_FILE=None,  # bike route file
    VEHICLE_ROUTE_FILE='motorized_routes_2020-09-16_24h_dist.rou.xml',  # vehicle route file
    NET_FILE='ingolstadt_24h.net.xml.gz',  # net file
    CONFIG_FILE='24h_sim.sumocfg',  # config file
    LABEL_TYPE='3d',  # label type (closest [only closest traffic participant gets detected] or 3d [for realistic detection - recommended])
    _3D_DISTANCE=55,  # 3D distance where objects are created
    TRAFFIC_PARTICIPANT_DENSITY=100,  # traffic participant density of the point cloud
    CHUNK_SIZE=1000, # number of individual data entries for ech chunk
    NUM_CHUNKS=12, # number of chunks --> Total size of the recorded dataset = CHUNK_SIZE * NUM_CHUNKS
    START_TIME=8 * 3600, # start time of the simulation
    END_TIME=24 * 3600, # end time of the simulation
)

"""
Specify parameters for the network wide dataset
"""
DATASET_NETWORK = dict(
    VEHICLE_OFFSET=500,  # offset for the vehicle IDs to allow a warmup phase
    MAX_STEPS=3600,  # maximum number of steps for the simulation for each vehicle
)
"""
Specify parameters for the intersection specific dataset
"""
DATASET_INTERSECTION = dict(
    INTERSECTION='cluster_25579770_2633530003_2633530004_2633530005', # the intersection to be used for the dataset generation
    INTERSECTION_RADIUS=250, # radius of the intersection where dataset entries will be saved
)
