TRAFFIC = dict(
    MODAL_SPLIT=dict(small_car=1, large_car=0, delivery=0, bus=0, truck=0),
    # modal split for the vehicles [small car, large car, delivery, bus, truck]
    VEHICLE_REPRESENTATION=dict(SMALL_CAR='box', LARGE_CAR='box', DELIVERY='box',
                                BUS='box', TRUCK='box', PERSON='box', BIKE='box'), # vehicle representation (box [recommended] or mesh [in developement])
    MESH_PATH = dict(SMALL_CAR='meshes/car.obj', LARGE_CAR='meshes/suv.obj', DELIVERY='meshes/van.obj',
                                BUS='meshes/bus.obj', TRUCK='meshes/truck.obj',
                     PERSON='meshes/pedestrian1.obj', BIKE='meshes/cyclist.stl'),
    SMALL_CAR=dict(
        LENGTH=4.5,  # length of the vehicle in meters
        WIDTH=1.8,  # width of the vehicle in meters
        HEIGHT=1.5),  # height of the vehicle in meters
    LARGE_CAR=dict(
        LENGTH=4.7,  # length of the vehicle in meters
        WIDTH=1.9,  # width of the vehicle in meters
        HEIGHT=1.8),  # height of the vehicle in meters
    DELIVERY=dict(
        LENGTH=6,  # length of the vehicle in meters
        WIDTH=2,  # width of the vehicle in meters
        HEIGHT=2.8),  # height of the vehicle in meters
    BUS=dict(
        LENGTH=12,  # length of the vehicle in meters
        WIDTH=2.5,  # width of the vehicle in meters
        HEIGHT=3.5),  # height of the vehicle in meters
    TRUCK=dict(
        LENGTH=12,  # length of the vehicle in meters
        WIDTH=2.5,  # width of the vehicle in meters
        HEIGHT=3.5),  # height of the vehicle in meters
    PERSON=dict(
        LENGTH=0.5,  # length of the pedestrian in meters
        WIDTH=0.5,  # width of the pedestrian in meters
        HEIGHT=1.8),  # height of the pedestrian in meters
    BIKE=dict(
        LENGTH=1.5,  # length of the bike in meters
        WIDTH=0.5,  # width of the bike in meters
        HEIGHT=1.5),  # height of the bike in meters
)

BUILDINGS = dict(
    BUILDING_REPRESENTATION='polygons',  # building representation (mesh [in development] or polygons [recommended])
    POLY_FILE='buildings_a9.poly.xml',  # poly file
    MESH_PATH='meshes/building.obj',  # path to the building mesh
    DENSITY = 10, # density of the buildings point cloud
    HEIGHT=3,  # height of the building in meters if polygons are used
)
