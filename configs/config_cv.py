import numpy as np


VEHICLE = dict(
    min_depth_image_occupancy_percentage=0.0009,     # number that represents the minimum required percentage of pixels occupied by a vehicle
)

PEDESTRIAN = dict(
    min_depth_image_occupancy_percentage=0.0009,     # number that represents the minimum required percentage of pixels occupied by a pedestrian
)

CYCLIST = dict(
    min_depth_image_occupancy_percentage=0.0009,     # number that represents the minimum required percentage of pixels occupied by a cyclist
)


DEFAULT = dict(
    detection_range=50,                             # meters
    objects_to_detect={'vehicle': VEHICLE,          # dict of objects to detect
                       'pedestrian': PEDESTRIAN,
                       'cyclist': CYCLIST},
    save_depth_image_dir=False#'./depth_images',          # save depth images
)

CAMERAS = dict(
    FRONT_CAMERA = dict(
        ACTIVE = 1,
        FOV=100,                                        # degree
        IMAGE_WIDTH=640,                                # pixels
        IMAGE_HEIGHT=480,                               # pixels
        ROTATION_ANGLE=np.array([0,0,0]),               # degrees (x, y, z) - camera coordinate system
        TRANSLATION_OFFSET=np.array([0,-1.5,0])         # meters (x, y, z) - camera coordinate system (translation offset)
    ),

    BACK_CAMERA = dict(
        ACTIVE = 1,
        FOV=100,                                        # degree
        IMAGE_WIDTH=640,                                # pixels
        IMAGE_HEIGHT=480,                               # pixels
        ROTATION_ANGLE=np.array([0,180,0]),             # degrees (x, y, z) - camera coordinate system
        TRANSLATION_OFFSET=np.array([0,-1.5,4.5]),      # meters (x, y, z) - camera coordinate system (translation offset)
    ),

    LEFT_CAMERA = dict(
        ACTIVE = 1,
        FOV=100,                                        # degree
        IMAGE_WIDTH=640,                                # pixels
        IMAGE_HEIGHT=480,                               # pixels
        ROTATION_ANGLE=np.array([0,90,0]),              # degrees (x, y, z) - camera coordinate system
        TRANSLATION_OFFSET=np.array([-2.25,-1.5,0.9]),  # meters (x, y, z) - camera coordinate system (translation offset)
    ),

    RIGHT_CAMERA = dict(
        ACTIVE = 1,
        FOV=100,                                        # degree
        IMAGE_WIDTH=640,                                # pixels
        IMAGE_HEIGHT=480,                               # pixels
        ROTATION_ANGLE=np.array([0,-90,0]),             # degrees (x, y, z) - camera coordinate system
        TRANSLATION_OFFSET=np.array([2.25,-1.5, 0.9]),  # meters (x, y, z) - camera coordinate system (translation offset)
    )
)