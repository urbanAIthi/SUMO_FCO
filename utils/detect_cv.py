import os
import numpy as np
from utils.utils_cv import filter_points_by_radius, \
    is_point_in_image, create_oracle_depth_image, \
    filter_by_oracle_depth, create_camera_matrices_from_config, \
    create_depth_image_from_array
from utils.file_utils import read_files
from utils.visualize import visualize_frame
from configs.config_cv import DEFAULT, CAMERAS
from typing import Dict


def split_detected_from_undetected_objects(
        image_plane_objects_detection_states: Dict[str, Dict[str, np.ndarray]],
        obj_pt_clouds: Dict[str, Dict[str, np.ndarray]],
        cameras):
    object_types = list(obj_pt_clouds.keys())
    detected_objects = {ot: [] for ot in object_types}
    other_objects = {ot: [] for ot in object_types}

    # Split detected from undetected objects
    for cam_pos in cameras:
        for obj_type in obj_pt_clouds:
            for fn in obj_pt_clouds[obj_type]:
                # check if the object is visible in the image plane, otherwise append it to "other objects"
                if cam_pos in image_plane_objects_detection_states and obj_type in image_plane_objects_detection_states[
                    cam_pos] \
                        and fn in image_plane_objects_detection_states[cam_pos][obj_type] \
                        and image_plane_objects_detection_states[cam_pos][obj_type][fn]:
                    detected_objects[obj_type].append(fn)
                else:
                    other_objects[obj_type].append(fn)

    # Remove duplicates
    detected_objects = {ot: list(set(detected_objects[ot])) for ot in detected_objects}
    other_objects = {ot: list(set(other_objects[ot])) for ot in other_objects}
    # only those that are in other objects but not in detected objects
    other_objects = {ot: [fn for fn in other_objects[ot] if fn not in detected_objects[ot]] for ot in other_objects}

    return detected_objects, other_objects


def check_objects_on_image_plane(camera_config: Dict, seq_pt_clouds: Dict) -> Dict:
    """Check if objects are on the image plane."""

    K = camera_config['K']
    P = camera_config['P']
    img_width = camera_config['image_width']
    img_height = camera_config['image_height']

    # Check if points are on image plane
    objects_pt_cloud_image_plane = {
        obj_type: {
            filename:
                is_point_in_image(
                    pt_cloud=seq_pt_clouds[obj_type][filename], K=K, P=P, image_size=(img_width, img_height)
                )
            for filename in seq_pt_clouds[obj_type]}
        for obj_type in seq_pt_clouds
    }

    # mask of visible pt clouds
    objects_pt_cloud_mask = {
        obj_type: {
            filename: objects_pt_cloud_image_plane[obj_type][filename][0] for filename in
            objects_pt_cloud_image_plane[obj_type]}
        for obj_type in objects_pt_cloud_image_plane
    }

    # disparity of visible pt clouds
    objects_pt_cloud_disparity = {
        obj_type: {
            filename: objects_pt_cloud_image_plane[obj_type][filename][1] for filename in
            objects_pt_cloud_image_plane[obj_type]}
        for obj_type in objects_pt_cloud_image_plane
    }

    return objects_pt_cloud_mask, objects_pt_cloud_disparity


def filter_objects_to_detect(seq_pt_clouds: Dict, objects_pt_cloud_mask: Dict) -> Dict:
    # Filter visible objects on the image plane
    objects_on_image_plane = {}

    for obj_type in seq_pt_clouds:
        for fn in seq_pt_clouds[obj_type]:
            if not np.any(objects_pt_cloud_mask[obj_type][fn]):
                continue
            file_obj_name = fn.split('_')[0]
            if file_obj_name not in objects_on_image_plane:
                objects_on_image_plane[file_obj_name] = []

            objects_on_image_plane[file_obj_name].append(fn)

    return objects_on_image_plane


def run_computer_vision_approach(ply_dir: str, cameras: Dict, scan_radius: int, objects_to_detect: Dict,
                                 save_depth_image_dir: str = None):
    """
    Run computer vision approach.
    :param ply_dir: str - path to the directory with point clouds
    :param cameras: Dict - camera config
    :param scan_radius: int - radius of the scan
    :param objects_to_detect: Dict - objects to detect
    :param save_depth_image_dir: str - path to the directory where to save depth images
    """

    # Iterate through the directory
    for par_folder, seq_pt_clouds in read_files(ply_dir):
        # Filter objects inside radius
        filtered_seq_pt_clouds = {}
        for obj_type in seq_pt_clouds:
            filtered_seq_pt_clouds[obj_type] = {}
            for fn in seq_pt_clouds[obj_type]:
                mask = filter_points_by_radius(
                    pt_cloud=seq_pt_clouds[obj_type][fn], center=np.array([0, 0, 0]), radius=scan_radius
                )
                if np.any(mask):
                    filtered_seq_pt_clouds[obj_type][fn] = seq_pt_clouds[obj_type][fn]

        camera_detections = {}
        camera_detections_occupancy = {}
        # Iterate cameras
        for camera_pos in cameras:
            # Get camera config
            camera_config = cameras[camera_pos]
            K = camera_config['K']
            P = camera_config['P']
            img_width = camera_config['image_width']
            img_height = camera_config['image_height']
            no_pixels = img_width * img_height

            # Check if objects are on image plane
            objects_pt_cloud_mask, objects_pt_cloud_disparity = check_objects_on_image_plane(
                camera_config=camera_config, seq_pt_clouds=filtered_seq_pt_clouds
            )

            # Filter all objects on image plane
            all_objects_on_image_plane = filter_objects_to_detect(filtered_seq_pt_clouds, objects_pt_cloud_mask)

            # Only those objects that we want to detect
            objects_to_detect_on_image_plane = {k: v for k, v in all_objects_on_image_plane.items() if
                                                k in objects_to_detect}

            # Create point cloud of all visible objects
            visible_obj_list = [filtered_seq_pt_clouds[obj_type][p] for obj_type in all_objects_on_image_plane for p in
                                all_objects_on_image_plane[obj_type]]
            visible_objects_pt_cloud = np.vstack(visible_obj_list) if visible_obj_list else np.empty((0, 3))

            # Create depth image
            oracle_depth = create_oracle_depth_image(
                K=K, P=P, pt_cloud=visible_objects_pt_cloud, image_size=(img_width, img_height), max_depth=scan_radius
            )

            # Save the oracle depth image
            if save_depth_image_dir:
                create_depth_image_from_array(
                    depth_array=oracle_depth, max_depth=scan_radius,
                    file_path=os.path.join(save_depth_image_dir, par_folder, camera_pos, 'oracle_depth.png')
                )

            # Objects that can be detected by the ego vehicles camera
            objects_on_image_detection_state = {obj: {} for obj in objects_to_detect}
            objects_on_image_detection_occupancy = {obj: {} for obj in objects_to_detect}

            # Iterate all objects to detect objects on image plane and check occlusion
            for obj_type in objects_to_detect_on_image_plane:
                # object specific loop
                for obj_filename in objects_to_detect_on_image_plane[obj_type]:
                    # object specific min depth image occupancy
                    min_depth_image_occupancy_percentage = objects_to_detect[obj_type][
                        'min_depth_image_occupancy_percentage']

                    obj_pt_cloud = filtered_seq_pt_clouds[obj_type][obj_filename]
                    image_plane_mask = objects_pt_cloud_mask[obj_type][obj_filename]
                    disparity_info = objects_pt_cloud_disparity[obj_type][obj_filename]

                    # CHECK mesh code -> mesh.py

                    # Depth only for the object
                    obj_depth = create_oracle_depth_image(
                        K, P, obj_pt_cloud, image_size=(img_width, img_height), max_depth=scan_radius,
                        image_plane_mask=image_plane_mask, disparity_info=disparity_info)
                    # Filter by overlapping depth measures
                    valid_depth_img_coords = filter_by_oracle_depth(oracle_depth_img=oracle_depth,
                                                                    ref_depth_img=obj_depth)

                    # save obj depth and valid depth img coords to file
                    if save_depth_image_dir:
                        create_depth_image_from_array(
                            depth_array=obj_depth, max_depth=scan_radius,
                            file_path=os.path.join(save_depth_image_dir, par_folder, camera_pos,
                                                   f'{obj_type}_{obj_filename}_obj_depth.png')
                        )

                    # Get the occupancy of the object in the image plane
                    valid_obj_pixel_occupancy = valid_depth_img_coords.shape[0]

                    # Min visibility calculation based on the obj depth and valid depth coordinates
                    if min_depth_image_occupancy_percentage <= valid_obj_pixel_occupancy / no_pixels:
                        # Set object visibility to True
                        objects_on_image_detection_state[obj_type][obj_filename] = True
                        objects_on_image_detection_occupancy[obj_type][obj_filename] = valid_obj_pixel_occupancy / no_pixels
                    else:
                        # Set object visibility to False
                        objects_on_image_detection_state[obj_type][obj_filename] = False
                        objects_on_image_detection_occupancy[obj_type][obj_filename] = valid_obj_pixel_occupancy / no_pixels

            camera_detections[camera_pos] = objects_on_image_detection_state
            camera_detections_occupancy[camera_pos] = objects_on_image_detection_occupancy


        yield par_folder, camera_detections, camera_detections_occupancy, seq_pt_clouds, objects_pt_cloud_mask, objects_pt_cloud_disparity

def get_object_max_occupancy(image_plane_objects_detection_occupancy: Dict[str, Dict[str, float]], obj_pt_clouds: Dict[str, Dict[str, np.ndarray]],
                              ego:str, simtime: int) -> Dict[str, float]:
    """Get the maximum occupancy of the image plane for each object type."""
    object_dict = {}
    for cam_pos in image_plane_objects_detection_occupancy:
        for obj_type in image_plane_objects_detection_occupancy[cam_pos]:
            for obj in image_plane_objects_detection_occupancy[cam_pos][obj_type]:
                if obj not in object_dict:
                    object_dict[obj] = image_plane_objects_detection_occupancy[cam_pos][obj_type][obj]
                else:
                    if image_plane_objects_detection_occupancy[cam_pos][obj_type][obj] > object_dict[obj]:
                        object_dict[obj] = image_plane_objects_detection_occupancy[cam_pos][obj_type][obj]
    # add occupancy of 0 to the objects that are not in the image plane
    for obj_type in obj_pt_clouds:
        if obj_type == 'building':
            continue
        else:
            for obj in obj_pt_clouds[obj_type]:
                if obj not in object_dict:
                    object_dict[obj] = 0.0
    # only keep everything before the . in the dict keys
    object_dict = {f'ego_{ego}_{key.split(".ply")[0]}_{simtime}': value for key, value in object_dict.items()}
    
    return object_dict

if __name__ == '__main__':
    ply_dir = '3d'

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
            image_plane_objects_detection_states=image_plane_objects_detection_states, obj_pt_clouds=obj_pt_clouds
        )

        # visualize_frame(detected_objects, other_objects, obj_pt_clouds, cameras, radius, objects_image_masks,
        #                 plot_image_plane_rays=True, plot_scan_radius=True)

