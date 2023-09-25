import xml.etree.ElementTree as ET
from sumolib import checkBinary
import os
import traci
from typing import List


def get_all_vehicles(path_to_xml) -> list:
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    vehicle_ids = [vehicle.get('id') for vehicle in root.iter('vehicle')]
    return vehicle_ids

def get_index_after_start(start_time: float, path_to_xml: str) -> int:
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    vehicle_start_times = [float(vehicle.get('depart')) for vehicle in root.iter('vehicle')]
    for i, time in enumerate(vehicle_start_times):
        if time > start_time:
            return i
    raise ValueError('No vehicle found after start time')

def configure_sumo(config_file, gui, max_steps):
    """
    Configure various parameters of SUMO.
    """
    # Setting the cmd mode or the visual mode
    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    # Setting the cmd command to run sumo at simulation time
    model_path = os.path.join('sumo_sim', config_file)
    sumo_cmd = [
        sumo_binary, "-c", model_path, "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps), "--xml-validation", "never", "--start", "--quit-on-end"
    ]

    return sumo_cmd

def update_sumocfg(path_to_cfg: str, net_file: str, route_files: List[str], start_value: int, end_value: int):
    # Parse the XML file
    tree = ET.parse(path_to_cfg)
    root = tree.getroot()
    if net_file is not None:
        # Update net-file value
        root.find(".//net-file").set("value", net_file)

    # Update route-files value (excluding any None values)
    valid_route_files = [rf for rf in route_files if rf is not None]
    root.find(".//route-files").set("value", ",".join(valid_route_files))

    # Update begin and end values
    root.find(".//begin").set("value", str(start_value))
    root.find(".//end").set("value", str(end_value))

    # Write the changes back to the file
    tree.write(path_to_cfg)
    
def update_modal_split(path_to_xml: str, modal_split: dict):
    print(modal_split)
    # Load the XML file
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    # Find the vTypeDistribution element
    vTypeDistribution = root.find('vTypeDistribution')

    # Iterate over the vType elements
    for vType in vTypeDistribution.findall('vType'):
        vTypeId = vType.get('id')
        if vTypeId in modal_split:
            # Update the probability attribute
            vType.set('probability', str(modal_split[vTypeId]))

    # Save the modified XML file
    tree.write(path_to_xml)

def simulate():
    for _ in range(1):
        traci.simulationStep()


if __name__ == "__main__":
    #update_modal_split('../sumo_sim/multimodal.rou.xml', TRAFFIC['MODAL_SPLIT'])
    update_sumocfg_file('../sumo_sim/config_a9-thi.sumocfg', 'multimodal.rou.xml', 'cycle.rou.xml', None)