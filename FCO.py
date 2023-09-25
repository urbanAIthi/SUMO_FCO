import cProfile
import pickle
import sys
import os
import shutil

import traci

from utils.sumo_utils import configure_sumo
from detector import Detector
from utils.visualize import visualize_fco


if __name__ == "__main__":
    fco_id = 'pv_12_7892_1' # define the id of the vehicle that should be the fco
    sumocfg_file = '24h_sim.sumocfg' # define the path to the sumocfg file
    detector = Detector(mode='cv') # initialize the detector mode (cv or nn) if nn is used also define the path to the model
    show_gui = False # show the sumo gui
    sumo_max_steps = 3600 # define the maximum number of steps for the sumo simulation
    save_path = 'visualization' # define the path to save the visualization

    # configure sumo
    sumo_cmd = configure_sumo(sumocfg_file, show_gui, sumo_max_steps)
    traci.start(sumo_cmd)

    # delete the save_path if it exists
    if os.path.exists('save_path'):
        shutil.rmtree('save_path')
    while True:
        traci.simulationStep()
        if fco_id in traci.vehicle.getIDList():
            all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians = detector.detect(fco_id)
            print(f'detected vehicles: {detected_vehicles}, detected_cyclists: {detected_cyclists}, detected_pedestrians: {detected_pedestrians}')
            visualize_fco(all_vehicles, all_cyclists, all_pedestrians,
                          detected_vehicles, detected_cyclists,
                          detected_pedestrians, [fco_id],
                          [traci.vehicle.getPosition(fco_id)[0], traci.vehicle.getPosition(fco_id)[1], 0],
                          show=True, save=False, save_path='save_path')
        if traci.simulation.getMinExpectedNumber() <= 0:
            sys.exit("Simulation ended")
