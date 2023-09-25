# SUMO_FCO
## Enhancing Realistic Floating Car Observers in Microscopic Traffic Simulation
##

This is the GitHub repository for our conference paper "Enhancing Realistic Floating Car Observers in Microscopic Traffic Simulation" presented at the ITSC 2023. Our research enables, to realisticly detect other traffic participants by ego vehicles that are equipped with up to four camera sensors. 

We offer two detection options. One is a direct computer vision approach, which generates depth images from generated 3D point clouds of road users. Traffic participants must have a certain proportion of the depth images in order to be detected. Furthermore, we enable a detection with neural networks, which emulates the results of the computer vision method. For this purpose, data sets are recorded using the computer vision method and neural networks are then trained. The trained networks can then be used to detect the traffic participants. This method is up to 18 times faster than the CV method and avoids that the microscopic simulation slows down during detection. 


## Usage:

### Computer Vision Method:
The computer vision method can be run directly without any training. The core component here is the `Detector` class within `detector.py`. An example of its usage can be found in `FCO.py`.

### Neural Network Method:
To emulate the computer vision method using neural networks, follow these three steps:

1. **Dataset Creation**:
   - Using the computer vision method, create a dataset. For this, we've provided two scripts:
      - `create_dataset.py`: This creates a dataset for the entire traffic network defined. In this setup, an FCO drives within the network and captures data. If it's no longer in the simulation, the simulation restarts with a new FCO.
      - `generate_dataset_intersection.py`: This script specifically generates a dataset for an intersection. It defines a radius around the intersection, and during each simulation step, FCO data from all vehicles within the radius is recorded.
   - Both scripts save the dataset to the `/data` directory, partitioned into multiple chunks. The relevant configurations for the dataset (`config_cv.py`, `config_dataset.py`, `config_simulation`) are also saved.

2. **Network Training**:
   - Execute the `train_detection.py` file to train the networks.
   - The trained networks and test results are stored in the `/models` directory. Furthermore, the relevant configurations (`config_networks.py`, as well as dataset configs `config_cv.py`, `config_dataset.py`, `config_simulation`) are saved.

3. **Deploying the Trained Network**:
   - After training, you can create an FCO using the trained networks. The primary component for this is the `Detector` class found in `detector.py`. You can refer to `FCO.py` for an example of its application.

## Installation:
```bash
git clone https://github.com/jegerner/SUMO_detector_plus.git
cd SUMO_detector_plus
pip install -r requirements.txt
```

## Citation:
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{gerner2023enhancing,
  title={Enhancing Realistic Floating Car Observers in Microscopic Traffic Simulation},
  author={Gerner, Jeremias and R{\"o}{\ss}le, Dominik and Cremers, Daniel and Bogenberger, Klaus and Sch{\"o}n, Torsten and Schmidtner, Stefanie},
  booktitle={26th IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year={2023},
  address={Bilbao, Bizkaia, Spain},
  month={24-28 September},
  publisher={IEEE}
}
```

This work uses the SUMO simulation network provided in: https://github.com/TUM-VT/sumo_ingolstadt.git <br>
The ViT implementation is based on the following repository: https://github.com/lucidrains/vit-pytorch.git

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.