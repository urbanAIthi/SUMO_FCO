import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import gzip

def plot_network(net_file):
    # Open the .gz file and read it
    with gzip.open(net_file, 'rb') as f:
        file_content = f.read()

    # parse XML content
    root = ET.fromstring(file_content)

    # iterate through each edge in the network
    for edge in root.iter('edge'):
        if edge.get('function') != 'internal':
            x = []
            y = []

            # iterate through each lane in the edge
            for lane in edge.iter('lane'):
                shape = lane.get('shape').split()

                # split the shape into x, y coordinates and add them to the list
                for point in shape:
                    x_coord, y_coord = map(float, point.split(','))
                    x.append(x_coord)
                    y.append(y_coord)

            # plot the edge
            plt.plot(x, y, color='black', linewidth=0.5)


    plt.show()

# usage
plot_network('../sumo_sim/ingolstadt_24h.net.xml.gz')