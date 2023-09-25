import matplotlib.pyplot as plt
from utils.create_box import create_box

polygons = create_box(5,10,5,5,90,[5,5,90])
# plot the polygon
fig, ax = plt.subplots()
ax.add_patch(polygons)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
ax.set_aspect('equal')
plt.savefig('box.png')