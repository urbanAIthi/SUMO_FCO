def create_points(info: list, b=2, l=5.0, h=1.5, density=100):
    x, y, angle = info
    theta = np.radians(angle)
    # Define the half-length and half-height of the rectangle
    half_l = l / 2
    half_b = b / 2

    # Calculate the vertex points
    vertices = np.array([[-half_b, half_l], [half_b, half_l], [half_b, -half_l], [-half_b, -half_l]])
    # Translate the vertices to the center point
    vertices += [x, y]
    org_vertices = vertices
    # Rotate the vertices by the angle theta around the center point
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    vertices = np.dot(vertices, rotation_matrix)

    # Find the minimum and maximum x and y coordinates
    x_min, y_min = np.min(org_vertices, axis=0)
    x_max, y_max = np.max(org_vertices, axis=0)
    x_distance = x_max - x_min
    y_distance = y_max - y_min

    # Create arrays of x and y coordinates using linspace
    x = np.linspace(x_min, x_max, int(x_distance * density))
    y = np.linspace(y_min, y_max, int(y_distance * density))
    z = np.linspace(h, h, 1)

    # Create a grid of x and y coordinates using meshgrid
    X, Y, Z = np.meshgrid(x, y, z)

    # Stack X and Y horizontally to create an array of points
    points_ = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
    points_ = np.dot(points_, rotation_matrix)

    points = list()
    for i in range(4):
        start = vertices[i]
        end = vertices[(i + 1) % 4]
        distance = int(np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2))
        num_points = distance * density
        x_vals = np.linspace(start[0], end[0], num_points)
        y_vals = np.linspace(start[1], end[1], num_points)
        z_vals = np.zeros(num_points)
        points.append(np.stack((x_vals, y_vals, z_vals), axis=-1))
    points = np.vstack(points)

    # Create additional points in the z direction
    z_vals = np.linspace(0, h, int(h * density))
    points = np.repeat(points, int(h * density), axis=0)
    z_vals = np.tile(z_vals, int(len(points) / len(z_vals)))
    points[:, 2] = z_vals
    points = np.vstack([points, points_])
    return points

def save_points(points, ego):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join('3d', f'{ego}_{traci.simulation.getTime()}.ply'), pcd)
    xyz = o3d.io.read_point_cloud(os.path.join('3d', f'{ego}_{traci.simulation.getTime()}.ply'))

def create_raw_car(type, b=2, l=5.0, h=1.5, density=100):
    if type == 'box':
        # Define the half-length and half-height of the rectangle
        half_l = l / 2
        half_b = b / 2

        # Calculate the vertex points
        vertices = np.array([[-half_b, half_l], [half_b, half_l], [half_b, -half_l], [-half_b, -half_l]])
        # Translate the vertices to the center point
        org_vertices = vertices

        # Find the minimum and maximum x and y coordinates
        x_min, y_min = np.min(org_vertices, axis=0)
        x_max, y_max = np.max(org_vertices, axis=0)
        x_distance = x_max - x_min
        y_distance = y_max - y_min

        # Create arrays of x and y coordinates using linspace
        x = np.linspace(x_min, x_max, int(x_distance * density))
        y = np.linspace(y_min, y_max, int(y_distance * density))
        z = np.linspace(h, h, 1)

        # Create a grid of x and y coordinates using meshgrid
        X, Y, Z = np.meshgrid(x, y, z)

        # Stack X and Y horizontally to create an array of points
        points_ = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        points = list()
        for i in range(4):
            start = vertices[i]
            end = vertices[(i + 1) % 4]
            distance = int(np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2))
            num_points = distance * density
            x_vals = np.linspace(start[0], end[0], num_points)
            y_vals = np.linspace(start[1], end[1], num_points)
            z_vals = np.zeros(num_points)
            points.append(np.stack((x_vals, y_vals, z_vals), axis=-1))
        points = np.vstack(points)

        # Create additional points in the z direction
        z_vals = np.linspace(0, h, int(h * density))
        points = np.repeat(points, int(h * density), axis=0)
        z_vals = np.tile(z_vals, int(len(points) / len(z_vals)))
        points[:, 2] = z_vals
        points = np.vstack([points, points_])
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(points)
        #o3d.visualization.draw_geometries([pcd])
    elif type =='car':
        pcd = o3d.io.read_point_cloud('Vehicle/Benz.ply')
        points = np.asarray(pcd.points)
        raise
    return points

def create_triangle(a=5, b=2, h=1.5, density=100):
    vertices = np.array([[0, 0], [-a, -b/2], [-a, b/2]])
    org_vertices = vertices

    # Find the minimum and maximum x and y coordinates
    x_min, y_min = np.min(org_vertices, axis=0)
    x_max, y_max = np.max(org_vertices, axis=0)
    x_distance = x_max - x_min
    y_distance = y_max - y_min

    # Create arrays of x and y coordinates using linspace
    x = np.linspace(x_min, x_max, int(x_distance * density))
    y = np.linspace(y_min, y_max, int(y_distance * density))

    # Create a grid of x and y coordinates using meshgrid
    X, Y = np.meshgrid(x, y)

    # Create a mask to filter out the points outside the triangle
    mask = (Y >= ((b * X) / a)) & (Y <= (-((b * X) / a) + b))
    X, Y = X[mask], Y[mask]

    # Stack X and Y horizontally to create an array of points
    points_ = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))

    # Create additional points in the z direction
    z_vals = np.linspace(0, h, int(h * density))
    points_ = np.repeat(points_, int(h * density), axis=0)
    z_vals = np.tile(z_vals, int(len(points_) / len(z_vals)))
    points_[:, 2] = z_vals

    points = points_
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame])
    return points


def create_positioned_car(raw_car, info):
    x, y, angle = info
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Perform rotation and translation in a single step
    points = np.empty_like(raw_car)
    points[:, 0] = raw_car[:, 0] * cos_theta - raw_car[:, 1] * sin_theta + x
    points[:, 1] = raw_car[:, 0] * sin_theta + raw_car[:, 1] * cos_theta + y
    points[:, 2] = raw_car[:, 2]

    return points

def create_positioned_building(raw_building, info, p_x, p_y):
    x, y, angle = info
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Translate the points so that (p_x, p_y) becomes the origin
    translated_points = raw_building - np.array([p_x, p_y, 0])

    # Perform rotation around the origin
    rotated_points = np.empty_like(translated_points)
    rotated_points[:, 0] = translated_points[:, 0] * cos_theta - translated_points[:, 1] * sin_theta
    rotated_points[:, 1] = translated_points[:, 0] * sin_theta + translated_points[:, 1] * cos_theta
    rotated_points[:, 2] = translated_points[:, 2]

    # Translate the points back to their original position and apply the (x, y) translation
    points = rotated_points + np.array([p_x - x, p_y - y, 0])

    return points