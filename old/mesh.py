#####################
# Tests with meshed #
#####################

# pt_cloud = o3d.geometry.PointCloud()
# pt_cloud.points = o3d.utility.Vector3dVector(obj_pt_cloud)
# pt_cloud.estimate_normals()

# # create mesh
# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pt_cloud)

# # refine mesh
# # mesh = mesh.filter_smooth_taubin(number_of_iterations=10, lambda_param=0.5, mu_param=0.5)

# # o3d.visualization.draw_geometries([mesh])
# # create depth map
# depth_map = np.zeros((480, 640), dtype=np.uint8)
# # project mesh into depth map
# pts = np.asarray(mesh.vertices)[mesh.triangles]
# pts = pts.reshape(-1, 3)
# mask, points = is_point_in_image(pts, K, P, image_size=(640, 480), max_distance=radius)
# points = points[mask]
# points[:, 2] /= points[:, 3]
# points = points[:, :3]
# points = points.reshape(-1, 3, 3)
# proj_pts = np.round(points).astype(np.int32)
# proj_pts[:, :, 0] = np.clip(proj_pts[:, :, 0], 0, 639)  # clip x-coordinates
# proj_pts[:, :, 1] = np.clip(proj_pts[:, :, 1], 0, 479)  # clip y-coordinates
# mask = (proj_pts[:, :, 2] >= 0) & (proj_pts[:, :, 2] < radius)  # mask of valid points
# depth_map.fill(0)  # clear depth map
# depth_map[proj_pts[mask, 1], proj_pts[mask, 0]] = 255

# test = Image.fromarray(depth_map)
# test.save('a.png')