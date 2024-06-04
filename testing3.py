import open3d as o3d
import numpy as np
import pandas as pd
import cv2
from PIL import Image

def look_at_matrix(camera_position, target_position, up_vector=np.array([0, 0, 1])):
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.vstack([right, -up, forward])

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -camera_position

    view_matrix = rotation_matrix @ translation_matrix
    return view_matrix

def calculate_focal_length(camera_position, look_at_position, base_focal_length=2000, base_distance=20):
    distance = np.linalg.norm(camera_position - look_at_position)
    focal_length = base_focal_length * (distance / base_distance)
    return focal_length

def add_text_to_image(image, text, position=(10, 30), font_scale=1, color=(0, 0, 0), thickness=2):
    # converts image to opencv format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    # adds text
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    # converts to PIL image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return Image.fromarray(image)

def validate_tracker_id(data, tracker_id):
    return tracker_id in data['tracker_id'].unique()

csv_path = 'C:/Users/karli/PycharmProjects/testing123/positions_export_9_20231122_051940.csv'
data = pd.read_csv(csv_path)

pcd = o3d.io.read_point_cloud('C:/Users/karli/PycharmProjects/testing123/Velodrome_SIS_corrected_downsampled.ply')
pcd = pcd.voxel_down_sample(voxel_size=0.05)

camera_origin = np.array([0, 0, 5])
sphere_radius = 0.2

while True:
    selected_tracker_id_1 = int(input("Enter the first tracker_id of the cyclist to display (Red): "))
    if validate_tracker_id(data, selected_tracker_id_1):
        break
    else:
        print(f"Tracker ID {selected_tracker_id_1} does not exist. Please try again.")

while True:
    selected_tracker_id_2 = int(input("Enter the second tracker_id of the cyclist to display (Blue): "))
    if validate_tracker_id(data, selected_tracker_id_2):
        break
    else:
        print(f"Tracker ID {selected_tracker_id_2} does not exist. Please try again.")

filtered_data_1 = data[data.tracker_id == selected_tracker_id_1]
filtered_data_2 = data[data.tracker_id == selected_tracker_id_2]

merged_data = pd.merge(filtered_data_1, filtered_data_2, on='time', suffixes=('_1', '_2'))

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

frame_count = 0

# video parameters
video_path = "C:/Users/karli/PycharmProjects/testing123/images/video.mkv"
frame_width = 1920
frame_height = 1061
fps = 50

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'h264'), fps, (frame_width, frame_height))

for index, row in merged_data.iterrows():
    look_at_1 = np.array([row['x_1'], row['y_1'], row['z_1']])
    look_at_2 = np.array([row['x_2'], row['y_2'], row['z_2']])
    real_speed_1 = row['real_speed_1']
    real_speed_2 = row['real_speed_2']

    view_matrix = look_at_matrix(camera_origin, (look_at_1 + look_at_2) / 2)

    # dynamic focal length
    focal_length = calculate_focal_length(camera_origin, (look_at_1 + look_at_2) / 2)

    pinhole_params = o3d.camera.PinholeCameraParameters()
    pinhole_params.extrinsic = view_matrix
    pinhole_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(500, 500, focal_length, focal_length, 250, 250)

    sphere_1 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    elevated_look_at_1 = look_at_1 + np.array([0, 0, sphere_radius])  # Move the sphere up by its radius
    sphere_1.translate(elevated_look_at_1)
    sphere_1.paint_uniform_color([1.0, 0.0, 0.0])  # sarkans

    sphere_2 = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    elevated_look_at_2 = look_at_2 + np.array([0, 0, sphere_radius])
    sphere_2.translate(elevated_look_at_2)
    sphere_2.paint_uniform_color([0.0, 0.0, 1.0])  # zils

    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.add_geometry(sphere_1)
    vis.add_geometry(sphere_2)

    vc = vis.get_view_control()
    vc.convert_from_pinhole_camera_parameters(pinhole_params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    # captures image
    image_path = f"C:/Users/karli/PycharmProjects/testing123/images/frame_{frame_count}.png"
    vis.capture_screen_image(image_path)

    # loads image and adds text
    image = Image.open(image_path)
    image = add_text_to_image(image, f'Atrums 1 (Sarkans): {real_speed_1:.2f} m/s, Atrums 2 (Zils): {real_speed_2:.2f} m/s', color=(0, 0, 0))

    # saves image with text
    image.save(f"C:/Users/karli/PycharmProjects/testing123/images/frame_with_text_{frame_count}.png")

    # Write frame to video
    frame = cv2.imread(f"C:/Users/karli/PycharmProjects/testing123/images/frame_with_text_{frame_count}.png")
    out.write(frame)

    frame_count += 1

vis.destroy_window()
out.release()

print(f"Video saved to {video_path}")
