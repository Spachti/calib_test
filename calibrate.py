# calibrate.py (Final Version)

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import cv2


def compose_transform(R_ab, t_ab, R_bc, t_bc):
    """
    Compose two transforms: frame_a -> frame_b -> frame_c to get frame_a -> frame_c.
    """
    R_ac = R_bc @ R_ab
    t_ac = (R_bc @ t_ab.reshape(3, 1)).flatten() + t_bc.flatten()
    return R_ac, t_ac


def project_to_image(points_cam, K, dist_coeffs):
    """
    Project 3D points in camera frame onto the image plane.
    """
    proj_points, _ = cv2.projectPoints(
        points_cam.reshape(-1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, dist_coeffs
    )
    return proj_points.reshape(-1, 2)


def find_correspondences(data_dir):
    """
    Finds corresponding corner reflector points in radar and lidar data.
    """
    # Separate radar and lidar files based on naming convention
    radar_files, lidar_files = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if "_rad" in f:
            radar_files.append(f)
        else:
            lidar_files.append(f)

    radar_correspondences = []
    lidar_correspondences = []

    print(f"Found {len(radar_files)} radar/lidar measurement file pairs.")

    for r_file, l_file in zip(radar_files, lidar_files):
        """
        Radar Processing
        """
        df_radar = pd.read_csv(r_file)

        # Coordinates are on even rows (0, 2, 4, ...)
        coords_radar = df_radar.iloc[::2][["x", "y", "z"]].to_numpy()
        # Intensity is the "y" value on odd rows (1, 3, 5, ...)
        intensities_radar = df_radar.iloc[1::2]["y"].to_numpy()

        # Clean up any NaN values
        valid_indices_r = ~np.isnan(coords_radar).any(axis=1) & ~np.isnan(
            intensities_radar
        )
        coords_radar = coords_radar[valid_indices_r]
        intensities_radar = intensities_radar[valid_indices_r]

        # Filter within a 1m to 6m radius to reduce noise
        sq_distances_2d = np.sum(coords_radar**2, axis=1)
        lower_bound_mask = 1**3 <= sq_distances_2d
        upper_bound_mask = sq_distances_2d <= 6**3
        distance_mask = lower_bound_mask & upper_bound_mask
        coords_radar = coords_radar[distance_mask]
        intensities_radar = intensities_radar[distance_mask]

        if len(intensities_radar) == 0:
            print(
                f"Warning: No valid radar detections in {os.path.basename(r_file)}. Skipping."
            )
            continue

        # Find the index of the detection with the lowest intensity
        radar_reflector_point = coords_radar[np.argmin(intensities_radar)]

        """
        Lidar Processing
        """
        df_lidar = pd.read_csv(l_file)

        # Lidar coordinates are on the even rows (0, 2, 4, ...)
        lidar_points = df_lidar.iloc[::2][["x", "y", "z"]].to_numpy()
        # Intensity is the "y" value on odd rows (1, 3, 5, ...)
        lidar_intensities = df_lidar.iloc[1::2]["y"].to_numpy()

        # Clean up any potential NaN values from lidar points
        valid_indices_l = ~np.isnan(lidar_points).any(axis=1)
        lidar_points = lidar_points[valid_indices_l]
        lidar_intensities = lidar_intensities[valid_indices_l]

        # Filter within a 4m radius to reduce noise
        distance_mask = np.sum(lidar_points**2, axis=1) <= 4**3
        lidar_points = lidar_points[distance_mask]
        lidar_intensities = lidar_intensities[distance_mask]

        if lidar_points.shape[0] < 5:  # Not enough points to cluster
            print("Warning: Not enough lidar points in ROI. Skipping.")
            continue

        # Filter by intensity to reduce noise
        intensity_threshold = np.percentile(lidar_intensities, 50)
        high_intensity_mask = lidar_intensities > intensity_threshold

        if np.sum(high_intensity_mask) < 8:  # Check if enough points remain
            print("Warning: Not enough high-intensity points. Skipping.")
            continue

        lidar_points = lidar_points[high_intensity_mask]
        lidar_intensities = lidar_intensities[high_intensity_mask]

        # Cluster to find the reflector
        clustering = DBSCAN(eps=0.3, min_samples=8).fit(lidar_points)
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            continue
        # Reflector is the smallest cluster
        smallest_cluster_label = unique_labels[np.argmin(counts)]

        # Calculate reflector centroid for Lidar
        lidar_reflector_point = np.mean(
            lidar_points[labels == smallest_cluster_label], axis=0
        )

        # Add corresponding points
        radar_correspondences.append(radar_reflector_point)
        lidar_correspondences.append(lidar_reflector_point)

    radar_correspondences = np.array(radar_correspondences)
    lidar_correspondences = np.array(lidar_correspondences)

    return {
        "2d_correspondences": (
            radar_correspondences[:, :2],
            lidar_correspondences[:, :2],
        ),
        "3d_correspondences": (radar_correspondences, lidar_correspondences),
    }


def cost_function(params, radar_points, lidar_points):
    """
    Calculates the sum of squared errors between transformed radar points and lidar points.
    """
    theta, tx, ty = params

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([tx, ty])

    transformed_radar_points = (R @ radar_points.T).T + t
    error = np.sum((transformed_radar_points - lidar_points) ** 2)
    return error


def visualize_result(radar_pts, lidar_pts, optimal_params, title):
    """
    Visualizes the alignment of radar and lidar points.
    """
    theta, tx, ty = optimal_params
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([tx, ty])
    transformed_radar_pts = (R @ radar_pts.T).T + t

    plt.figure(figsize=(8, 8))
    plt.scatter(
        lidar_pts[:, 0],
        lidar_pts[:, 1],
        c="blue",
        label="Lidar Correspondences",
        s=100,
        alpha=0.7,
    )
    plt.scatter(
        transformed_radar_pts[:, 0],
        transformed_radar_pts[:, 1],
        c="red",
        marker="x",
        label="Calibrated Radar Points",
        s=100,
    )
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def verify_projection(
    R_radar_lidar, t_radar_lidar, correspondences_3d, data_dir, frame_index=0
):
    """
    Projects radar and lidar points onto the camera image for a given frame.
    """

    # Identify necessary files
    base_name = os.path.basename(
        sorted(glob.glob(os.path.join(data_dir, "*_rad.csv")))[frame_index]
    ).split("_")[0]
    img_file = os.path.join(data_dir, f"{base_name}.jpg")
    calib_cam_file = "cfl_calibration.npz"
    calib_lidar2cam_file = "lidar2cfl_new_all.npz"

    if not all(
        os.path.exists(f)
        for f in [
            img_file,
            calib_cam_file,
            calib_lidar2cam_file,
        ]
    ):
        print("Verification files not found. Skipping projection.")
        return

    # Load sensor data
    radar_points, lidar_points = correspondences_3d
    radar_reflector, lidar_reflector = (
        radar_points[frame_index],
        lidar_points[frame_index],
    )

    # Load calibration and image data
    calib_cam = np.load(calib_cam_file)
    K = calib_cam["camera_matrix"]
    dist_coeffs = calib_cam["dist_coeffs"]
    calib_lidar2cam = np.load(calib_lidar2cam_file)
    t_lidar_cam = calib_lidar2cam["t"]
    R_lidar_cam = calib_lidar2cam["R"]

    img = cv2.imread(img_file)
    h, w, _ = img.shape

    # Compose transformations for radar --> camera
    R_radar_cam, t_radar_cam = compose_transform(
        R_radar_lidar, t_radar_lidar, R_lidar_cam, t_lidar_cam
    )

    # Transform points to camera
    radar_points_cam = (R_radar_cam @ radar_reflector.T).T + t_radar_cam
    lidar_points_cam = (R_lidar_cam @ lidar_reflector.T).T + t_lidar_cam

    # Project to image
    if lidar_points_cam.shape[0] > 0:
        proj_points_lidar = project_to_image(lidar_points_cam, K, dist_coeffs)
    else:
        proj_points_lidar = np.array([])
    if radar_points_cam.shape[0] > 0:
        proj_points_radar = project_to_image(radar_points_cam, K, dist_coeffs)
    else:
        proj_points_radar = np.array([])

    # Remove projected points that are outside the image boundaries
    if proj_points_lidar.shape[0] > 0:
        valid_lidar_idx = (
            (proj_points_lidar[:, 0] >= 0)
            & (proj_points_lidar[:, 0] < w)
            & (proj_points_lidar[:, 1] >= 0)
            & (proj_points_lidar[:, 1] < h)
        )
        proj_points_lidar = proj_points_lidar[valid_lidar_idx]

    if proj_points_radar.shape[0] > 0:
        valid_radar_idx = (
            (proj_points_radar[:, 0] >= 0)
            & (proj_points_radar[:, 0] < w)
            & (proj_points_radar[:, 1] >= 0)
            & (proj_points_radar[:, 1] < h)
        )
        proj_points_radar = proj_points_radar[valid_radar_idx]

    # Draw Lidar points (blue dot)
    for p in proj_points_lidar:
        cv2.circle(
            img, tuple(p.astype(int)), radius=15, color=(255, 0, 0), thickness=-1
        )
    # Draw Radar reflector point (red circle)
    for p in proj_points_radar:
        cv2.circle(img, tuple(p.astype(int)), radius=15, color=(0, 0, 255), thickness=3)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.title(f"Corner reflector in frame {base_name}: Lidar (blue dot) and Radar (red circle) projections")
    plt.axis("off")
    os.makedirs("projections", exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join("projections", f"frame_{base_name}_reflector_sensor_to_camera.png")
    )


def main():
    """
    Main function to run the calibration process.
    """
    DATA_DIR = "test_data"

    print("Step 1: Finding correspondences...")
    correspondences = find_correspondences(DATA_DIR)
    radar_pts_2d, lidar_pts_2d = correspondences["2d_correspondences"]
    for r, l in zip(radar_pts_2d, lidar_pts_2d):
        print(f"Radar: {r}, Lidar: {l}")
    if len(radar_pts_2d) < 3:
        print("Error: Not enough correspondences found to perform optimization.")
        return
    print(f"Found {len(radar_pts_2d)} corresponding points.")

    initial_t = np.array([2.856, 0.635])
    initial_theta_rad = np.deg2rad(50)
    initial_guess_params = [initial_theta_rad, initial_t[0], initial_t[1]]

    print("\nStep 2: Running optimization...")
    result = minimize(
        cost_function,
        initial_guess_params,
        args=(radar_pts_2d, lidar_pts_2d),
        method="Nelder-Mead",
    )

    if not result.success:
        print("Optimization failed!")
        return

    print("Optimization successful!")
    theta_opt_rad, tx_opt, ty_opt = result.x
    theta_opt_deg = np.rad2deg(theta_opt_rad)

    print("\nCalibration results:")
    print(f"Optimized angle (theta): {theta_opt_deg:.4f} degrees")
    print(f"Optimized translation (tx, ty): [{tx_opt:.4f}, {ty_opt:.4f}]")

    tz_initial = -1.524
    t_radar_lidar = np.array([tx_opt, ty_opt, tz_initial])
    R_radar_lidar = np.array(
        [
            [np.cos(theta_opt_rad), -np.sin(theta_opt_rad), 0.0],
            [np.sin(theta_opt_rad), np.cos(theta_opt_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    print("\nFinal 3D Radar-to-Lidar transformation:")
    print("\nRotation matrix (R_radar_lidar):")
    print(np.round(R_radar_lidar, 4))
    print("\nTranslation vector (t_radar_lidar):")
    print(np.round(t_radar_lidar, 4))

    print("\nVisualizing final alignment after calibration...")
    visualize_result(
        radar_pts_2d, lidar_pts_2d, result.x, "Alignment after calibration"
    )
    for i in range(len(radar_pts_2d)):
        verify_projection(
            R_radar_lidar,
            t_radar_lidar,
            correspondences["3d_correspondences"],
            DATA_DIR,
            frame_index=i,
        )
    print("Done.")


if __name__ == "__main__":
    main()
