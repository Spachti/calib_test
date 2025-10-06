## Proposed solution
The "calibrate.py" script calibrates a radar and lidar by matching corresponding detections of an orange corner reflector. To this end, it first applies distance filters to each sensor's data to reduce noise. It then identifies the reflector's specific location using two different methods: 
- For Radar, it leverages a pre-assigned color tag (presumably obtained by sensor fusion with camera) corresponding to the object's color by selecting the point with the minimum green-channel value.
- For Lidar, it applies DBSCAN clustering to points that pass an intensity threshold. After clustering, it identifies the smallest valid cluster (assuming the reflector is the smallest distinct object in the ROI).
<!-- -->
Using these corresponding points, an optimization routine computes the rigid body transformation that best aligns the two sensors. Finally, the accuracy of this alignment is confirmed by projecting both sensors' data onto a camera image to verify the calibration.

## Task 
Make a tool to perform the extrinsic calibration of the radar and lidar ( rotation and translation matrix )

The radar has no elevation so z component is not reliable and should be ignored, that makes the problem simpler as we need just one angle in the rotation matrix.

In the test data you have the left front camera, lidar and front left radar (1jpeg, 2csv files). Data with same number is from almost same timestamp.  The camera is not needed for the calibration, it is just for reference reprojection. Basically , what you have are a set of measurements with a corner reflector and the lidar, you could match them and get the task done. We use the corner reflect as that reflection is strong and we ca see the point and also we see the corner reflector in the lidar.

screenshot.jpg shows a top view of the lidar and radar points projected on the ground ( blue lidar, red radar ) given some initial guess of transformation. You can use the quick_view.py script to take a look at the data ( use arrows to go through the files ).

The lidar and radar csv files can be read with : 

    df = pd.read_csv(path)
    points = df[['x', 'y', 'z']].to_numpy()

You can use as initial guess the translation : 

    t = array([ 2.856,  0.635, -1.524]) 

and a rotation matrix (if helpful):

        th = np.deg2rad(50)
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                        [np.sin(th),  np.cos(th), 0.0],
                        [0.0,         0.0,        1.0]])

What you need to find is basically the true **t** and the angle **th**.

You can reproject on camera and see if results are ok. To read the calibration files use :

    Camera intrinsics like in open cv (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) , camera matrix (K) and distortion coeffcients : 

    calib = np.load(calib_cam_npz_file_repo)
    K = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]

    and lidar to camera extrinsic calibration (translation t and rotation R):

    calib = np.load(calib_lidar2cam_file_npz_repo)
    t_lidar_cam = calib["t"]
    R_lidar_cam = calib["R"]

After you find the radar to lidar extrinsics you could find the radar to camera like this :
 
    def compose_transform(R_ab, t_ab, R_bc, t_bc):
        """
        Compose two transforms:
          frame_a -> frame_b -> frame_c
        to get frame_a -> frame_c
        """
        R_ac = R_bc @ R_ab
        t_ac = R_bc @ t_ab + t_bc
        return R_ac, t_ac

    R_radar_cam, t_radar_cam = compose_transform(R_radar_lidar, t_radar_lidar, R_lidar_cam, t_lidar_cam)
  
 Then project with code similar to this for instance :

     def project_to_image(points_cam, K, dist_coeffs):
        """Project 3D points in camera frame onto image plane."""
        proj_points, _ = cv2.projectPoints(
            points_cam, np.zeros((3, 1)), np.zeros((3, 1)), K, dist_coeffs
        )
        return proj_points.reshape(-1, 2)

     radar_points_cam = (R_radar_cam @ radar_points.T).T + t_radar_cam
     proj_points = project_to_image(radar_points_cam, K, dist_coeffs)
    

USE AI to not spend ages on it ! Use whatever you like to code and send us your github link. Write 3-4 lines to explain what you have done.
Hint : All you need to do is find corespondences (the corner reflector in various sensing modalities) and find the best fit transformation with your favorite solver. Each set of (radar,lidar) will give you one corespondence. Quick results are more fancy than elaborate GUIs. Things are not going to be perfect, recall radar has no elevation measurement.  

