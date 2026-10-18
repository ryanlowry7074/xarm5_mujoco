import mujoco
import mujoco.viewer
import time
import numpy as np

# Define Joints
base_pan = 0
shoulder_lift = 1
elbow_pitch = 2
writst_pitch = 3
wrist_roll = 4

# Load the XML model and create object
model = mujoco.MjModel.from_xml_path('xarm5.xml')
data = mujoco.MjData(model)

#Launch Passive Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation running. Close the viewr window to exit.")

    # Sequence of XYZ targets inside your enclosure (in meters)
    waypoints = [
        np.array([0.1487, 0.1142, 0.0591]),  # Point 1
        np.array([0.5805, 0.1142, 0.0591]),  # Point 2
        np.array([0.5805, -0.1144, 0.0591]), # Point 3
        np.array([0.1487, -0.1144, 0.0591]),  # Point 4
        np.array([0.1487, 0.1142, 0.0591])   # Back to Point 1
    ]
    
    current_target_idx = 0
    target_pos = waypoints[current_target_idx]
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')

    # Pre-allocate Jacobians and define the vertical downward vector
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    target_z = np.array([0.0, 0.0, -1.0]) # Points straight down

    while viewer.is_running():
        # 1. Find the current XYZ position of the end effector
        current_pos = data.site_xpos[tcp_id]
        current_mat = data.site_xmat[tcp_id].reshape(3, 3)
        current_z = current_mat[:, 2]  # The direction the tool is currently pointin

        # 2. Calculate position and rotational errors
        error_pos = target_pos - current_pos
        error_rot = np.cross(current_z, target_z) # Drives the wrist to stay vertical
        
        # Combine them into a single 6D error array
        error_full = np.concatenate((error_pos, error_rot))

        # Check distance for waypoints (using only the position error)
        if np.linalg.norm(error_pos) < 0.015:
            current_target_idx = (current_target_idx + 1) % len(waypoints)
            target_pos = waypoints[current_target_idx]
            print(f"Reached point. Moving to target {current_target_idx + 1}")
            
        # 3. Get both kinematic Jacobians and stack them vertically
        mujoco.mj_jacSite(model, data, jac_p, jac_r, tcp_id)
        jac_full = np.vstack((jac_p, jac_r))
        
        # 4. Calculate required joint movements using the 6D Jacobian
        dq = np.linalg.pinv(jac_full) @ error_full
        
        # Clamp the requested joint velocity to prevent violent snapping
        max_speed = 0.2
        dq = np.clip(dq, -max_speed, max_speed)
        
        # 5. Apply the joint adjustments with a much smaller step size
        step_size = 0.05
        data.ctrl[:] += step_size * dq
        
        # Step physics to match the 60Hz visual frame rate
        for _ in range(8):
            mujoco.mj_step(model, data)
            
        viewer.sync()
        time.sleep(1/60)