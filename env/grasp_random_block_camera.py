import numpy as np
import genesis as gs
import torch
from .util import euler_to_quaternion
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat, quat_to_T
import cv2
from scipy.optimize import linear_sum_assignment

class GraspRandomBlockCamEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 3  
        self.state_dim = 6  

        np.set_printoptions(threshold=np.inf)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=240,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            vis_options=gs.options.VisOptions(segmentation_level='entity'),
            show_viewer=vis,
        
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.URDF(
            file = './wam_description/urdf/wam_finger.urdf',
            fixed = True,
            )
        )

        
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.06, 0.06, 0.06), # block
                pos=(0.65, 0.0, 0.02),
                collision=True,
            )
        )
        self.cube.set_friction(3.0)

        self.ik_cache = {}  # Shared IK cache for all robots
        self.num_envs = num_envs
        # self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))
        
        
        #fixed transformation
        self.cam_0_transform = trans_quat_to_T(np.array([0, -0.08, 0.095]), xyz_to_quat(np.array([175, 0, 0])))


        # self.envs_idx = np.arange(self.num_envs)

        self.cameras = []            # <-- Initialize here
        self.cam_transforms = []     # <-- And here

        self.setup_cameras()         # <-- Move this BEFORE build
        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))  # <-- Then build the scene
        self.envs_idx = np.arange(self.num_envs)
        self.build_env()
    
    def setup_cameras(self):
        for _ in range(self.num_envs):
            cam = self.scene.add_camera(
                res=(1280, 720),
                fov=87,
                GUI=False,
            )
            self.cameras.append(cam)
            self.cam_transforms.append(self.cam_0_transform.copy())
            

    def update_camera_poses(self):
        for i in range(self.num_envs):
            end_effector = self.franka.get_link("wam_link_7")
            transform_matrix = trans_quat_to_T(
                end_effector.get_pos(envs_idx=i),
                end_effector.get_quat(envs_idx=i)
            ).cpu().numpy()[0]
            self.cameras[i].set_pose(transform=transform_matrix @ self.cam_transforms[i])

    def render_camera_views(self):
        rendered = []
        for cam in self.cameras:
            _, _, segmentation_mask, _ = cam.render(segmentation=True)
            rendered.append(segmentation_mask)
        return rendered

    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        # 13 DOF after fixing 2 bhand spread joints
        franka_pos = torch.tensor([-2.0374e-02,  1.3912e+00, -4.5302e-01,  1.0389e+00, -2.5591e+00,
        -9.1209e-01, -5.2248e-01,  2.3408e-02,  2.3388e-02,  2.3324e-02,
         8.8804e-06,  8.8778e-06,  1.7770e-05]).to(self.device)
        #franka_pos = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.04, 0.04]).to(self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("wam_link_7")

        self.arm_jnt_names = [
        "wam_joint_1",
        "wam_joint_2",
        "wam_joint_3",
        "wam_joint_4",
        "wam_joint_5",
        "wam_joint_6",
        ]

        self.arms_dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.arm_jnt_names]

        self.franka.set_dofs_kp(
            kp = np.array([300 for _ in range(6)]),
            dofs_idx_local = self.arms_dofs_idx,
        )
        self.franka.set_dofs_kv(
            kv = np.array([400 for _ in range(6)]),
            dofs_idx_local = self.arms_dofs_idx,
        )
        self.franka.set_dofs_force_range(
            lower = np.array([-30 for _ in range(6)]),
            upper = np.array([ 30 for _ in range(6)]),
            dofs_idx_local = self.arms_dofs_idx,
        )   

        self.hand_jnt_names = [
            "bhand_finger1",
            "bhand_finger2",
            "bhand_finger3"
        ]

        self.hand_dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.hand_jnt_names]
        self.finger_pos = torch.full((self.num_envs, 3), 0, dtype=torch.float32, device=self.device)

        self.franka.set_dofs_kp(
            kp=np.array([300 for _ in self.hand_jnt_names]),
            dofs_idx_local=self.hand_dofs_idx,
        )
        self.franka.set_dofs_kv(
            kv=np.array([450 for _ in self.hand_jnt_names]),
            dofs_idx_local=self.hand_dofs_idx,
        )
        self.franka.set_dofs_force_range(
            lower=np.array([-5 for _ in self.hand_jnt_names]),
            upper=np.array([15 for _ in self.hand_jnt_names]),
            dofs_idx_local=self.hand_dofs_idx,
        )

        # Set friction on finger links
        for link in self.franka.links:
            if "bhand_finger" in link.name:
                link.set_friction(3.0)
                
        self.tcp_offset = np.array([0.0, 0.0, 0.06]) # Offset from link_7 to TCP

        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([1.0, 0.0, 0.4], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        tcp_target_pos = self.pos - torch.tensor(self.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = tcp_target_pos,
            quat = self.quat,
        )

        self.franka.control_dofs_position(self.qpos[:, :7], self.motors_dof, self.envs_idx)


    def reset(self):
        self.build_env()
        ## random cube position
        cube_pos = np.array([1.0, 0.0, 0.05])
        x_min, x_max = 0.95, 0.95  
        y_min, y_max = -0.0, 0.0
        random_x = np.random.uniform(x_min, x_max, size=self.num_envs)
        random_y = np.random.uniform(y_min, y_max, size=self.num_envs)
        cube_pos = np.column_stack((random_x, random_y, np.full(self.num_envs, cube_pos[2])))
        ## random cube orientation
        fixed_roll = 0
        fixed_pitch = 0
        random_yaws = np.random.uniform(0, 0 * np.pi, size=self.num_envs) 
        quaternions = np.array([euler_to_quaternion(fixed_roll, fixed_pitch, yaw) for yaw in random_yaws])
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)   
        self.cube.set_quat(quaternions, envs_idx=self.envs_idx) 

        obs1 = self.cube.get_pos()

        obs2 = (self.franka.get_link("bhand_finger1_link_2").get_pos() + self.franka.get_link("bhand_finger2_link_2").get_pos() + self.franka.get_link("bhand_finger3_link_2").get_pos()) / 3
        state = torch.concat([obs1, obs2], dim=1)
        



        return state

    def step(self, actions):
        #action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 0 # Close gripper
        action_mask_2 = actions == 1 # Lift gripper
        action_mask_3 = actions == 2 # Lower gripper
        #action_mask_4 = actions == 4 # Do Nothing
        # action_mask_4 = actions == 4 # Move left
        # action_mask_5 = actions == 5 # Move right
        # action_mask_6 = actions == 6 # Move forward
        # action_mask_7 = actions == 7 # Move backward


        #idxs_0 = torch.nonzero(action_mask_0).squeeze(-1)
        idxs_1 = torch.nonzero(action_mask_1).squeeze(-1)

        #self.finger_pos[idxs_0, :] = 0
        self.finger_pos[idxs_1, :] = 1.62


        # self.finger_pos[action_mask_0, :] = 0
        # self.finger_pos[action_mask_1, :] = 1.62
        #self.finger_pos[action_mask_2] = 1.7

        
        pos = self.pos.clone()
        idxs_2 = torch.nonzero(action_mask_2).squeeze(-1)
        pos[idxs_2, 2] += 0.01
        idxs_3 = torch.nonzero(action_mask_3).squeeze(-1)
        pos[idxs_3, 2] -= 0.01

        # pos[action_mask_2, 2] += 0.01
        # pos[action_mask_3, 2] -= 0.01

        # if pos[:, 2] > 0.75:
        #     pos[:, 2] = 0.75

        # if pos[:, 2] < 0:
        #     pos[:, 2] = 0
        pos[:, 2] = torch.clamp(pos[:, 2], min=0.2, max=0.75)

        
        # pos[action_mask_4, 0] -= 0.05
        # pos[action_mask_5, 0] += 0.05
        # pos[action_mask_6, 1] -= 0.05
        # pos[action_mask_7, 1] += 0.05

        self.pos = pos


        
        tcp_target_pos = self.pos - torch.tensor(self.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        pos_key = tuple(tcp_target_pos.cpu().numpy().flatten())  # Convert tensor to a hashable key

        
        if pos_key in self.ik_cache:
            self.qpos = self.ik_cache[pos_key]  # Use cached solution
            #print("Using Cache")
        else:
            self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=tcp_target_pos,
            quat=self.quat,
            respect_joint_limit = True
            )
            self.ik_cache[pos_key] = self.qpos  # Store in cache
            #print("Calculate and Store")

        
        self.franka.control_dofs_position(self.qpos[:, :7], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.finger_pos, self.hand_dofs_idx, self.envs_idx)
        self.scene.step()

        end_effector = self.franka.get_link("wam_link_7")

        # segmented_cube_mask = None  # Define it outside the loop


        cube_props_list = []
        cube_area_list = []
        cube_x_list = []
        cube_y_list = []
        corners_list = []
        corner_reward_list = []
        # Define ideal corner coordinates
        IDEAL_CORNERS = torch.tensor([[ 357.0, 588.0], [363.0, 719.0], [ 652.0, 719.0], [653.0, 484.0], [489.0, 481.0]], device=self.device)

        env_spacing = np.array([2.0, 2.0])  # match your scene.build spacing

        for i in range(self.num_envs):
            # Compute environment offset based on grid layout
            row = i // int(np.sqrt(self.num_envs))
            col = i % int(np.sqrt(self.num_envs))
            env_offset = np.array([col * env_spacing[0] - (env_spacing[0] * (np.sqrt(self.num_envs)-1) / 2),
                                row * env_spacing[1] - (env_spacing[1] * (np.sqrt(self.num_envs)-1) / 2),
                                0.0])
            T_env = translation_matrix(env_offset)

            # Pose of end effector in *local* env
            pos_i = end_effector.get_pos(envs_idx=[i])
            quat_i = end_effector.get_quat(envs_idx=[i])
            # Pose of end-effector in its own env frame
            T_ee = trans_quat_to_T(
                end_effector.get_pos(envs_idx=[i]),
                end_effector.get_quat(envs_idx=[i])
            ).cpu().numpy().squeeze()

            T_cam_local = self.cam_transforms[i]  # relative offset

            # Camera pose = T_env * T_ee * T_cam_offset
            cam_T = T_env @ T_ee @ self.cam_transforms[i]
            self.cameras[i].set_pose(transform=cam_T)

            rgb, _, seg, _ = self.cameras[i].render(segmentation=True)
            seg_display = (seg == 2).astype(np.uint8) * 255

            cube_props, cube_area, cube_x, cube_y, corners = calculate_cube_properties(seg == 2)
            cube_props_list.append(cube_props)
            cube_area_list.append(cube_area)
            cube_x_list.append(cube_x)
            cube_y_list.append(cube_y)
            corners_list.append(corners)

            corner_reward = compute_corner_reward(corners, IDEAL_CORNERS, device=self.device)
            corner_reward *= 2

            corner_reward_list.append(corner_reward)

        corner_reward_tensor = torch.tensor(corner_reward_list, device=self.device).view(-1)



        cube_props = torch.cat(cube_props_list, dim=0)  # [num_envs, 3]




        # transform_matrices = trans_quat_to_T(end_effector.get_pos(), end_effector.get_quat()).cpu().numpy()

        
        # self.cam_0.set_pose(transform=transform_matrices[0] @ self.cam_0_transform)

        # _, depth_image, segmentation_mask, _ = self.cam_0.render(segmentation=True)
        # segmented_cube_mask = (segmentation_mask == 2)
        
        # cube_props, cube_area, cube_x, cube_y, corners = calculate_cube_properties(segmented_cube_mask)
        cube_percent = torch.tensor((cube_area/1228800)*100, device=self.device)
        cube_x = torch.tensor(cube_x, device=self.device)
        cube_y = torch.tensor(cube_y, device=self.device)

    
        # Define target values for valid grasping position
        TARGET_CUBE_PERCENT = torch.tensor(1.0, device=self.device)
        #TARGET_CUBE_X = torch.tensor(952, device=self.device)
        #TARGET_CUBE_Y = torch.tensor(948, device=self.device)
        
        
        # Calculate how close we are to the target values
        cube_percent_diff = torch.abs(cube_percent - TARGET_CUBE_PERCENT)
        #cube_x_diff = torch.abs(cube_x - TARGET_CUBE_X)
        #cube_y_diff = torch.abs(cube_y - TARGET_CUBE_Y)
               
        # Define thresholds for what we consider "close enough"
        PERCENT_THRESHOLD = torch.tensor(0.5, device=self.device)  # Within 0.5% of target
        COORD_THRESHOLD = torch.tensor(50, device=self.device)    # Within 50 pixels of target
        
        # Check if we're in a valid grasping position based on visual properties
        #valid_visual_grasp = (
         #   (cube_percent_diff < PERCENT_THRESHOLD) & 
         #   (cube_x_diff < COORD_THRESHOLD) & 
        #    (cube_y_diff < COORD_THRESHOLD)
        #)



        block_position = self.cube.get_pos(envs_idx=self.envs_idx)
        gripper_position = (self.franka.get_link("bhand_finger1_link_2").get_pos(envs_idx=self.envs_idx) + 
                          self.franka.get_link("bhand_finger2_link_2").get_pos(envs_idx=self.envs_idx) + 
                          self.franka.get_link("bhand_finger3_link_2").get_pos(envs_idx=self.envs_idx)) / 3
        states = torch.concat([cube_props, gripper_position], dim=1)

        # print("envs_idx", self.envs_idx)
        # print("block_position", block_position.shape)
        # print("block_position[:, 2]", block_position[:, 2].shape)

        # is_grasping = (torch.norm(gripper_position - block_position, dim=1) < 0.05)
        # finger_closed = (self.finger_pos[:, 0] > 1)

        # contacts = self.franka.get_contacts(self.plane)
        # valid_mask = contacts['valid_mask'][0]
        # collision_penalty = 0
        # for i, is_valid in enumerate(valid_mask):
        #     if is_valid:
        #         collision_penalty = -3

        contacts = self.franka.get_contacts(self.cube)
        valid_mask_np = contacts['valid_mask']  # shape: (num_envs, num_contacts)
        valid_mask = torch.tensor(valid_mask_np, dtype=torch.bool, device=self.device)
        grasp_reward = torch.where(valid_mask.any(dim=1), torch.tensor(10.0, device=self.device), torch.tensor(0.0, device=self.device))

        # height_penalty = torch.zeros(self.num_envs, device=self.device)
        # mask = gripper_position[:, 2] > 0.3
        # height_penalty[mask] = gripper_position[mask, 2] * (-30) + -5.0
        
        min_lift_height = 0.02  # 2 cm
        lift_reward = torch.maximum(torch.full_like(block_position[:, 2], min_lift_height), block_position[:, 2]) * 1000 - 27.79


        # print("lift_reward:", block_position[:, 2].shape)
        # print("height_penalty:", height_penalty.shape)
        # print("grasp_reward:", grasp_reward.shape)
        # print("corner_reward_tensor:", corner_reward_tensor.shape)


        


        #rewards = -torch.norm(block_position - gripper_position, dim=1) + torch.maximum(torch.tensor(0.02), block_position[:, 2]) * 10

        # Calculate rewards
        rewards = (
            + lift_reward
            # + (is_grasping & finger_closed) * 5.0  # Grasp stability
            # + collision_penalty  # Penalty for touching the ground
            # + height_penalty
            + grasp_reward
            #+ valid_visual_grasp * 10.0  # Large reward for achieving correct visual properties
            #- (cube_x_diff/1000 + cube_y_diff/1000)  # Small continuous reward for getting closer to target values
            + corner_reward  # Reward for matching ideal corner positions
        )
        # print(f"lift reward: {lift_reward}")
        # # print(f"height penalty: {height_penalty}")
        # print(f"grasp reward: {grasp_reward}")
        # print(f"corner reward: {corner_reward_tensor}")
        # print(f"total reward: {rewards}")
        # print(f"pos: {pos}")
        # print(f"action: {actions}")
        
        dones = block_position[:, 2] > 0.35
        return states, rewards, dones, grasp_reward
    
def translation_matrix(offset):
    T = np.eye(4)
    T[:3, 3] = offset
    return T

def calculate_cube_properties(segmented_cube_mask, device="cuda"):
    # Convert boolean mask to uint8 for OpenCV
    mask_uint8 = segmented_cube_mask.astype(np.uint8) * 255
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no cube is detected, return zero tensors
        return torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device), 0, 0, 0, None
    
    # Get the largest contour (should be the cube)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to get corners
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Get the corners
    corners = approx.reshape(-1, 2)  # Shape: (N, 2) where N is number of corners
    
    # Compute the area as the sum of True values
    cube_area = np.sum(segmented_cube_mask)

    # Get the coordinates of all True values (cube pixels)
    cube_pixels = np.argwhere(segmented_cube_mask)


    # Compute the centroid (mean of x and y coordinates)
    center_y, center_x = cube_pixels.mean(axis=0)

    # Create a single tensor of shape (1,3)
    cube_properties_tensor = torch.tensor([[center_x, center_y, cube_area]], dtype=torch.float32, device=device)

    return cube_properties_tensor, cube_area, center_x, center_y, corners

def compute_corner_reward(detected_corners, ideal_corners, scale=1000.0, device="cuda"):
    if detected_corners is None or len(detected_corners) == 0:
        return torch.tensor(-0.2, device=device)  # small penalty
 
    detected = torch.tensor(detected_corners, dtype=torch.float32, device=device)
    num = min(len(detected), len(ideal_corners))
    detected = detected[:num]
    ideal = ideal_corners[:num]
 
    # Compute pairwise distances
    cost_matrix = torch.cdist(detected, ideal).cpu().numpy()  # Shape [N, N]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Hungarian match
 
    total_cost = cost_matrix[row_ind, col_ind].sum()
    reward = -total_cost / scale
 
    return torch.tensor(reward, device=device)

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = GraspRandomBlockCamEnv(vis=True)