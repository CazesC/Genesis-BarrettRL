import numpy as np
import genesis as gs
import torch
from .util import euler_to_quaternion
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat, quat_to_T

class GraspRandomBlockVelEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 8  
        self.state_dim = 6  

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=vis,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.URDF(
            file = './wam_description/urdf/new_wam.urdf',
            fixed = True,
            )
        )

        
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.08, 0.08, 0.08), # block
                pos=(0.65, 0.0, 0.02),
            )
        )

        # self.cam_0 = self.scene.add_camera(
        #  res=(1280, 960),
        # fov=30,
        # GUI=True,
        # )
        self.ik_cache = {}  # Shared IK cache for all robots
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))
        
        
        # fixed transformation
       # self.cam_0_transform = trans_quat_to_T(np.array([0.03, 0, 0.03]), xyz_to_quat(np.array([180+5, 0, -90])))

        self.envs_idx = np.arange(self.num_envs)
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor([0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0]).to(self.device)
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

        self.hand_jnt_names = [
            "bhand_finger1",
            "bhand_finger2",
            "bhand_finger3"
        ]

        self.hand_dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.hand_jnt_names]
        self.finger_pos = torch.full((self.num_envs, 3), 0, dtype=torch.float32, device=self.device)

        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([0.9, 0.0, 0], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )

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
        self.franka.control_dofs_position(self.qpos[:, :7], self.motors_dof, self.envs_idx)


    def reset(self):
        self.build_env()
        ## random cube position
        cube_pos = np.array([1.0, 0.0, 0.04])
        x_min, x_max = 0.85, 0.95  
        y_min, y_max = -0.01, 0.01  
        random_x = np.random.uniform(x_min, x_max, size=self.num_envs)
        random_y = np.random.uniform(y_min, y_max, size=self.num_envs)
        cube_pos = np.column_stack((random_x, random_y, np.full(self.num_envs, cube_pos[2])))
        ## random cube orientation
        fixed_roll = 0
        fixed_pitch = 0
        random_yaws = np.random.uniform(0, 2 * np.pi, size=self.num_envs) 
        quaternions = np.array([euler_to_quaternion(fixed_roll, fixed_pitch, yaw) for yaw in random_yaws])
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)   
        self.cube.set_quat(quaternions, envs_idx=self.envs_idx) 

        obs1 = self.cube.get_pos()

        obs2 = (self.franka.get_link("bhand_finger1_link_2").get_pos() + self.franka.get_link("bhand_finger2_link_2").get_pos() + self.franka.get_link("bhand_finger3_link_2").get_pos()) / 3
        state = torch.concat([obs1, obs2], dim=1)
        



        return state

    def step(self, actions):

        
        action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 1 # Close gripper
        action_mask_2 = actions == 2 # Lift gripper
        action_mask_3 = actions == 3 # Lower gripper
        action_mask_4 = actions == 4 # Move left
        action_mask_5 = actions == 5 # Move right
        action_mask_6 = actions == 6 # Move forward
        action_mask_7 = actions == 7 # Move backward


        
        self.finger_pos[action_mask_0] = 0
        self.finger_pos[action_mask_1] = 1.7
        #self.finger_pos[action_mask_2] = 1.7

        
        pos = self.pos.clone()
        pos[action_mask_2, 2] += 0.05
        pos[action_mask_3, 2] -= 0.05
        # pos[action_mask_4, 0] -= 0.05
        # pos[action_mask_5, 0] += 0.05
        # pos[action_mask_6, 1] -= 0.05
        # pos[action_mask_7, 1] += 0.05

        self.pos = pos

        
        pos_key = tuple(pos.cpu().numpy().flatten())  # Convert tensor to a hashable key

        
        if pos_key in self.ik_cache:
            self.qpos = self.ik_cache[pos_key]  # Use cached solution
            print("Using Cache")
        else:
            self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
            respect_joint_limit = True
            )
            self.ik_cache[pos_key] = self.qpos  # Store in cache
            print("Calculate and Store")

        
        

        self.franka.control_dofs_position(self.qpos[:, :7], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.finger_pos, self.hand_dofs_idx, self.envs_idx)
        self.scene.step()

        #end_effector = self.franka.get_link("wam_link_7")

        #transform_matrices = trans_quat_to_T(end_effector.get_pos(), end_effector.get_quat()).cpu().numpy()

        
        #self.cam_0.set_pose(transform=transform_matrices @ self.cam_0_transform)
        

        block_position = self.cube.get_pos()

        gripper_position = (self.franka.get_link("bhand_finger1_link_2").get_pos() + self.franka.get_link("bhand_finger2_link_2").get_pos() + self.franka.get_link("bhand_finger3_link_2").get_pos()) / 3
        states = torch.concat([block_position, gripper_position], dim=1)

        #block_quat = self.cube.get_quat()  # Get block's orientation
        #gripper_quat = self.franka.get_link("bhand_finger1_link_2").get_quat()

        is_grasping = (torch.norm(gripper_position - block_position, dim=1) < 0.05)  # Close enough?
        finger_closed = (self.finger_pos[:, 0] > 1)  # Fingers mostly closed?
       
        above_block_reward = (gripper_position[:, 2] - block_position[:, 2] - 0.2).abs().clamp(max=0.05) * 10
        ground_penalty = (gripper_position[:, 2] < 0.15) * -5.0

        rewards = (
            -torch.norm(block_position - gripper_position, dim=1)  # Get closer to block
            + torch.maximum(torch.tensor(0.02), block_position[:, 2]) * 10  # Lift block
            + (is_grasping & finger_closed) * 5.0  # Grasp stability
            + above_block_reward  # Reward for being 20cm above block
            #+ ground_penalty  # Penalty for touching the ground
        )

    


        dones = block_position[:, 2] > 0.35
        return states, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = GraspRandomBlockVelEnv(vis=True)