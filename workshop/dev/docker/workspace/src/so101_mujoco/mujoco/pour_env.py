import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import math

class So101PourEnv(gym.Env):
    """
    Gymnasium environment for the SO-101 robot pouring task.
    State: Joint angles, Object 3D coords, Water levels.
    Action: Delta joint targets + Gripper ctrl.
    """
    def __init__(self, xml_path="scene.xml"):
        super(So101PourEnv, self).__init__()
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        
        # Action space: 5 ARM joints + 1 Gripper (Continuous) [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # Observation space: 
        # [qpos_arm(5), qvel_arm(5), gripper_width(1), bottle_pos(3), beaker_pos(3), beaker_ml(1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.grasp_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.bottle_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")
        self.beaker_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")
        
        self.beaker_ml = 0.0
        self.target_ml = 100.0
        self.max_steps = 500
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        self.beaker_ml = 0.0
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Apply action as delta joint targets (scaled)
        delta = action[:5] * 0.05
        self.d.ctrl[:5] += delta
        self.d.ctrl[5] = action[5] # Gripper absolute
        
        # Step simulation
        mujoco.mj_step(self.m, self.d)
        
        # Simulated water flow
        tilt = self._get_tilt_deg()
        if tilt > 30:
            flow = 25.0 * min(1.0, (tilt-30)/60.0) * self.m.opt.timestep
            self.beaker_ml += flow
            
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.beaker_ml >= 150 or self.current_step >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        qpos = self.d.qpos[:5]
        qvel = self.d.qvel[:5]
        grip = np.array([self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]]])
        b_pos = self.d.site_xpos[self.bottle_site]
        g_pos = self.d.site_xpos[self.beaker_site]
        return np.concatenate([qpos, qvel, grip, b_pos, g_pos, [self.beaker_ml]]).astype(np.float32)

    def _compute_reward(self):
        # 1. Distance to bottle (while not grasped)
        dist = np.linalg.norm(self.d.site_xpos[self.grasp_site] - self.d.site_xpos[self.bottle_site])
        reward = -dist
        
        # 2. Beaker fill reward (if pouring)
        if self.beaker_ml > 0:
            reward += 10.0 * (1.0 - abs(self.beaker_ml - self.target_ml)/self.target_ml)
            
        # 3. Penalty for overspill
        if self.beaker_ml > 110:
            reward -= 50.0
            
        return reward

    def _get_tilt_deg(self):
        b_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
        b_qadr = self.m.jnt_qposadr[self.m.body_jntadr[b_id]]
        quat = self.d.qpos[b_qadr+3:b_qadr+7]
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return math.degrees(math.acos(np.clip(np.dot(rot[:, 2], [0, 0, 1]), -1, 1)))

if __name__ == "__main__":
    env = So101PourEnv()
    obs, _ = env.reset()
    print(f"Env initialized. Obs shape: {obs.shape}")
    for _ in range(10):
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        print(f"Reward: {reward:.2f}")
