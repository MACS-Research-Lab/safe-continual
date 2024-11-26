# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""HalfCheetah environment with a safety constraint on velocity and feet fault nonstationarity."""

from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer

import numpy as np

TASK_CYCLE = ['nominal', 'back', 'nominal', 'front', 'back', 'nominal', 'front'] # recommend 8 million

TASK_NUMS = {
    'nominal': 0,
    'back': 1,
    'front': 2
}

TASK_LENGTH = 1_000_000 / 10 # divide by number of parallel processes
# TASK_LENGTH = 40_000 / 10 # divide by number of parallel processes
# TASK_LENGTH = 250 / 10 # divide by number of parallel processes
NOMINAL_FRONT_FOOT = np.array([0.046, 0.07,  0.   ])
NOMINAL_FRONT_SHIN = np.array([0.046, 0.106,  0.   ])
NOMINAL_FRONT_THIGH = np.array([0.046, 0.133,  0.   ])

FRONT_FOOT_POSITION = np.array([0.13,  0.,   -0.18])
FRONT_SHIN_POSITION = np.array([-0.14,  0.,   -0.24])
FRONT_THIGH_POSITION = np.array([0.5, 0.,  0. ])

FRONT_FOOT_MASS = np.array([0.88451883])
FRONT_SHIN_MASS = np.array([1.20083682])
FRONT_THIGH_MASS = np.array([1.43807531])

NOMINAL_BACK_FOOT = np.array([0.046, 0.094, 0.   ])
NOMINAL_BACK_SHIN = np.array([0.046, 0.15, 0.   ])
NOMINAL_BACK_THIGH = np.array([0.046, 0.145, 0.   ])

BACK_FOOT_POSITION = np.array([-0.28,  0.,   -0.14])
BACK_SHIN_POSITION = np.array([0.16,  0.,   -0.25])
BACK_THIGH_POSITION = np.array([-0.5,  0.,   0. ])

BACK_FOOT_MASS = np.array([1.09539749])
BACK_SHIN_MASS = np.array([1.5874477])
BACK_THIGH_MASS = np.array([1.54351464])



class SafetyHalfCheetahVelocityEnv(HalfCheetahEnv):
    """HalfCheetah environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        self.current_task = 0
        self.steps_since_change = 0
        self.current_task_name = TASK_CYCLE[self.current_task]
        super().__init__(**kwargs)
        self._velocity_threshold = 3.2096 # 2.8795
        self.model.light(0).castshadow = False
        self.task_nums = TASK_NUMS

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
        }

        cost = float(x_velocity > self._velocity_threshold)

        if self.mujoco_renderer.viewer:
            clear_viewer(self.mujoco_renderer.viewer)
            add_velocity_marker(
                viewer=self.mujoco_renderer.viewer,
                pos=self.get_body_com('torso')[:3].copy(),
                vel=x_velocity,
                cost=cost,
                velocity_threshold=self._velocity_threshold,
            )
        if self.render_mode == 'human':
            self.render()

        self.check_task()

        self.steps_since_change += 1
        return observation, reward, cost, terminated, False, info
    
    def check_task(self):
        if self.steps_since_change > TASK_LENGTH:
            self.steps_since_change = 0
            self.current_task = (self.current_task + 1) % len(TASK_CYCLE)
            self.change_task()

    def change_task(self):
        task_name = TASK_CYCLE[self.current_task]
        self.current_task_name = task_name

        # Always return to nominal first
        self.set_nominal()
        if task_name == 'nominal':
            # Do nothing, we only need to set it to nominal
            pass
        elif task_name == 'front':
            self.model.geom('ffoot').size = NOMINAL_FRONT_FOOT * 0.0001
            self.model.geom('fshin').size = NOMINAL_FRONT_SHIN * 0.0001
            self.model.geom('fthigh').size = NOMINAL_FRONT_THIGH * 0.0001

            self.model.body('ffoot').pos = FRONT_FOOT_POSITION * 0
            self.model.body('fshin').pos = FRONT_SHIN_POSITION * 0
            self.model.body('fthigh').pos = FRONT_THIGH_POSITION * 0

            self.model.body('ffoot').mass = FRONT_FOOT_MASS * 0.0001
            self.model.body('fshin').mass = FRONT_SHIN_MASS * 0.0001
            self.model.body('fthigh').mass = FRONT_THIGH_MASS * 0.0001
        elif task_name == 'back':
            self.model.geom('bfoot').size = NOMINAL_BACK_FOOT * 0.0001
            self.model.geom('bshin').size = NOMINAL_BACK_SHIN * 0.0001
            self.model.geom('bthigh').size = NOMINAL_BACK_THIGH * 0.0001

            self.model.body('bfoot').pos = BACK_FOOT_POSITION * 0
            self.model.body('bshin').pos = BACK_SHIN_POSITION * 0
            self.model.body('bthigh').pos = BACK_THIGH_POSITION * 0

            self.model.body('bfoot').mass = BACK_FOOT_MASS * 0.0001
            self.model.body('bshin').mass = BACK_SHIN_MASS * 0.0001
            self.model.body('bthigh').mass = BACK_THIGH_MASS * 0.0001
        else:
            raise NotImplementedError(f"Task type {task_name} not defined")


        self.reset()

    def set_nominal(self):
        self.model.geom('ffoot').size = NOMINAL_FRONT_FOOT
        self.model.geom('fshin').size = NOMINAL_FRONT_SHIN
        self.model.geom('fthigh').size = NOMINAL_FRONT_THIGH
        self.model.geom('bfoot').size = NOMINAL_BACK_FOOT
        self.model.geom('bshin').size = NOMINAL_BACK_SHIN
        self.model.geom('bthigh').size = NOMINAL_BACK_THIGH

        self.model.body('ffoot').pos = FRONT_FOOT_POSITION
        self.model.body('fshin').pos = FRONT_SHIN_POSITION
        self.model.body('fthigh').pos = FRONT_THIGH_POSITION
        self.model.body('bfoot').pos = BACK_FOOT_POSITION
        self.model.body('bshin').pos = BACK_SHIN_POSITION
        self.model.body('bthigh').pos = BACK_THIGH_POSITION

        self.model.body('ffoot').mass = FRONT_FOOT_MASS
        self.model.body('fshin').mass = FRONT_SHIN_MASS
        self.model.body('fthigh').mass = FRONT_THIGH_MASS
        self.model.body('bfoot').mass = BACK_FOOT_MASS
        self.model.body('bshin').mass = BACK_SHIN_MASS
        self.model.body('bthigh').mass = BACK_THIGH_MASS
