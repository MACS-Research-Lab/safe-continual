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
"""Ant environment with a safety constraint on velocity."""

from safety_gymnasium.tasks.safe_velocity.safety_ant_velocity_v0 import (
    SafetyAntVelocityEnv as AntEnv,
)

from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer

import numpy as np

import mujoco

TASK_CYCLE = ['nominal', 'back', 'nominal', 'front', 'back', 'nominal', 'front'] # recommend 8 million
TASK_LENGTH = 1_000_000 / 10 # divide by number of parallel processes
# TASK_LENGTH = 250 / 10 # divide by number of parallel processes
TASK_NUMS = {
    'nominal': 0,
    'back': 1,
    'front': 2
}

LEG_SIZE = np.array([0.08, 0.14142136, 0.])
LEG_MASS = np.array([0.03915775])

AUX_SIZE = np.array([0.08,       0.14142136, 0.        ])
AUX_MASS = np.array([0.03915775])

AUX1_POS = np.array([0.2, 0.2, 0. ])

class SafetyAntVelocityEnv(AntEnv):
    """Ant environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        self.current_task = 0
        self.steps_since_change = 0
        self.current_task_name = TASK_CYCLE[self.current_task]
        super().__init__(**kwargs)
        self._velocity_threshold = 2.6222
        self.model.light(0).castshadow = False
        self.task_nums = TASK_NUMS

    def step(self, action):  # pylint: disable=too-many-locals
        xy_position_before = self.get_body_com('torso')[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com('torso')[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_survive': healthy_reward,
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),
            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info['reward_ctrl'] = -contact_cost

        reward = rewards - costs

        velocity = np.sqrt(x_velocity**2 + y_velocity**2)
        cost = float(velocity > self._velocity_threshold)

        if self.mujoco_renderer.viewer:
            clear_viewer(self.mujoco_renderer.viewer)
            add_velocity_marker(
                viewer=self.mujoco_renderer.viewer,
                pos=self.get_body_com('torso')[:3].copy(),
                vel=velocity,
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
        LEG_SIZE_SCALE = 0.0001
        if self.current_task_name == 'back':
            self.model.geom("aux_3_geom").size /= LEG_SIZE_SCALE       # left_back_leg
            self.model.geom("back_leg_geom").size /= LEG_SIZE_SCALE
            self.model.geom("third_ankle_geom").size /= LEG_SIZE_SCALE

            self.model.geom("aux_4_geom").size /= LEG_SIZE_SCALE       # right_back_leg
            self.model.geom("rightback_leg_geom").size /= LEG_SIZE_SCALE
            self.model.geom("fourth_ankle_geom").size /= LEG_SIZE_SCALE
        elif self.current_task_name == 'front':
            self.model.geom("aux_1_geom").size /= LEG_SIZE_SCALE       # front_left_leg
            self.model.geom("left_leg_geom").size /= LEG_SIZE_SCALE
            self.model.geom("left_ankle_geom").size /= LEG_SIZE_SCALE

            self.model.geom("aux_2_geom").size /= LEG_SIZE_SCALE
            self.model.geom("right_leg_geom").size /= LEG_SIZE_SCALE
            self.model.geom("right_ankle_geom").size /= LEG_SIZE_SCALE

        task_name = TASK_CYCLE[self.current_task]
        self.current_task_name = task_name

        # Always return to nominal first
        # self.set_nominal() # TODO: move the code above
        if task_name == 'nominal':
            # Do nothing, we only need to set it to nominal
            pass
        elif task_name == 'front':
            self.model.geom("aux_1_geom").size *= LEG_SIZE_SCALE       # front_left_leg
            self.model.geom("left_leg_geom").size *= LEG_SIZE_SCALE
            self.model.geom("left_ankle_geom").size *= LEG_SIZE_SCALE

            self.model.geom("aux_2_geom").size *= LEG_SIZE_SCALE
            self.model.geom("right_leg_geom").size *= LEG_SIZE_SCALE
            self.model.geom("right_ankle_geom").size *= LEG_SIZE_SCALE
        elif task_name == 'back':
            self.model.geom("aux_3_geom").size *= LEG_SIZE_SCALE       # left_back_leg
            self.model.geom("back_leg_geom").size *= LEG_SIZE_SCALE
            self.model.geom("third_ankle_geom").size *= LEG_SIZE_SCALE

            self.model.geom("aux_4_geom").size *= LEG_SIZE_SCALE       # right_back_leg
            self.model.geom("rightback_leg_geom").size *= LEG_SIZE_SCALE
            self.model.geom("fourth_ankle_geom").size *= LEG_SIZE_SCALE
        else:
            raise NotImplementedError(f"Task type {task_name} not defined")


        self.reset()

    def set_nominal(self):
        self.model.geom('back_leg_geom').size = LEG_SIZE
        self.model.geom('rightback_leg_geom').size = LEG_SIZE
        self.model.geom('left_leg_geom').size = LEG_SIZE
        self.model.geom('right_leg_geom').size = LEG_SIZE

        self.model.body('back_leg').mass = LEG_MASS
        self.model.body('right_back_leg').mass = LEG_MASS
        self.model.body('front_left_leg').mass = LEG_MASS
        self.model.body('front_right_leg').mass = LEG_MASS