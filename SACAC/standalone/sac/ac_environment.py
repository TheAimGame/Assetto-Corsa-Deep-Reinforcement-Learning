import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ac_controller import ACController
from ac_socket import ACSocket
from sac.utils.logx import colorize
from sac.utils.track_spline import get_heading_error, get_distance_to_center_line
class AcEnv(gym.Env):
    """
    The custom gymnasium environment for the Assetto Corsa.
    """
    metadata = {"render_modes": [], "render_fps": 0}
    _observations = None
    _invalid_flag = 0.0
    _sock = None
    def __init__(self, render_mode: Optional[str] = None, max_speed=200.0, steer_scale=[-360, 360], spline_points=[[], []]):
        self.controller = ACController(steer_scale)
        self.max_speed = max_speed
        self.observation_space = spaces.Box(
            low=np.array(
                [0.000, 0.0, -2000.0, -2000.0, -2000.0, 0.0, 1.0, 0.000, 0, 0]),
            high=np.array([1.000, max_speed, 2000.0,
                          2000.0, 2000.0, 1.0, 2.0, 1.000, 2.0 * math.pi, 500]),
            shape=(10,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.000]),
            high=np.array([1.0, 1.000]),
            shape=(2,),
            dtype=np.float32
        )
        self.spline_points = spline_points
        assert render_mode is None or render_mode in self.metadata["render_modes"] 
        self.render_mode = render_mode
    def set_sock(self, sock: ACSocket):
        self._sock = sock
    def _update_obs(self):
        """
        Get the current observation from the game over socket.
        """
        self._sock.update()
        data = self._sock.data
        try:
            data_str = data.decode('utf-8')
            data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
        except:
            print(colorize("Error parsing data, returning empty dict!", "red"))
            return {}
        track_progress = float(data_dict['track_progress'])
        speed_kmh = float(data_dict['speed_kmh'])
        world_loc_x = float(data_dict['world_loc[0]'])
        world_loc_y = float(data_dict['world_loc[1]'])
        world_loc_z = float(data_dict['world_loc[2]'])
        lap_count = float(data_dict['lap_count'])
        previous_track_progress = self._observations[0] if self._observations is not None else 0.000
        velocity_x = float(data_dict['velocity[0]'])
        velocity_z = float(data_dict['velocity[1]'])
        heading_error = get_heading_error(self.spline_points, world_loc_x, world_loc_y, np.array([
            velocity_x, velocity_z]))
        dist_offcenter = get_distance_to_center_line(
            self.spline_points, world_loc_x, world_loc_y)
        lap_invalid = self._invalid_flag
        if data_dict['lap_invalid'] == 'True':
            lap_invalid = 1.0
        self._invalid_flag = lap_invalid
        self._observations = np.array(
            [track_progress, speed_kmh, world_loc_x, world_loc_y, world_loc_z, lap_invalid, lap_count, previous_track_progress, heading_error, dist_offcenter], dtype=np.float32)
        return self._observations
    def _get_info(self):
        """
        Extra information returned by step and reset functions.
        """
        return {}
    def _get_reward_1(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward considering just speed.
        """
        speed_reward = self._observations[1] / self.max_speed
        if self._observations[5] == 1.0:
            return penalty_offtrack
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress
        if self._observations[1] <= min_speed:
            return penalty_lowspeed
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]
        return speed_reward * weight
    def _get_reward_2(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward considering speed and progress on track.
        """
        speed_reward = self._observations[1] / self.max_speed
        progress_reward = self._observations[0]
        if self._observations[5] == 1.0:
            return penalty_offtrack
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress
        if self._observations[1] <= min_speed:
            return penalty_lowspeed
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]
        return (progress_reward + speed_reward) * weight
    def _get_reward_3(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        Reward for delta progress on track.
        """
        progress = self._observations[0]
        previous_progress = self._observations[7]
        delta_progress = progress - previous_progress
        if self._observations[5] == 1.0:
            return penalty_offtrack
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress
        if self._observations[1] <= min_speed:
            return penalty_lowspeed
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]
        return delta_progress * weight
    def _get_reward_4(self, penalty_offtrack=-1.0, penalty_lowspeed=-0.4, min_speed=5.0, min_progress=0.0001, penalty_lowprogress=-0.8, progress_weights=[]):
        """
        A reward considering speed and delta progress on track.
        """
        speed_reward = self._observations[1] / self.max_speed
        progress = self._observations[0]
        previous_progress = self._observations[7]
        delta_progress = progress - previous_progress
        if self._observations[5] == 1.0:
            return penalty_offtrack
        if self._observations[0] <= self._observations[7] + min_progress:
            return penalty_lowprogress
        if self._observations[1] <= min_speed:
            return penalty_lowspeed
        weight = 1.0
        progress_point = int(self._observations[0] * 10)
        if progress_point >= 1 and progress_point <= 9:
            weight = progress_weights[progress_point - 1]
        return (delta_progress + speed_reward) * weight
    def _get_reward_5(self, weight_wrongdir=1.0, weight_offcenter=1.0, weight_extra_offcenter=1.0, weight_lowspeed=1.0, min_speed=10.0, extra_offcenter_penalty=False):
        """
        A reward considering speed, angle and distance from center of track.
        :return: The reward.
        """
        speed = self._observations[1]  
        theta = self._observations[8]
        dist_offcenter = self._observations[9]
        reward = math.cos(theta) - abs(math.sin(theta)) - \
            abs(dist_offcenter) + speed / self.max_speed
        if extra_offcenter_penalty:
            reward -= (weight_extra_offcenter * abs(dist_offcenter))
        return reward
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to initiate a new episode.
        :param seed: The seed for the environment's random number generator
        :param options: The options for the environment
        :return: The initial observation and info
        """
        super().reset(seed=seed)
        self.controller.reset_car()
        self._invalid_flag = 0.0
        observation = self._update_obs()
        info = self._get_info()
        return observation, info
    def step(self, action: np.ndarray, ignore_done: bool = False):
        self.controller.perform(action[0], action[1])
        observation = self._update_obs()
        progress_goal = 0.99
        lap_invalid = observation[5]
        lap_count = observation[6]
        track_progress = observation[0]
        if ignore_done:
            terminated = False
        else:
            terminated = lap_count > 1.0 or track_progress >= progress_goal
        truncated = False
        reward = None
        info = None
        if not ignore_done:
            reward = self._get_reward_5()
            info = self._get_info()
        return observation, reward, terminated, truncated, info
    def render(self):
        """
        not needed for AC.
        """
        print(colorize("Rendering not supported for this environment!", "red"))
    def close(self):
        """
        Close the environment and the socket connection.
        """
        self._sock.end_training()
        self._sock.on_close()