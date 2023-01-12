"""Implementation of the Hopper environment supporting
domain randomization optimization."""
from collections import deque
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

from gym.wrappers.pixel_observation import PixelObservationWrapper

from matplotlib import pyplot as plt

try:
    import cv2
except:
    print("No CV2")

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0


    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_parameters())

        return self._get_obs()

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        m1 = self.model.body_mass[1]


        #+/- 0.5
        m2 = np.random.uniform(3.42699082, 4.42699082)
        m3 = np.random.uniform(2.31433605,3.31433605)
        m4 = np.random.uniform(4.5893801,5.5893801)

        masses = np.array([m1, m2, m3, m4])

        # print("Masses: ",masses)

        return masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        # DOMAIN RANDOMIZATION!
        self.set_random_parameters()

        # print(self.get_parameters())
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



def my_make_env(PixelObservation = True, stack_frames = 4, scale=False, domain = "source"):
    """Configure the environment."""
    assert domain in ['source', 'target'], f"Please choose a domain in ['source', 'target']. The domain {domain} is not valid!"
    if PixelObservation:
        if domain == 'source': 
            env = PixelObservationWrapper(gym.make('CustomHopper-source-v0'))
        else:
            env = PixelObservationWrapper(gym.make('CustomHopper-target-v0'))

        
        env = WarpFrame(env)
        

        
        if scale:
            env = ScaledFloatFrame(env)
        if stack_frames > 1:
            env = FrameStack(env, stack_frames)

        env = ImageToPyTorch(env)
        
    else:
        if domain == 'source': 
            env = gym.make('CustomHopper-source-v0')
        else:
            env = gym.make('CustomHopper-target-v0')
    return env


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 224
        self.height = 224
        self.channel = 3
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, self.channel), dtype=np.uint8)

    def observation(self, frame):
        '''
        This function retrieves a resized frame of shape (`width` x `heigt`), converted from _RGB_ to _GRAY Scale_

        Parameters
        -----------
        `frame`: ndarray
            Input src image
        
        `width`: int
            Value representing the width of the image

        `height`: int
            Value representing the height of the image

        Returns
        ----------
        Resized frame: ndarray
                Output resized image, with shape (`width` x `heigt`), converted from _RGB_ to _GRAY Scale_
        '''
        if type(frame) != np.ndarray:
            frame = frame['pixels']

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return frame


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            if type(ob) != np.ndarray:
                self.frames.append(ob['pixels'])
            else:
                self.frames.append(ob)

        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        if type(ob) != np.ndarray: 
            self.frames.append(ob['pixels'])
        else:
            self.frames.append(ob)
        
            
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        if type(observation) != np.ndarray:
            return np.array(observation['pixels']).astype(np.float32) / 255.0
        else:
            return np.array(observation).astype(np.float32) / 255.0
        # observation['pixels'] = np.array(observation['pixels']).astype(np.float32) / 255.0
        # return observation


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        # if type(frames[0]) != np.ndarray:
        #     self._frames = [frame['pixels'] for frame in frames]
        # else:
        #     self._frames = frames
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
            
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        frame = np.swapaxes(observation, 2, 0)[None,:,:,:]
        
        return frame
        # return np.swapaxes(observation, 2, 0)

"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

