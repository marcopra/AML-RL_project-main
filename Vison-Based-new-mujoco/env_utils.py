import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

from matplotlib import pyplot as plt
from IPython.display import clear_output
import cv2

cv2.ocl.setUseOpenCL(False) # disable GPU usage by OpenCV

class CustomHopperWrapper(gym.Wrapper):
    """Custom Wrapper for the Hopper-v4 gym environment

    - Adds domain randomization capabilities
    - Handles source and target domains

    See https://gymnasium.farama.org/api/wrappers/
    """

    def __init__(self, env, domain):
        super().__init__(env)
        assert self.env.unwrapped.spec.id == 'Hopper-v4'

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.env.model.body_mass[1] -= 1.0

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """

        m1 = self.model.body_mass[1]
        m2 = np.random.uniform(1.5, 3)
        m3 = np.random.uniform(2,3)
        m4 = np.random.uniform(2,3)

        masses = np.array([m1, m2, m3, m4])        
        return masses
        # raise NotImplementedError()

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.env.model.body_mass[1:] )
        return masses

    def set_parameters(self, masses):
        """Set each hopper link's mass to a new value"""
        self.env.model.body_mass[1:] = masses


def make_env(domain, render_mode=None):
    """Returns the wrapped Hopper-v4 environment"""
    assert domain in ['source', 'target']

    env = gym.make('Hopper-v4', render_mode=render_mode)
    env = gym.wrappers.StepAPICompatibility(env, output_truncation_bool=False)  # Retro-compatibility for stable-baselines3
    env = CustomHopperWrapper(env, domain=domain)  # Use custom implementation for source/target variants
    
    return env


def my_make_env(stack_frames = 4, scale=False, domain = "source"):
    """Configure the environment."""
    env = PixelObservationWrapper(make_env(domain=domain, render_mode='rgb_array'))
    # assert 'NoFrameskip' in env.spec.id

    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if stack_frames > 1:
        env = FrameStack(env, stack_frames)

    env = ImageToPyTorch(env)
    return env


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

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
            
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


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
        ob, _ = self.env.reset()
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


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        if type(frames[0]) != np.ndarray:
            self._frames = [frame['pixels'] for frame in frames]
        else:
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
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)