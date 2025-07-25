import cv2
import numpy as np
import gym
from gym import ObservationWrapper
from gym.spaces import Box
from collections import deque

class PreprocessFrame(ObservationWrapper):
    """
    Convierte cada frame RGB a escala de grises, lo redimensiona a (84x84)
    y lo normaliza a [0,1], devolviendo un array float32 de shape (84,84,1).
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(shape[0], shape[1], 1),
            dtype=np.float32
        )
    
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized[:, :, None]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self.observation(obs), info
        else:
            return self.observation(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return self.observation(obs), reward, done, info
        else:
            obs, *rest = result
            return (self.observation(obs), *rest)
        
class MaxAndSkipEnv(gym.Wrapper):
    """
    Salta `skip` frames y devuelve el max-pool de los dos últimos.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        shape = env.observation_space.shape
        self._buf = np.zeros((2, *shape), dtype=env.observation_space.dtype)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            result = self.env.step(action)
            # unpack flexible retornos
            if len(result) == 4:
                obs, r, done, info = result
            else:
                obs, r, done, info = result[0], result[1], result[2], result[3]
            self._buf[i % 2] = obs
            total_reward += r
            if done:
                break
        max_frame = np.maximum(self._buf[0], self._buf[1])
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    """
    Acota las recompensas a {-1, 0, +1}.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        out = self.env.step(action)
        # Si la salida tiene 5 valores (Gymnasium)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            reward = np.sign(reward)
            return obs, reward, terminated, truncated, info
        # Si la salida tiene 4 valores (Gym clásico)
        else:
            obs, reward, done, info = out
            reward = np.sign(reward)
            return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def reward(self, reward):
        return np.sign(reward)
    

class FrameStackEnv(gym.Wrapper):
    """
    Wrapper personalizado que apila los últimos k frames ya preprocesados.
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(shp[0], shp[1], k),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        for _ in range(self.k):
            self.frames.append(obs)
        stacked = np.stack(self.frames, axis=2)
        return (stacked, info) if isinstance(result, tuple) else stacked

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        stacked = np.stack(self.frames, axis=2)
        return stacked, reward, done, info

class FireResetEnv(gym.Wrapper):
    """
    Ejecuta la acción 'B' (fire) tras reset para iniciar el juego,
    y soporta env.step() que devuelva 4 o 5 valores.
    """
    def __init__(self, env, fire_action=6):
        super().__init__(env)
        self.fire_action = fire_action

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        out = self.env.step(self.fire_action)

        if len(out) == 5:
            obs_f, reward, terminated, truncated, info_f = out
            done = terminated or truncated
        else:
            obs_f, reward, done, info_f = out

        info = {**info, **info_f}
        obs = obs_f

        if done:
            result2 = self.env.reset(**kwargs)
            if isinstance(result2, tuple):
                obs, info2 = result2
            else:
                obs, info2 = result2, {}
            info.update(info2)

        return (obs, info)