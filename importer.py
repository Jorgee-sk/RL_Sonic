import retro
import gym
from gym.utils import seeding
import numpy as np
import cv2

if not hasattr(seeding, "hash_seed"):
    def hash_seed(seed):
        # misma fórmula que gym<=0.21
        return abs(int(seed * 0x5DEECE66D + 0xB) % (2**31))
    seeding.hash_seed = hash_seed


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SonicDiscretizer(Discretizer):
    """
    Sonic specific discrete actions
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']])

def main():
    env = retro.make(game='SonicTheHedgehog-Sms', state='Level1')
    env = SonicDiscretizer(env)
    cv2.namedWindow('Sonic RL', cv2.WINDOW_AUTOSIZE)
    done = True
    
    for step in range(1000000):
        
        if done:
            env.reset()
        if (cv2.waitKey(1) & 0xFF == ord('q')) | (cv2.getWindowProperty('Sonic RL', cv2.WND_PROP_VISIBLE) < 1):
            env.reset()
            break

        state, reward, done, info = env.step(env.action_space.sample())

        bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        cv2.imshow("Sonic RL", bgr)
    
    cv2.destroyAllWindows()
    env.close()


if __name__ == '__main__':
    main()