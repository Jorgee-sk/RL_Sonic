import retro
import gym
from gym.utils import seeding
from preprocess_frame import PreprocessFrame, FireResetEnv, MaxAndSkipEnv, FrameStackEnv, ClipRewardEnv
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

def make_env():
    env = retro.make(game='SonicTheHedgehog-Sms', state='Level1')
    return env


def main():
    env = make_env()
    env = SonicDiscretizer(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessFrame(env, shape=(84,84))
    env = FrameStackEnv(env, k=4)
    env = ClipRewardEnv(env)
    env = FireResetEnv(env)

    window_name = 'Sonic RL - Preprocessed'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    done = True

    for step in range(1_000_000):

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if done:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                _, info = reset_result
            done = False

        action = env.action_space.sample()
        result = env.step(action)
        state, reward, done, info = (result if len(result) == 4 else (*result, {}))
        print(f"Step {step} → shape={state.shape}, dtype={state.dtype}, "
              f"min={state.min():.3f}, max={state.max():.3f}, reward={reward}, info={info}, action={action}")

        img = (state[:, :, 0] * 255).astype('uint8')
        cv2.imshow(window_name, img)

    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()