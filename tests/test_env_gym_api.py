from rl.sb3.make_env import make_env
def test_spaces_and_step():
    env = make_env({"num_zones":4})()
    obs, _ = env.reset()
    assert obs.shape[0] > 0
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert obs.shape[0] > 0
