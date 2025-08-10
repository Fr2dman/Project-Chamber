# rl/export_policy.py
import torch, pickle
from stable_baselines3 import SAC

m = SAC.load("checkpoints/sb3_sac/model.zip", device="cpu")
obs_dim = m.policy.observation_space.shape[0]

class Scripted(torch.nn.Module):
    def __init__(self, pol): super().__init__(); self.fe=pol.features_extractor; self.mlp=pol.mlp_extractor; self.mu=pol.actor.mu
    def forward(self, x): z=self.fe(x); pi,_=self.mlp(z); a=torch.tanh(self.mu(pi)); return a

ts = torch.jit.trace(Scripted(m.policy).eval(), torch.zeros(1, obs_dim))
ts.save("device/policy/sac_policy.ts")

with open("checkpoints/sb3_sac/vecnorm.pkl","rb") as f: stats=pickle.load(f)
with open("device/policy/vecnorm.pkl","wb") as f: pickle.dump(stats,f)
