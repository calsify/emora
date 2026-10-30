import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--caricature", type=float, default=1.4)
ap.add_argument("--eye", type=float, default=1.25)
ap.add_argument("--mouth", type=float, default=1.0)
ap.add_argument("--nose", type=float, default=1.0)
a = ap.parse_args()
out = os.path.abspath(a.out)
DECA_DIR = os.path.abspath(os.environ.get("DECA_DIR", "third_party/DECA"))
sys.path.insert(0, DECA_DIR); os.chdir(DECA_DIR)

import numpy as np, torch
from decalib.models.FLAME import FLAME
from decalib.utils.config import cfg

dev = "cuda" if torch.cuda.is_available() else "cpu"
flame = FLAME(cfg.model).to(dev).eval()
shape = torch.tensor(np.load(f"{out}/codes.npz")["shape"], dtype=torch.float32, device=dev)[None] * a.caricature
Z_EXP, Z_POSE = torch.zeros(1, 50, device=dev), torch.zeros(1, 6, device=dev)

@torch.no_grad()
def run(exp=Z_EXP, pose=Z_POSE):
    v, _, lm = flame(shape_params=shape, expression_params=exp, pose_params=pose)
    return v[0].cpu().numpy(), lm[0].cpu().numpy()

V0, L0 = run()

def make_stylizer():
    regions = [(range(36, 42), 0.028, a.eye), (range(42, 48), 0.028, a.eye),
               (range(48, 68), 0.045, a.mouth), (range(27, 36), 0.035, a.nose)]
    R = []
    for idx, radius, s in regions:
        c = L0[list(idx)].mean(0)
        w = np.exp(-(np.linalg.norm(V0 - c, axis=1) / radius) ** 2)[:, None]
        R.append((c, w, s))
    def f(V):
        o = V.copy()
        for c, w, s in R:
            o = o + (o - c) * (s - 1) * w
        return o
    return f
style = make_stylizer()
S0 = style(V0)

J = np.zeros((68 * 3, 50))
for i in range(50):
    e = Z_EXP.clone(); e[0, i] = 1.0
    J[:, i] = (run(exp=e)[1] - L0).ravel()
lam = 0.05 * np.trace(J.T @ J) / 50
def solve(d):
    x = np.linalg.solve(J.T @ J + lam * np.eye(50), J.T @ d)
    return np.clip(x, -3, 3)

cx = L0[:, 0].mean()
side = lambda i: np.sign(L0[i, 0] - cx)
UP, LO, CORNERS = [49,50,51,52,53,61,62,63], [55,56,57,58,59,65,66,67], [48,54,60,64]
def smile(d):
    for i in CORNERS: d[i] = [side(i)*0.007, 0.008, 0]
def pucker(d):
    for i in CORNERS: d[i] = [-side(i)*0.012, 0, 0.006]
    for i in UP + LO: d[i, 2] = 0.008
def funnel(d):
    for i in CORNERS: d[i] = [-side(i)*0.008, 0, 0.005]
    for i in UP: d[i] = [0, 0.004, 0.008]
    for i in LO: d[i] = [0, -0.004, 0.008]
def close(d):
    for i in (61,62,63): d[i, 1] = -0.003
    for i in (65,66,67): d[i, 1] = 0.003
def roll_lower(d):
    for i in LO: d[i] = [0, 0.008, -0.004]
def upper_up(d):
    for i in UP: d[i, 1] = 0.004
def blink(d):
    for up, lo in ((37,41), (38,40), (43,47), (44,46)): d[up] = L0[lo] - L0[up]
def brow_up(d):
    for i in (21, 22): d[i, 1] = 0.008
    for i in (19, 20, 23, 24): d[i, 1] = 0.003
TARGETS = {"mouthSmile": smile, "mouthPucker": pucker, "mouthFunnel": funnel, "mouthClose": close,
           "mouthRollLower": roll_lower, "mouthUpperUp": upper_up, "eyeBlink": blink, "browInnerUp": brow_up}

names, deltas = [], []
def add(name, V):
    names.append(name)
    deltas.append((style(V) - S0).astype(np.float32))

jaw = torch.tensor([[0, 0, 0, 0.30 * float(os.environ.get("JAW_SIGN", 1)), 0, 0]], device=dev)
add("jawOpen", run(pose=jaw)[0])
for name, fn in TARGETS.items():
    d = np.zeros((68, 3)); fn(d)
    x = solve(d.ravel())
    add(name, run(exp=torch.tensor(x, dtype=torch.float32, device=dev)[None])[0])

np.savez(f"{out}/morphs.npz", neutral=S0.astype(np.float32),
         deltas=np.stack(deltas), names=np.array(names))
print("targets:", names)
