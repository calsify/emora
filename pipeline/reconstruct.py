import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
image, out = os.path.abspath(a.image), os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
DECA_DIR = os.path.abspath(os.environ.get("DECA_DIR", "third_party/DECA"))
sys.path.insert(0, DECA_DIR); os.chdir(DECA_DIR)

import numpy as np, torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg

if not image.lower().endswith((".jpg", ".png", ".bmp")):
    sys.exit("DECA only accepts .jpg / .png / .bmp (not .jpeg). Rename the file.")

cfg.model.use_tex = False
cfg.rasterizer_type = "pytorch3d"
deca = DECA(config=cfg, device="cuda")
img = datasets.TestData(image, iscrop=True, face_detector="fan")[0]["image"].to("cuda")[None]

with torch.no_grad():
    code = deca.encode(img)
    opdict, _ = deca.decode(code)

    v0, _, _ = deca.flame(shape_params=code["shape"],
                          expression_params=torch.zeros_like(code["exp"]),
                          pose_params=torch.zeros_like(code["pose"]))
    opdict["verts"] = v0

    m = deca.uv_face_eye_mask
    tex = opdict["uv_texture_gt"]
    skin = tex[0][:, m[0, 0] > 0.5].median(dim=1).values.view(1, 3, 1, 1)
    core = 1 - F.max_pool2d(1 - m, 15, 1, 7)
    core = gaussian_blur(core, kernel_size=[25, 25], sigma=[8.0, 8.0])
    opdict["uv_texture_gt"] = tex * core + skin * (1 - core)

obj = os.path.join(out, "neutral.obj")
deca.save_obj(obj, opdict)
d = obj.replace(".obj", "_detail.obj")
if os.path.exists(d):
    os.remove(d)
np.savez(os.path.join(out, "codes.npz"),
         shape=code["shape"][0].cpu().numpy(), exp=code["exp"][0].cpu().numpy())
print("done ->", out)
