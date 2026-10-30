import sys, bpy, numpy as np
obj_path, npz_path, glb_path = sys.argv[sys.argv.index("--") + 1:][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='NEGATIVE_Z', up_axis='Y')
except AttributeError:
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y')
ob = [o for o in bpy.data.objects if o.type == 'MESH'][0]
me = ob.data

d = np.load(npz_path)
neutral, deltas, names = d["neutral"], d["deltas"], [str(n) for n in d["names"]]
assert len(me.vertices) == len(neutral), f"vertex count mismatch {len(me.vertices)} vs {len(neutral)}"
to_blender = lambda a: np.stack([a[..., 0], -a[..., 2], a[..., 1]], -1).astype(np.float32)

me.vertices.foreach_set("co", to_blender(neutral).ravel())
ob.shape_key_add(name="Basis", from_mix=False)
for n, dl in zip(names, deltas):
    sk = ob.shape_key_add(name=n, from_mix=False)
    sk.data.foreach_set("co", to_blender(neutral + dl).ravel())
for p in me.polygons:
    p.use_smooth = True
me.update()
ob.name = "Head"
bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', export_morph=True,
                          export_yup=True, export_apply=False)
print("exported", glb_path)
