import os
from pathlib import Path

path_lib = Path('/share/group_guoxiaoyan/group_hand/chenxingyu/PanoHand/result/model')


mtl_list = sorted(list(path_lib.glob('**/*.mtl')))
obj_list = sorted(list(path_lib.glob('**/*.obj')))

print(len(mtl_list), len(obj_list))

for f in mtl_list:
    os.remove(str(f))