from pathlib import Path
from utils import makedirs
from shutil import copyfile
import os

path_lib = Path('lighthand/out')
w_list = sorted(list(path_lib.glob('**/*.pt')))
save_root = '/home/chenxingyu/.jupyter'

i=0
for w in w_list:
    if w.parts[-4] not in ['FreiHAND',]:
        continue
    if w.parts[-1] not in ['checkpoint_last.pt']:
        continue
    save_dir = os.path.join(save_root, *w.parts[:-1])
    if not os.path.exists(save_dir):
        makedirs(save_dir)
    save_file = os.path.join(save_dir, w.parts[-1])
    print(str(w), save_file)
    copyfile(str(w), save_file)
    i += 1
print(i)
