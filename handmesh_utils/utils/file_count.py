import os
from pathlib import Path
import pickle

path = '/share/group_guoxiaoyan/group_hand/chenxingyu/PanoHand_release/trans_pose_batch1'

path_lib = Path(path)
# png_list = sorted(list(path_lib.glob('**/*.png'))) 
# jpg_list = sorted(list(path_lib.glob('**/*.jpg'))) 
# heic_list = sorted(list(path_lib.glob('**/*.HEIC')))
# jpeg_list = sorted(list(path_lib.glob('**/*.jpeg')))
# JPG_list = sorted(list(path_lib.glob('**/*.JPG')))

# print(len(png_list), len(jpg_list), len(jpeg_list), len(JPG_list), len(heic_list), len(png_list)+len(jpg_list)+len(heic_list)+len(jpeg_list)+len(JPG_list))
# with open('/share/group_guoxiaoyan/shared_data_for_interns/hand/wrist_wo_wristband/annotations/wrist_back_pass_with_mesh.pkl', 'rb') as f:
#     anno1 = pickle.load(f, encoding='utf8')
# with open('/share/group_guoxiaoyan/shared_data_for_interns/hand/code/wrist/scripts/mesh_plot/annotations/wrist_wo_wristband_front_all_mesh_and_keypoint_with_mesh_with_mask_all_no_pass.pkl', 'rb') as f:
#     anno2 = pickle.load(f, encoding='utf8')
# with open('/share/group_guoxiaoyan/shared_data_for_interns/hand/code/wrist/scripts/mesh_plot/annotations/wrist_wo_wristband_front_all_mesh_and_keypoint_with_mesh_with_mask_all_pass_wrist_mask.pkl', 'rb') as f:
#     anno3 = pickle.load(f, encoding='utf8')

# print(len(anno1.keys()), len(anno2.keys()), len(anno3.keys()))

img_list = sorted(list(path_lib.glob('**/pic256/**/*.png')))
img_list_ori = [img_path for img_path in img_list if '215' not in img_path.parts[-1]]
img_list_sample = [img_path for img_path in img_list if int(img_path.parts[-1].split('.')[1]) % 3 !=0 ]

print(len(img_list_ori), len(img_list_sample))

for p in img_list_sample:
    img_name = p.parts[-1]
    img_name_split = img_name.split('.')
    num = int(img_name_split[1])
    path = str(p)
    mesh_path = os.path.join(*p.parts[:-2], str(num) + '.obj').replace('pic256', 'model_mano')
    mask_path = os.path.join(mesh_path.replace('obj', 'png').replace('model_mano', 'mask256'))
    # print(path, mesh_path, mask_path)
    os.remove(path)
    try:
        os.remove(mesh_path)
    except:
        print(f'fail to remove {mesh_path}')
    try:
        os.remove(mask_path)
    except:
        print(f'fail to remove {mask_path}')