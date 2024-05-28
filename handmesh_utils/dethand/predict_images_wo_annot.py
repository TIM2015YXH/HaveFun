import os
import random
import sys
import torch
import cv2
import numpy as np
import glob
from utils.decode import ctdet_decode
from utils.post_process import ctdet_post_process_test
# from datasets.augmentations import SSDAugmentation
from single_hourglass import PoseResNet, base_model
import time
import json
import pickle
import vctoolkit as vc
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
        
def pad_image(image):
    size = image.shape
    max_length = max(size)
    padding_image = np.zeros((max_length, max_length, 3), dtype=np.uint8)
    flag = 0
    if size[0] > size[1]:
        start_index = (size[0]-size[1]) // 2
        padding_image[:, start_index: start_index + size[1], :] = image
        flag = 1
    elif size[1] > size[0]:
        start_index = (size[1] - size[0]) // 2
        padding_image[start_index: start_index + size[0], :, :] = image
        flag = 2
    else:
        padding_image[:] = image
        start_index = 0
    return padding_image, start_index, flag

the_device = torch.device("cpu")
mean = np.asarray([0.40789655, 0.44719303, 0.47026116], dtype=np.float32).reshape(1, 1, 3)
std = np.asarray([0.2886383, 0.27408165, 0.27809834], dtype=np.float32).reshape(1, 1, 3)
K = 40
num_classes = 1
heads = {
    'hm': 1,
    'wh': 2,
    'reg': 2
}
head_conv = 64
scale = 1
down_ratio = 4
inp_height = 256; inp_width=256

detect_results = {}

id_to_ges = {0: 'heart2', 1: 'five', 2: 'eight', 3: 'victory', 4: 'heart', 5: '666', 6: 'great', 7: 'ok', 8: 'fist', 9: 'pointer', 10: 'lift'}

# detect_results = {
#         'image1': [
#             {'type': 'hand', 'bbox': [0, 0, 0.4, 0.4], 'confidence': 1},
#         ],
#         'image2': [
#             {'type': 'hand', 'bbox': [0.4, 0.4, 1, 1], 'confidence': 0.9},
#             {'type': 'hand', 'bbox': [0, 0, 0.5, 0.5], 'confidence': 0.8}
#         ]

#     }

def main_wrapper(file_list, model_path):
    model = PoseResNet(heads, head_conv=head_conv, base=base_model) 
    model_path = model_path
    model.load_state_dict(torch.load(model_path, map_location=the_device)['state_dict']) 
    model.eval()
    model = model.to(the_device)
    
    data_new = {}
    
    couant = 0
    bad_count = 0
    start_time = time.time()
    image_id_now = 0
    for image_inf in vc.progress_bar(file_list):
#         print(image_id_now)
        image_id_now += 1
        a_image_path = image_inf
        if '.DS_Store' in a_image_path:
            continue
        detect_results[a_image_path] = []
        # print(a_image_path)
        #count += 1
        a_image_name = os.path.basename(a_image_path)
#         print(a_image_path)
        try:
            a_image = Image.open(a_image_path)
        except:
            continue
        a_image = np.array(a_image)[..., ::-1].copy()
        # a_image = cv2.imread(a_image_path)
        height, width = a_image.shape[0:2]
	
        new_height = int(height * scale)
        new_width  = int(width * scale)
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}
        

        a_padded_image, start_index, flag  = pad_image(a_image)        
        a_padded_image = cv2.resize(a_padded_image, (256, 256))
        input_image = (a_padded_image/255.0 - mean) / std  
#         input_image = np.flip(input_image, axis=-1).copy()

#         ssd = SSDAugmentation(flag = 'test')
#         input_image, _, _, _ = ssd(a_image,np.random.random((1, 4)),np.array([0]), np.zeros([height, width]))
        input_image = torch.from_numpy(input_image.transpose(2, 0, 1)).unsqueeze(dim=0).float()
        input_image = input_image.to(the_device)
        output = model(input_image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
    
        dets = ctdet_decode(hm, wh, reg=reg, K=K)   
        dets = dets.detach().cpu().numpy()
        
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process_test(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], num_classes)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        #print(dets[0],'apehaha')
        dets = dets[0][1]
        #print(dets)
#         a_target_image_path = os.path.join(target_dir, a_image_name) 
#         print("Image Save Dir: ", a_target_image_path)
        #a_target_pts_path = os.path.join(target_dir, a_image_name)[:-4] + ".pts"
        #results_vecs = []
        
        for a_det in dets:
            dic_one_box = {}
            x1, y1, x2, y2, conf = a_det 
#             if conf < 0.5:
#                 continue
#             print(x1, y1, x2, y2, conf)
            x1_int = int(x1 ); y1_int = int(y1 ); x2_int = int(x2 ); y2_int = int(y2 )
            x1 /= width
            y1 /= height
            x2 /= width
            y2 /= height
            #if conf < 0.3: continue
            cv2.rectangle(a_image, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 3)
#             cv2.imwrite('{}.jpg'.format(random.random()), a_image)
            if image_inf in data_new:
                #data_new[image_inf]['instances'][0][4]
                conf_prev = data_new[image_inf]['instances'][0]['bbox'][4]
                if conf < conf_prev: # only keep the most confident one
                    continue
            annot_new = {}
            annot_new['image_shape'] = (height, width, 3)
            instances = []
            instance = {}
            instance['bbox'] = [x1, y1, x2, y2, conf]
            if '左手' in image_inf:
                instance['lr'] = 'l'
            elif '右手' in image_inf:
                instance['lr'] = 'r'
            else:
                bad_count += 1
            annot_new['type'] = 'hand'
            instances.append(instance)
            annot_new['instances'] = instances
            data_new[image_inf] = annot_new

            #a_tmp_str = str(conf) + "\t" + str(x1) + "\t" + str(y1) + "\t" + str(x2) + "\t" + str(y2) + "\n"
             #results_vecs.append(a_tmp_str) 
                
            #dic_one_box['type'] = 'hand'
            #dic_one_box['bbox'] = [x1/width, y1/height, x2/width, y2/height]
            #dic_one_box['confidence'] = conf
            #detect_results[a_image_path].append(dic_one_box)
                

#         cv2.imwrite(a_target_image_path, a_image)
        #f = open(a_target_pts_path, "w")
        #f.writelines(results_vecs)
        #f.close()
        
    #json_str = json.dumps(detect_results, cls=MyEncoder)
    #awith open('results_one_class.json', 'w') as json_file:
    #    json_file.write(json_str)
#     with open('result_by_exp022.pkl', 'wb') as f:
#         pickle.dump(data_new, f)

#     print('bad count' , bad_count)

#     end_time = time.time()
#     avg_time = (end_time - start_time) / (count+1)
#     print("avg_time:\t", avg_time)
    
    return data_new
