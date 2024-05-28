import shutil
import os
import os.path as osp
import csv


# dstroot2view = "config/cam_rot_2view_tpose/2view"
# dstroot4view = "config/cam_rot_2view_tpose/4view"
dstroot4view = "config/cam_rot_2view_tpose/4view/+-45"
# dstroot8view = "config/cam_rot_2view_tpose/8view"
for idx in range(99):
    # dst_config2view = osp.join(dstroot2view, f"{idx:03d}.csv")
    # csv_data2view = [
    #     [1, 3.2, 90, 0, f"data/dart/training/{idx:03d}/basecolor/2.png"],
    #     [1, 3.2, 90, 180, f"data/dart/training/{idx:03d}/basecolor/6.png"]
    # ]
    
    # with open(dst_config2view, 'w', newline='') as file:
    #     writer = csv.writer(file)
        
    #     writer.writerow(['zero123_weight', 'radius', 'polar', 'azimuth', 'image'])
        
    #     writer.writerows(csv_data2view)
    
    dst_config4view = osp.join(dstroot4view, f"{idx:03d}.csv")
    # csv_data4view = [
    #     [1, 3.2, 90, -90, f"/cpfs/shared/public/chenxingyu/few_shot_data/dart/training/{idx:03d}/basecolor/0.png"],
    #     [1, 3.2, 90, 0, f"/cpfs/shared/public/chenxingyu/few_shot_data/dart/training/{idx:03d}/basecolor/2.png"],
    #     [1, 3.2, 90, 90, f"/cpfs/shared/public/chenxingyu/few_shot_data/dart/training/{idx:03d}/basecolor/4.png"],
    #     [1, 3.2, 90, 180, f"/cpfs/shared/public/chenxingyu/few_shot_data/dart/training/{idx:03d}/basecolor/6.png"]
    # ]
    csv_data4view = [
        [1, 3.2, 90, -45, f"data/dart/training/{idx:03d}/basecolor/1.png"],
        [1, 3.2, 90, 45, f"data/dart/training/{idx:03d}/basecolor/3.png"],
        [1, 3.2, 90, 135, f"data/dart/training/{idx:03d}/basecolor/5.png"],
        [1, 3.2, 90, 225, f"data/dart/training/{idx:03d}/basecolor/7.png"]
    ]
    
    with open(dst_config4view, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['zero123_weight', 'radius', 'polar', 'azimuth', 'image'])
        
        writer.writerows(csv_data4view)
    
    # dst_config8view = osp.join(dstroot8view, f"{idx:03d}.csv")
    # csv_data8view = [
    #     [1, 3.2, 90, -90, f"data/dart/training/{idx:03d}/basecolor/0.png"],
    #     [1, 3.2, 90, -45, f"data/dart/training/{idx:03d}/basecolor/1.png"],
    #     [1, 3.2, 90, 0, f"data/dart/training/{idx:03d}/basecolor/2.png"],
    #     [1, 3.2, 90, 45, f"data/dart/training/{idx:03d}/basecolor/3.png"],
    #     [1, 3.2, 90, 90, f"data/dart/training/{idx:03d}/basecolor/4.png"],
    #     [1, 3.2, 90, 135, f"data/dart/training/{idx:03d}/basecolor/5.png"],
    #     [1, 3.2, 90, 180, f"data/dart/training/{idx:03d}/basecolor/6.png"],
    #     [1, 3.2, 90, 225, f"data/dart/training/{idx:03d}/basecolor/7.png"]
    # ]
    
    # with open(dst_config8view, 'w', newline='') as file:
    #     writer = csv.writer(file)
        
    #     writer.writerow(['zero123_weight', 'radius', 'polar', 'azimuth', 'image'])
        
    #     writer.writerows(csv_data8view)
