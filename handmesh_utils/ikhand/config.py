# The unit is always decimeter

# on server
DEVICE = 'cuda'
MANO_PATH = '/home/zhouyuxiao/workspace/ikhand/code/yzhou-ikhand/ikhand/mano/mano_v1_2/models/MANO_LEFT.pkl'
MANO_VERTS = 778

# on macbook
# DEVICE = 'cpu'
# MANO_PATH = 'mano/mano_v1_2/models/MANO_LEFT.pkl'

MANO_SCALE = 10 # from meter to decimeter

CKPT_DIR = '/home/zhouyuxiao/workspace/ikhand/ckpt'
LOG_DIR = '/home/zhouyuxiao/workspace/ikhand/log'

LATEST_MODEL_INFO_FILE = 'latest_checkpoint.json'

CAM_F = 1493
CAM_C = 512

LOG_EVERY = 10
SAVE_EVERY = 100

# yajiao's results on xingyu's data
TEST_DATA_1_DIR = 'input/yajiao_xyz'

SELECTED_FRAMES = [
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_1170/440.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_1170/938.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_1170/322.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_1170/910.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_1170/734.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4041/413.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4041/917.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4041/978.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4041/639.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4041/465.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4042/422.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4042/808.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4043/368.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_4043/712.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5108/1337.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5108/756.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5163/68.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5163/382.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5361/811.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5361/562.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5362/16.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5362/289.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5481/900.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5482/1001.jpg',
  '/home/zhouyuxiao/3rdparty/chengxingyu/hand_test_data/video/wrist_test/frames/IMG_5482/778.jpg',
]