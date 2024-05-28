data_dir=data

echo "-----Start Training---------"
dataset=$1
identity=$2
gender=$3
N_view=$4
target=$5
take=$6
size=$7
view_angle=$8
echo "identity: ${identity}"
echo "gender: ${gender}"
echo "Number of view: ${N_view}"

image_config=config/${dataset}/${identity}/${N_view}_camera_rotate.csv
workspace=out/${dataset}/${identity}/camera_rotate_tpose_${N_view}
echo "image_config: $image_config"
echo "workspace: $workspace"


CUDA_VISIBLE_DEVICES=0 python main.py \
    --image_config ${image_config} \
    --workspace ${workspace} \
    --dmtet \
    --init_with ${workspace}/checkpoints/df_ep0175.pth \
    --test \
    --pose_path ${data_dir}/${dataset}/driving/${target}/Take${take}_${size}.npy \
    --smplx_path models/smplx \
    --gender $gender \
    --bg_radius 0 \
    --dataset_size_test $size \
    --gt_smplx ${data_dir}/${dataset}/training/${identity}/smplx/tpose.pkl \
    --various_pose \
    --tpose \
    --view_angle $view_angle
    # --expression