data_dir=data

echo "-----Start Evaluation---------"
dataset=$1
identity=$2
gender=$3
N_view=$4
echo "identity: ${identity}"
echo "gender: ${gender}"
echo "Number of view: ${N_view}"
image_config=config/${dataset}/${identity}/${N_view}_camera_rotate.csv
workspace=out/${dataset}/${identity}/camera_rotate_tpose_${N_view}
echo "image_config: $image_config"
echo "workspace: $workspace"


CUDA_VISIBLE_DEVICES=0 python main.py \
    --image_config $image_config \
    --workspace $workspace \
    --dmtet \
    --init_with ${workspace}/checkpoints/df_ep0175.pth \
    --test \
    --pose_path ${data_dir}/${dataset}/training/Apose.pkl \
    --smplx_path models/smplx \
    --gender $gender \
    --gt_smplx ${data_dir}/${dataset}/training/${identity}/smplx/tpose.pkl \
    --various_pose \
    --tpose \
    --bg_radius 0 \
    --expression \
    --eval_supl \
    --H 1024 \
    --W 1024 

