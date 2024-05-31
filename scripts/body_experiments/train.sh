data_dir=data

echo "-----Start Training---------"
dataset=$1
identity=$2
gender=$3
N_view=$4
sds=$5
echo "dataset: ${dataset}"
echo "identity: ${identity}"
echo "gender: ${gender}"
echo "Number of view: ${N_view}"
echo "lambda_guidance: ${sds}"

image_config=config/${dataset}/${identity}/${N_view}_camera_rotate.csv
workspace=out/${dataset}/${identity}/camera_rotate_tpose_${N_view}
echo "image_config: $image_config"
echo "workspace: $workspace"

iters=17500
echo "Maximum iters: ${iters}"

python main.py \
    --image_config $image_config \
    --workspace $workspace \
    --dmtet \
    --iters ${iters} \
    --lr_iters 2500 \
    --lambda_rgb 100 \
    --init_with ${data_dir}/${dataset}/training/${identity}/smplx/tpose.obj \
    --save_mesh \
    --head_recon 2 \
    --lambda_normal 50 \
    --lambda_depth 50 \
    --test_interval 25 \
    --lambda_mask 1000 \
    --hand_recon 2 \
    --bg_radius 0 \
    --smplx_path models/smplx \
    --gender $gender \
    --gt_smplx ${data_dir}/${dataset}/training/${identity}/smplx/tpose.pkl \
    --various_pose \
    --tpose \
    --expression \
    --lambda_guidance ${sds} \
    --const_ambient_ratio 1.0 