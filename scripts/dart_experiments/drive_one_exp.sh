export identity=$1
export view=$2

formatted_number=$(printf "%03d" $identity)
python main.py \
    --image_config config/FS-DART/$view/$formatted_number.csv \
    --workspace out/dart_exp/$view/$formatted_number \
    --dmtet \
    --iters 3000 \
    --lambda_rgb 500 \
    --lambda_normal 0 \
    --lambda_depth 0 \
    --lambda_mask 1000 \
    --init_with data/FS-DART/init/$identity/779points_scaled_$identity.obj \
    --test_interval 5 \
    --bg_radius 0 \
    --various_pose \
    --mano_path models/mano/MANO_RIGHT.pkl \
    --lambda_sds 0.01 \
    --progressive_view \
    --lap_vn \
    --real_hand_pose \
    --lr_iters 5000 \
    --lambda_mesh_laplacian 1 \
    --const_ambient_ratio 1.0 \
    --H 512 \
    --W 512 \
    --test \
    --dataset_size_test 1000 \
    --pose_path data/FS-DART/driving/interhand_poseseq_train.npy \
    --hand_dst_shape data/FS-DART/init/$identity/shape_$identity.npy

