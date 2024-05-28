export identity=$1
export N_view=$2
export lambda_sds_scale=$3
formatted_number=$(printf "%03d" $identity)

echo "identity: ${identity}"
echo "Number of view: ${N_view}"
export CONFIG=config/FS-DART/${N_view}view/${formatted_number}.csv
echo "image_config: config/FS-DART/${N_view}view/${formatted_number}.csv"

if [ $N_view = "2" -o $N_view = "8" ]
then
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --image_config config/FS-DART/${N_view}view/$formatted_number.csv \
        --workspace out/dart_exp/${N_view}view/$formatted_number \
        --dmtet \
        --iters 3000 \
        --lambda_rgb 500 \
        --lambda_normal 0 \
        --lambda_depth 0 \
        --lambda_mask 1000 \
        --init_with data/FS-DART/training/$identity/779points_scaled_$identity.obj \
        --save_mesh \
        --test_interval 5 \
        --bg_radius 0 \
        --various_pose \
        --hand_tpose \
        --mano_path models/mano/MANO_RIGHT.pkl \
        --lambda_sds ${lambda_sds_scale} \
        --progressive_view \
        --lap_vn \
        --real_hand_pose \
        --lr_iters 5000 \
        --lambda_mesh_laplacian 1 \
        --const_ambient_ratio 1.0

    CUDA_VISIBLE_DEVICES=0 python main.py \
        --image_config config/FS-DART/${N_view}view/$formatted_number.csv \
        --workspace out/dart_exp/${N_view}view/$formatted_number \
        --dmtet \
        --iters 3000 \
        --lambda_rgb 500 \
        --lambda_normal 0 \
        --lambda_depth 0 \
        --lambda_mask 1000 \
        --init_with data/FS-DART/training/$identity/779points_scaled_$identity.obj \
        --test_interval 5 \
        --bg_radius 0 \
        --various_pose \
        --hand_tpose \
        --mano_path models/mano/MANO_RIGHT.pkl \
        --lambda_sds ${lambda_sds_scale} \
        --progressive_view \
        --lap_vn \
        --real_hand_pose \
        --lr_iters 5000 \
        --lambda_mesh_laplacian 1 \
        --const_ambient_ratio 1.0 \
        --eval_metrics \
        --H 512 \
        --W 512 
fi

if [ $N_view = "4" ]
then
    export ANGLE=+-45 ## change here
    echo "USING ANGLE ${ANGLE} !!!!!!!!!!!!!!!!!!!!!!!!!!"
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --image_config config/FS-DART/${N_view}view/${ANGLE}/$formatted_number.csv \
        --workspace out/dart_exp/${N_view}view${ANGLE}/$formatted_number \
        --dmtet \
        --iters 3000 \
        --lambda_rgb 500 \
        --lambda_normal 0 \
        --lambda_depth 0 \
        --lambda_mask 1000 \
        --init_with data/FS-DART/training/$identity/779points_scaled_$identity.obj \
        --save_mesh \
        --test_interval 5 \
        --bg_radius 0 \
        --various_pose \
        --hand_tpose \
        --mano_path models/mano/MANO_RIGHT.pkl \
        --lambda_sds ${lambda_sds_scale} \
        --progressive_view \
        --lap_vn \
        --real_hand_pose \
        --lr_iters 5000 \
        --lambda_mesh_laplacian 1 \
        --const_ambient_ratio 1.0

    CUDA_VISIBLE_DEVICES=0 ${env}/bin/python main.py \
        --image_config config/FS-DART/${N_view}view/${ANGLE}/$formatted_number.csv \
        --workspace out/dart_exp/${N_view}view${ANGLE}/$formatted_number \
        --dmtet \
        --iters 3000 \
        --lambda_rgb 500 \
        --lambda_normal 0 \
        --lambda_depth 0 \
        --lambda_mask 1000 \
        --init_with data/FS-DART/training/$identity/779points_scaled_$identity.obj \
        --test_interval 5 \
        --bg_radius 0 \
        --various_pose \
        --hand_tpose \
        --mano_path models/mano/MANO_RIGHT.pkl \
        --lambda_sds ${lambda_sds_scale} \
        --progressive_view \
        --lap_vn \
        --real_hand_pose \
        --lr_iters 5000 \
        --lambda_mesh_laplacian 1 \
        --const_ambient_ratio 1.0 \
        --eval_metrics \
        --H 512 \
        --W 512 

fi