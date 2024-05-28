set -x

export SEQUENCE_ID="8700";
export CROP_DATA_ROOT=/cpfs/shared/public/chenxingyu/few_shot_data/real_hand/${SEQUENCE_ID}; # change to your sequence
export IMGS_CROP_ROOT=${CROP_DATA_ROOT}/images_crop;
export MASKS_CROP_ROOT=${CROP_DATA_ROOT}/mobrecon_results_crop/mask;
export JSONS_CROP_ROOT=${CROP_DATA_ROOT}/mobrecon_results_crop/json;
export SHAPES_CROP_ROOT=${CROP_DATA_ROOT}/shape_crop;
export CUDA_ID=0;

export FRONT_ID="000082"; # change to your picture, same as img name
export BACK_ID="000260";
export SAVE_PARAM_PATH=tmp_realhand_params;
export NO_WRIST_ROOT=tmp_wrist_imgs;
export SAVE_MESH_PATH=tmp_mano;
export SAVE_ALIGNED_DATA_PATH=tmp_aligned_data;

env=/cpfs/shared/public/chenxingyu/envs/pt111_mnt
# step 1: remove wrist

CUDA_VISIBLE_DEVICES=${CUDA_ID} ${env}/bin/python real_hand_codes/remove_wrist.py --src_root ${IMGS_CROP_ROOT} --mask_root ${MASKS_CROP_ROOT} --dst_root ${NO_WRIST_ROOT}

# step 2: get real hand scale
cd handmesh_utils
CUDA_VISIBLE_DEVICES=${CUDA_ID} ${env}/bin/python ikhand/get_scale.py \
    --shape_root ${SHAPES_CROP_ROOT} --front_id ${FRONT_ID} --back_id ${BACK_ID} --dst_path ${SAVE_PARAM_PATH}
cd ..

# step 3: get all needed parameters
CUDA_VISIBLE_DEVICES=${CUDA_ID} ${env}/bin/python real_hand_codes/get_mano_both_forward.py --shape_root ${SHAPES_CROP_ROOT} --front_id ${FRONT_ID} --back_id ${BACK_ID} --dst_path ${SAVE_PARAM_PATH}

# step 4: get init mesh using real shape
CUDA_VISIBLE_DEVICES=${CUDA_ID} ${env}/bin/python real_hand_codes/sample_mano.py \
    -n 1 --dst_path ${SAVE_MESH_PATH} --scale 0 --real_hand_scale ${SAVE_PARAM_PATH}/real_scale_${FRONT_ID}.npy --font_id ${FRONT_ID} --shape_root ${SHAPES_CROP_ROOT}

# step 5: align imgs
cd handmesh_utils
CUDA_VISIBLE_DEVICES=${CUDA_ID} ${env}/bin/python dethand/crop_aligned_data.py \
    --png_dir ${NO_WRIST_ROOT} --json_dir ${JSONS_CROP_ROOT} --front_id ${FRONT_ID} --back_id ${BACK_ID} \
    --xyz_rel_root ../${SAVE_PARAM_PATH} --no_wrist_imgs_root ../${NO_WRIST_ROOT} --save_path ../${SAVE_ALIGNED_DATA_PATH} 
cd ..

# step 6: remove bkgd
CUDA_VISIBLE_DEVICES=${CUDA_ID} ${env}/bin/python real_hand_codes/bkgd_remove.py \
    --imgs_root ${SAVE_ALIGNED_DATA_PATH}

# step 7: put everything together just like /cpfs/shared/public/chenxingyu/few_shot_data/real_hand/2view2pose_082_260 
