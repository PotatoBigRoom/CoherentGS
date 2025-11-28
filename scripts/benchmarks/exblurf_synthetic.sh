SCENE_DIR="$HOME/data/bad-gaussian/data/Deblur-GS/synthetic_release"
RESULT_DIR="results/benchmark_mcmc_500k/exblurf_synthetic_linear"
SCENE_LIST="blurcozy2room blurfactory blurpool blurtanabata blurwine"

CAP_MAX=500000

for SCENE in $SCENE_LIST;
do
    GPUS="0"
    DATA_FACTOR=1
    SCALE_FACTOR=0.25
    TRAJ_TYPE="bezier"

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=$GPUS python simple_trainer_deblur.py mcmc \
        --eval_steps -1 \
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --scale_factor $SCALE_FACTOR \
        --camera-optimizer.mode $TRAJ_TYPE \
        --strategy.cap-max $CAP_MAX \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=$GPUS python simple_trainer_deblur.py mcmc \
            --disable_viewer \
            --data_factor $DATA_FACTOR \
            --camera-optimizer.mode $TRAJ_TYPE \
            --deblur_eval_enable_pose_opt \
            --strategy.cap-max $CAP_MAX \
            --scale_factor $SCALE_FACTOR \
            --data_dir $SCENE_DIR/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in $SCENE_LIST;
do
    echo "=== Deblur Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/deblur*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== NVS Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/nvs*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done