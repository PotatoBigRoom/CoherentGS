#!/usr/bin/env bash
# Simple runner for CoherentGS simple_deblur_difix.py
#
# 用法：
#   1) 仅打印示例命令：
#        bash DeblurDIFIXZK/CoherentGS/run.sh
#   2) 单卡直接运行（使用下方变量）：
#        bash DeblurDIFIXZK/CoherentGS/run.sh single
#   3) 多卡直接运行（使用下方变量）：
#        bash DeblurDIFIXZK/CoherentGS/run.sh multi
#   4) 仅评估（需设置 CKPT）：
#        CKPT=/path/to/ckpt.pt bash DeblurDIFIXZK/CoherentGS/run.sh eval
#
# 通过环境变量自定义：
#   DATA_DIR=/path/to/data
#   RESULT_DIR=/path/to/output
#   TRAIN_INDICES="5 15 25"   # 空格分隔；也可写成 [5,15,25]
#   GPUS="0,1,2,3"            # 多卡示例

set -euo pipefail

ROOT_DIR="/remote-home/fcr/Event_proj"
cd "$ROOT_DIR"

# 默认变量，可按需覆盖
DATA_DIR=${DATA_DIR:-"/remote-home/fcr/Event_proj/DeblurDIFIXZK/BAD-Gaussians-gsplat-only_vgg3/data/bad-nerf-gtK-colmap-nvs/blurpool"}
RESULT_DIR=${RESULT_DIR:-"/remote-home/fcr/Event_proj/DeblurDIFIXZK/results"}
TRAIN_INDICES=${TRAIN_INDICES:-"5 15 25"}
GPUS=${GPUS:-"0,1,2,3"}

print_examples() {
  echo ""
  echo "[示例] 单卡训练："
  echo "  python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default \"--data_dir\" \"$DATA_DIR\" \"--result_dir\" \"$RESULT_DIR\" \"--train_indices\" $TRAIN_INDICES"
  echo ""
  echo "[示例] 指定训练ID（列表形式）："
  echo "  python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default --data_dir $DATA_DIR --result_dir $RESULT_DIR --train_indices [5,15,25]"
  echo ""
  echo "[示例] 多卡训练（4卡，步数缩放）："
  echo "  CUDA_VISIBLE_DEVICES=$GPUS python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default --steps_scaler 0.25 --data_dir $DATA_DIR --result_dir $RESULT_DIR --train_indices $TRAIN_INDICES"
  echo ""
  echo "[示例] 仅评估（需设定 CKPT）："
  echo "  python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default --eval_only True --ckpt /path/to/ckpt.pt --data_dir $DATA_DIR --result_dir $RESULT_DIR"
  echo ""
}

run_single() {
  python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --train_indices $TRAIN_INDICES
}

run_multi() {
  CUDA_VISIBLE_DEVICES=$GPUS python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default \
    --steps_scaler 0.25 \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --train_indices $TRAIN_INDICES
}

run_eval() {
  if [[ -z "${CKPT:-}" ]]; then
    echo "[错误] 评估模式需要设置 CKPT 环境变量，例如："
    echo "  CKPT=/path/to/ckpt.pt bash DeblurDIFIXZK/CoherentGS/run.sh eval"
    exit 1
  fi
  python DeblurDIFIXZK/CoherentGS/simple_deblur_difix.py default \
    --eval_only True \
    --ckpt "$CKPT" \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR"
}

MODE=${1:-print}
case "$MODE" in
  print)
    print_examples ;;
  single)
    run_single ;;
  multi)
    run_multi ;;
  eval)
    run_eval ;;
  *)
    echo "未知模式: $MODE"
    print_examples
    exit 1 ;;
esac

