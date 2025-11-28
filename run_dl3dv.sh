#!/bin/bash

# 运行 simple_trainer_deblur.py 的独立脚本 (GPU 0)
# 专门用于运行 simple_trainer_deblur.py 在所有场景上

# 设置基础路径
BASE_DIR=""
DATA_DIR=""
RESULTS_DIR=""

# 脚本路径
SCRIPT_TRAINER="$BASE_DIR/simple_deblur_difix.py"

# 固定使用GPU 0
export CUDA_VISIBLE_DEVICES=3

# 创建结果目录
mkdir -p "$RESULTS_DIR"


declare -a DL3DV_SCENES=(
    "0001"
    "0002"
    "0003"
    "0004"
    "0005"
)

# 颜色输出函数
print_header() {
    echo -e "\n\033[1;34m========================================\033[0m"
    echo -e "\033[1;34m$1\033[0m"
    echo -e "\033[1;34m========================================\033[0m\n"
}

print_info() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# 运行单个实验的函数
run_trainer_experiment() {
    local scene_name=$1
    local data_path=$2
    
    local result_dir="$RESULTS_DIR/3views200/${scene_name}"
    
    
    # 创建结果目录
    mkdir -p "$result_dir"
    #
    # 运行训练
    cd "$BASE_DIR"
    DATA_FACTOR=4
    SCALE_FACTOR=0.25
    TRAJ_TYPE="bezier"
     #--virtual-view-st--virtual-view-start-step 2000 \
     #   --virtual-view-interval 250 art-step 1500 \
      #  --virtual-view-interval 200 \
      #--max-steps 7000 \
    python "$SCRIPT_TRAINER" default \
        --data-dir "$data_path" \
        --result-dir "$result_dir" \
        --disable-viewer \
        --virtual-view-start-step 1500 \
        --virtual-view-interval 200 \
        --max-steps 7000 \
        --camera-optimizer.mode $TRAJ_TYPE \
        --train-indices 5 15 25 \
        2>&1 | tee "${result_dir}_training.log"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_info "✅ 实验完成: $scene_name 使用 simple_trainer_deblur.py"
    else
        print_error "❌ 实验失败: $scene_name 使用 simple_trainer_deblur.py (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 主实验循环
main() {
    print_header "开始所有场景的 simple_trainer_deblur.py 实验 (GPU 0)"
    
    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    
    # 处理 bad-nerf-gtK-colmap-nvs 场景
    print_header "处理 bad-nerf-gtK-colmap-nvs 场景"

    
    for scene in "${DL3DV_SCENES[@]}"; do
        local data_path="$DATA_DIR/$scene/3views"
        
        if [ ! -d "$data_path" ]; then
            print_warning "跳过不存在的场景: $data_path"
            continue
        fi
        
        total_experiments=$((total_experiments + 1))
        
        if run_trainer_experiment "$scene" "$data_path"; then
            successful_experiments=$((successful_experiments + 1))
        else
            failed_experiments=$((failed_experiments + 1))
        fi
        
        # 等待一段时间让GPU释放内存
        print_info "⏸️  等待GPU内存释放..."
        sleep 10
    done
    
    # 输出最终统计
    print_header "simple_trainer_deblur.py 实验完成统计"
    print_info "总实验数: $total_experiments"
    print_info "成功实验数: $successful_experiments"
    print_info "失败实验数: $failed_experiments"
    
    if [ $total_experiments -gt 0 ]; then
        local success_rate=$(echo "scale=2; $successful_experiments * 100 / $total_experiments" | bc 2>/dev/null || echo "N/A")
        print_info "成功率: ${success_rate}%"
    else
        print_info "成功率: 0%"
    fi
    
    # 生成实验报告
    local report_file="$RESULTS_DIR/trainer_experiment_report_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "simple_trainer_deblur.py 实验报告 - $(date)"
        echo "============================================="
        echo "GPU: 0"
        echo "脚本: simple_trainer_deblur.py"
        echo "总实验数: $total_experiments"
        echo "成功实验数: $successful_experiments"
        echo "失败实验数: $failed_experiments"
        if [ $total_experiments -gt 0 ]; then
            echo "成功率: $(echo "scale=2; $successful_experiments * 100 / $total_experiments" | bc 2>/dev/null || echo "N/A")%"
        else
            echo "成功率: 0%"
        fi
        echo ""
        echo "实验场景列表:"
        echo "-------------"
        
        for scene in "${BAD_NERF_SCENES[@]}" "${REAL_BLUR_SCENES[@]}" "${DL3DV_SCENES[@]}"; do
            echo "- $scene"
        done
        
    } > "$report_file"
    
    print_info "实验报告已保存到: $report_file"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查脚本是否存在
    if [ ! -f "$SCRIPT_TRAINER" ]; then
        print_error "脚本不存在: $SCRIPT_TRAINER"
        exit 1
    fi
    
    # 检查数据目录是否存在
    if [ ! -d "$DATA_DIR" ]; then
        print_error "数据目录不存在: $DATA_DIR"
        exit 1
    fi
    
    # 检查GPU是否可用
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi 不可用，无法检查GPU状态"
    else
        print_info "GPU 0 状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -1
    fi
    
    print_info "依赖检查完成"
}

# 脚本入口
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h     显示此帮助信息"
    echo "  --check        仅检查依赖，不运行实验"
    echo ""
    echo "说明:"
    echo "  此脚本专门运行 simple_trainer_deblur.py 在所有场景上 (使用GPU 0)"
    echo "  总共会运行 $(( ${#BAD_NERF_SCENES[@]} + ${#REAL_BLUR_SCENES[@]} + ${#DL3DV_SCENES[@]} )) 个实验"
    exit 0
fi

if [ "$1" = "--check" ]; then
    check_dependencies
    exit 0
fi

# 运行主程序
check_dependencies
main
