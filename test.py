# gpu_mem_filler.py
import argparse, time, signal, sys
import torch

DTYPE_MAP = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def bytes_str(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def allocate_on_device(dev: int, target_alloc_bytes: int, chunk_bytes: int, dtype: torch.dtype, quiet=False):
    tensors = []
    allocated = 0
    elem_size = torch.tensor([], dtype=dtype).element_size()
    # 先建立上下文，避免第一次分配带来的开销误判
    with torch.cuda.device(dev):
        _ = torch.empty(1, device=dev)
        torch.cuda.synchronize(dev)

    while allocated < target_alloc_bytes:
        try:
            want = min(chunk_bytes, target_alloc_bytes - allocated)
            n_elems = max(1, want // elem_size)
            with torch.cuda.device(dev):
                t = torch.empty(n_elems, dtype=dtype, device=dev)
            tensors.append(t)
            allocated += n_elems * elem_size
            if not quiet:
                free_b, total_b = torch.cuda.mem_get_info(dev)
                print(f"[GPU {dev}] allocated={bytes_str(allocated)} | free={bytes_str(free_b)} / total={bytes_str(total_b)}")
        except torch.cuda.OutOfMemoryError:
            # 分块太大就二分下降，直到 1MB
            chunk_bytes //= 2
            if chunk_bytes < 1<<20:  # 1 MB
                if not quiet:
                    print(f"[GPU {dev}] Reached near OOM; stop at {bytes_str(allocated)}")
                break
        except RuntimeError as e:
            # 兼容旧版 PyTorch 的 OOM 抛错
            if "out of memory" in str(e).lower():
                chunk_bytes //= 2
                if chunk_bytes < 1<<20:
                    break
            else:
                raise
    return tensors, allocated

def main():
    p = argparse.ArgumentParser(description="Fill GPU memory for testing.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", type=int, help="只占用这张 GPU（编号）")
    g.add_argument("--all-gpus", action="store_true", help="占用所有可见 GPU（默认）")
    t = p.add_mutually_exclusive_group()
    t.add_argument("--fraction", type=float, help="按总显存比例占用（0~1），例如 0.98")
    t.add_argument("--leave-mb", type=int, help="保留这么多 MB 空闲（默认 512MB）")
    t.add_argument("--target-mb", type=int, help="目标占用大小（MB），按实际可用显存截断")
    p.add_argument("--chunk-mb", type=int, default=256, help="每次分配的分块大小（MB），默认 256")
    p.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="float32", help="分配张量的数据类型（影响分配速度/粒度）")
    p.add_argument("--hold-seconds", type=float, default=-1, help="持有时长；<0 表示一直持有，Ctrl+C 结束")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用。请在有 GPU 的环境运行。")
        sys.exit(1)

    if args.gpu is None and not args.all-gpus:
        args.all_gpus = True

    dev_count = torch.cuda.device_count()
    devices = list(range(dev_count)) if args.all_gpus else [args.gpu]
    for d in devices:
        if d is None or d < 0 or d >= dev_count:
            print(f"GPU 编号无效：{d}")
            sys.exit(1)

    dtype = DTYPE_MAP[args.dtype]
    chunk_bytes = args.chunk_mb * (1<<20)

    holders = {}  # dev -> list[tensors]
    allocated_tot = 0

    # 释放函数
    def release(*_):
        print("\n释放显存…")
        for d, lst in holders.items():
            lst.clear()
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        time.sleep(0.5)
        for d in devices:
            free_b, total_b = torch.cuda.mem_get_info(d)
            print(f"[GPU {d}] free={bytes_str(free_b)} / total={bytes_str(total_b)}")
        sys.exit(0)

    signal.signal(signal.SIGINT, release)
    signal.signal(signal.SIGTERM, release)

    for d in devices:
        free_b, total_b = torch.cuda.mem_get_info(d)

        # 计算目标占用
        if args.target_mb is not None:
            target = min(args.target_mb * (1<<20), free_b - (16<<20))  # 至少留 16MB
        elif args.fraction is not None:
            target = int(total_b * args.fraction)
            target = min(target, free_b - (16<<20))
        else:
            leave = (args.leave_mb if args.leave_mb is not None else 512) * (1<<20)
            target = max(0, free_b - leave)

        if target <= 0:
            print(f"[GPU {d}] 可用显存不足（free={bytes_str(free_b)}），跳过。")
            holders[d] = []
            continue

        if not args.quiet:
            print(f"[GPU {d}] free={bytes_str(free_b)} / total={bytes_str(total_b)} -> target_alloc={bytes_str(target)}")
        tensors, got = allocate_on_device(d, target, chunk_bytes, dtype, args.quiet)
        holders[d] = tensors
        allocated_tot += got
        if not args.quiet:
            print(f"[GPU {d}] done, allocated {bytes_str(got)}")

    if allocated_tot == 0:
        print("没有分配到任何显存。可能显存已被占满或权限受限。")
        sys.exit(0)

    if args.hold_seconds is not None and args.hold_seconds >= 0:
        if not args.quiet:
            print(f"保持 {args.hold_seconds}s 后释放…  (Ctrl+C 可提前释放)")
        time.sleep(args.hold_seconds)
        release()
    else:
        if not args.quiet:
            print("显存已占用，按 Ctrl+C 释放。")
        signal.pause()  # 等待信号

if __name__ == "__main__":
    main()
