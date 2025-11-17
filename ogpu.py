import torch
import time
import threading

def gpu_worker(gpu_id, size=40960, sleep_time=0.01):
    """在指定 GPU 上持续执行矩阵运算以保持一定的利用率"""
    torch.cuda.set_device(gpu_id)
    a = torch.randn((size, size), device=f"cuda:{gpu_id}")
    b = torch.randn((size, size), device=f"cuda:{gpu_id}")
    while True:
        c = torch.mm(a, b)
        # 防止显存过高，可在每次循环后清理
        del c
        torch.cuda.synchronize()
        time.sleep(sleep_time)  # 控制利用率（降低此值 -> 提高利用率）

def main():
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Starting workers...")
    threads = []
    for gpu_id in range(num_gpus):
        t = threading.Thread(target=gpu_worker, args=(gpu_id,))
        t.daemon = True
        t.start()
        threads.append(t)

    # 主线程保持运行
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
