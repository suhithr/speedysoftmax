import torch
import os
import logging
logger = logging.getLogger(__name__)


def benchmark_torch_softmax(M: int = 1024, N: int = 32768):
    os.makedirs("benchmarks", exist_ok=True)
    logging.basicConfig(filename='benchmarks/exec_torch.og', level=logging.INFO)
    num_iters = 5

    matrix = torch.randn(M, N, device='cuda', dtype=torch.float32)
    logger.info(f'Running Softmax on matrix of size {M, N}.')
    print(f'Running Softmax on matrix of size {M, N}.')

    # Warm up
    for _ in range(5):
        __ = torch.nn.functional.softmax(matrix, dim=-1)
    torch.cuda.synchronize()

    total_time = 0
    for _ in range(num_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        __ = torch.nn.functional.softmax(matrix, dim=-1)
        end_event.record()

        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
    
    logger.info(f'Execution time: {total_time:.4f} ms')
    print(f'Execution time: {total_time:.4f} ms')

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

if __name__ == '__main__':
    benchmark_torch_softmax()