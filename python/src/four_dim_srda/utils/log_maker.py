from logging import getLogger

import torch

logger = getLogger()


def output_gpu_memory_summary_log(device_index=0):
    logger.info(torch.cuda.memory_summary(device=device_index))


def output_gpu_memory_info_log(device_index=0):
    # Get the total memory available on the GPU
    total_memory = torch.cuda.get_device_properties(device=device_index).total_memory

    # Get the currently allocated memory　(actual usage)
    allocated_memory = torch.cuda.memory_allocated(device=device_index)

    # Get the reserved memory by PyTorch (includes cache and non-allocated reserved memory)
    cached_memory = torch.cuda.memory_reserved(device=device_index)

    # Calculate the free memory available by subtracting allocated and cached memory from total memory
    free_memory = total_memory - allocated_memory - cached_memory

    logger.info(f"Total Memory: {total_memory / 1024**2:.2f} MB")
    logger.info(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
    logger.info(f"Cached Memory: {cached_memory / 1024**2:.2f} MB")
    logger.info(f"Free Memory: {free_memory / 1024**2:.2f} MB\n")
