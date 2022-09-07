"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
from torch import (
    stack as torch_stack,
    no_grad as torch_no_grad,
    __version__ as torch___version__,
)
from torch.distributed import (
    is_available as dist_is_available,
    is_initialized as dist_is_initialized,
    get_world_size as dist_get_world_size,
    get_rank as dist_get_rank,
    reduce as dist_reduce,
    barrier as dist_barrier,
    all_gather_object as dist_all_gather_object,
)


def get_world_size():
    if not dist_is_available():
        return 1
    if not dist_is_initialized():
        return 1
    return dist_get_world_size()


def get_rank():
    if not dist_is_available():
        return 0
    if not dist_is_initialized():
        return 0
    return dist_get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist_is_available():
        return
    if not dist_is_initialized():
        return
    world_size = dist_get_world_size()
    if world_size == 1:
        return
    dist_barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist_all_gather_object(output, data)
    return output


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch_no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch_stack(values, dim=0)
        dist_reduce(values, dst=0)
        if dist_get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        return dict(zip(names, values))
