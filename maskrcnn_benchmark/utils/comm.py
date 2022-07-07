"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

from pickle import dumps as pickle_dumps, loads as pickle_loads
from packaging.version import parse as version_parse
from torch import (
    stack as torch_stack,
    cat as torch_cat,
    no_grad as torch_no_grad,
    ByteStorage as torch_ByteStorage,
    __version__ as torch___version__,
)
from torch.cuda import (
    ByteTensor as torch_cuda_ByteTensor,
    LongTensor as torch_cuda_LongTensor,
    ByteStorage as torch_cuda_ByteStorage,
)
from torch.distributed.dist import (
    is_available as dist_is_available,
    is_initialized as dist_is_initialized,
    get_world_size as dist_get_world_size,
    get_rank as dist_get_rank,
    reduce as dist_reduce,
    barrier as dist_barrier,
    all_gather as dist_all_gather,
)


if version_parse(torch___version__) <= version_parse('1.10'):
    from_buffer = torch_ByteStorage.from_buffer
else:
    from_buffer = torch_cuda_ByteStorage.from_buffer


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
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    to_device = "cuda"
    #to_device = torch.device("cpu")

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle_dumps(data)
    del data
    storage = from_buffer(buffer)
    del buffer
    tensor = torch_cuda_ByteTensor(storage, device=to_device)
    del storage
    # obtain Tensor size of each rank
    local_size = torch_cuda_LongTensor([tensor.numel()], device=to_device)
    size_list = [torch_cuda_LongTensor([0], device=to_device) for _ in range(world_size)]
    dist_all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch_cuda_ByteTensor(size=(max_size,), device=to_device))
    if local_size != max_size:
        padding = torch_cuda_ByteTensor(size=(max_size - local_size,), device=to_device)
        tensor = torch_cat((tensor, padding), dim=0)
    dist_all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle_loads(buffer))

    return data_list


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
