from torch import sum as torch_sum, mean as torch_mean, float32 as torch_float32
from torch.nn.function import log_softmax as F_log_softmax


def soft_cross_entropy_loss(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = F_log_softmax(input.view(input.shape[0], -1), dtype=torch_float32, dim=1).type_as(input)
    batchloss = - torch_sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch_mean(batchloss)
    elif reduction == 'sum':
        return torch_sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')
