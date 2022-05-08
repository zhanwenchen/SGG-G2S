import torch
from torch import arange as torch_arange, log as torch_log, sigmoid as torch_sigmoid
from torch.nn import Module
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from maskrcnn_benchmark._C import sigmoid_focalloss_forward, sigmoid_focalloss_backward

# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        return sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        return sigmoid_focalloss_backward(
            *(ctx.saved_tensors),
            d_loss.contiguous(),
            ctx.num_classes,
            ctx.gamma,
            ctx.alpha
        ), None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    class_range = torch_arange(1, num_classes+1, dtype=targets.dtype, device=targets.device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch_sigmoid(logits)
    term1 = (1 - p) ** gamma * torch_log(p)
    term2 = p ** gamma * torch_log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        return loss_func(logits, targets, self.gamma, self.alpha).sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        return tmpstr + ")"
