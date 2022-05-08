from torch import zeros as torch_zeros, sum as torch_sum, mean as torch_mean, full as torch_full
from torch.nn import Module, LogSoftmax


class Label_Smoothing_Regression(Module):
    def __init__(self, e=0.01, reduction='mean'):
        super().__init__()

        self.log_softmax = LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        device = labels.device
        one_hot = torch_zeros(labels.size(0), classes, device=device)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch_full((labels.size(0), 1), value, device=device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        return self._one_hot(target, length, value=1 - smooth_factor) + smooth_factor / length

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        loss = torch_sum(- self.log_softmax(x) * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch_sum(loss)
        elif self.reduction == 'mean':
            return torch_mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')
