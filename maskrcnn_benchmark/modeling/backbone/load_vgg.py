from torchvision.models import vgg16


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
    model = vgg16(pretrained=pretrained)
    del model.features._modules['30']  # Get rid of the maxpool
    del model.classifier._modules['6']  # Get rid of class layer
    if not use_dropout:
        del model.classifier._modules['5']  # Get rid of dropout
        if not use_relu:
            del model.classifier._modules['4']  # Get rid of relu activation
            if not use_linear:
                del model.classifier._modules['3']  # Get rid of linear layer
    return model
