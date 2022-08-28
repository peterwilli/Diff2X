import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    total_params = sum(
        param.numel() for param in m.netG.parameters()
    )
    print('Model [{:s}] is created with {} params'.format(m.__class__.__name__, total_params))
    return m
