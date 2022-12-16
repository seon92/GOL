from __future__ import print_function

from networks.gol import GOL


def prepare_model(opt):
    model = eval(opt.model)(opt)
    return model




