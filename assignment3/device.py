import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # TODO: mps is slower than cpu....
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")
