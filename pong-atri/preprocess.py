import torch


def preprocess(observation):
    
    out = torch.from_numpy(observation).to(torch.float32)
    out = out[35:195, :, :]
    out = out[::2, ::2, 0]
    out[(out == 144.0) | (out == 109.0)] = 0.0
    out[out != 0] = 255.0
    return out / 255.0



def preprocess_batch(observations_batch):
    observations_batch = torch.from_numpy(observations_batch).to(torch.float32) / 255.0

    observations_batch = observations_batch[:, 35:195, :, :]

    observations_batch = observations_batch[:, ::2, ::2, 0]

    max_val = observations_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    observations_batch[observations_batch == max_val] = 0.0
    observations_batch[observations_batch > 0] = 1.0

    return observations_batch