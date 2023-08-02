import torch


def mask_by_noise(mask_len, probs, temperature=1.0):
    confidence = torch.log(probs) + temperature * \
       torch.distributions.gumbel.Gumbel(torch.tensor([1.0]), torch.tensor([2.0])).sample(probs.shape)
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    cut_off = torch.gather(sorted_confidence, dim=-1, index=mask_len)
    masking = (confidence < cut_off)
    return masking
