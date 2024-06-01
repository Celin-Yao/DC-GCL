import torch


def scatter(logits, index, k, device):
    bs = logits.shape[0]
    x_index = torch.arange(bs).reshape(-1, 1).expand(bs, k)
    x_index = x_index.reshape(-1).tolist()
    y_index = index.reshape(-1).tolist()
    output = torch.zeros_like(logits).to(device)
    output[x_index, y_index] = 1.0
    return output


def gumbel_softmax(logits, rate, device, tau=1, hard=True, dim=-1):
    def _gen_gumbels():
        gumbel = -torch.empty_like(logits).exponential_().log().to(device)
        if torch.isnan(gumbel).sum() or torch.isinf(gumbel).sum():
            gumbel = _gen_gumbels()
        return gumbel

    k = int(logits.size(dim=1) * rate)
    gumbels = _gen_gumbels()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = scatter(logits, index, k, device)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
