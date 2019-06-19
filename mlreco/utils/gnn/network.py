# defines incidence matrix for primaries
import numpy as np
import torch

def primary_bipartite_incidence(batches, primaries, cuda=True):
    """
    incidence matrix of bipartite graph between primary clusters and non-primary clusters
    """
    others = np.where([ not(x in primaries) for x in np.arange(len(batches))])[0]
    ret = torch.tensor([[i, j] for i in primaries for j in others if batches[i] == batches[j]], dtype=torch.long, requires_grad=False).t().contiguous()
    if cuda:
        ret = ret.cuda()
    return ret
    