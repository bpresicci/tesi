#voglio trovare un criterio per usare alcune epoche. Provo tagliando in base allo score
import numpy as np
def min_score(idx_best, epoch, scores, min_percentage):
    scores_mean = np.mean(scores)
    min_scores = scores_mean * min_percentage
    new_idx_best = []
    for i in range(len(idx_best)):
        if scores[i] >= min_scores:
            new_idx_best = new_idx_best + [idx_best[i]]
    new_idx_best = np.array(new_idx_best)
    if len(new_idx_best) != len(idx_best):
        epoch_list = np.zeros((len(new_idx_best), len(epoch[0]), len(epoch[0][0])))
        n_epochs = len(new_idx_best)
        for i in range(n_epochs):
            epoch_list[i] = epoch[new_idx_best[i]]
    else:
        epoch_list = epoch
    return new_idx_best, epoch_list