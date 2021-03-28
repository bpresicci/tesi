#voglio trovare un criterio per usare alcune epoche.
import numpy as np
def remove_epochs(idx_best, epoch, n_removed_epochs):
  new_n_epoch = len(idx_best) - n_removed_epochs
  new_idx_best = np.zeros(new_n_epoch)
  new_epoch = np.zeros((new_n_epoch, len(epoch[0]), len(epoch[0][0])))
  for i in range(new_n_epoch):
    new_idx_best[i] = idx_best[i]
    new_epoch[i] = epoch[idx_best[i]]
    return new_idx_best, new_epoch