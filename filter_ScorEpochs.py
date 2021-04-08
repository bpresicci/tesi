import numpy as np

def filter_ScorEpochs_1(dataset, labels, scores, idx, percentage, tot_conditions, subject_start, subject_end):
  new_dataset = []
  new_labels = []
  new_scores = []
  for condition in range(tot_conditions):
    for subject in range(subject_start, subject_end):
      if subject  not in [88, 92, 100]:
        if condition == 0:
          id_des = subject
        else:
          id_des = -subject
        bool_id = idx == id_des
        idx_ep = zip(*np.where(idx == id_des))
        scores_mean = np.mean(scores[bool_id])
        for epoch in idx_ep:
          if scores[epoch] >= (scores_mean * percentage):
            new_dataset = new_dataset + [dataset[epoch]]
            new_labels = new_labels + [labels[epoch]]
            new_scores = new_scores + [scores[epoch]]
      else:
        continue
  new_dataset = np.array(new_dataset)
  new_labels = np.array(new_labels)
  new_scores = np.array(new_scores)
  return new_dataset, new_labels, new_scores


def filter_ScorEpochs_2(dataset, labels, scores, idx, percentage, tot_conditions, subject_start, subject_end):
  new_dataset = []
  new_labels = []
  new_scores = []
  for condition in range(tot_conditions):
    for subject in range(subject_start, subject_end):
      if subject  not in [88, 92, 100]:
        if condition == 0:
          id_des = subject
        else:
          id_des = -subject
        bool_id = idx == id_des
        new_tot_epoch_per_user = int(percentage * len(idx[bool_id]))
        subject_ep_idx = list(zip(*np.where(idx == id_des))) #indici delle epoche dell'utente specifico
        position_best_scores = np.argsort(scores[bool_id]) #posizioni dei punteggi dell'utente nel vettore scores[bool_id] dalla peggiore alla migliore
        position_best_scores = position_best_scores[::-1] #posizioni dalla migliore alla peggiore
        for i in range(new_tot_epoch_per_user):
          new_dataset = new_dataset + [dataset[subject_ep_idx[position_best_scores[i]]]]#idx_ep[idx_best[epoch]] è l'indice dell'epoca seguendo l'ordine dei punteggi migliori per ogni utent
          new_labels = new_labels + [labels[subject_ep_idx[position_best_scores[i]]]]
          new_scores = new_scores + [scores[subject_ep_idx[position_best_scores[i]]]]
      else:
        continue
  new_dataset = np.array(new_dataset)
  new_labels = np.array(new_labels)
  new_scores = np.array(new_scores)
  return new_dataset, new_labels, new_scores