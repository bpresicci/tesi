import numpy as np

def filter_ScorEpochs(dataset, labels, scores, idx, percentage, tot_subjects, tot_conditions, subject_start, subject_end):
  new_dataset = []
  new_labels = []
  new_scores = []
  for condition in range(tot_conditions):
    for subject in range(subject_start, subject_end):
      if subject  not in [88, 92, 100]:
        if condition == 0:
          id_des = subject
        else:
          id_des = subject + tot_subjects
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