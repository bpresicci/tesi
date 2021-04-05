import numpy as np

def filter_ScorEpochs(dataset, labels, scores_user_specific, percentage):
    new_dataset = []
    new_labels = []
    new_scores = []
    user = 0
    idx_ep = 0
    for epoch in range(len(dataset)):
      scores_mean = np.mean(scores_user_specific[user])
      if scores_user_specific[user][idx_ep] >= (scores_mean * percentage):
          new_dataset = new_dataset + [dataset[epoch]]
          new_labels = new_labels + [labels[epoch]]
          new_scores = new_scores + [scores_user_specific[user]]
      idx_ep += 1
      if idx_ep == len(scores_user_specific[0]):
          idx_ep = 0
          user += 1
          if user < len(scores_user_specific):
            scores_mean = np.mean(scores_user_specific[user])
    new_dataset = np.array(new_dataset)
    new_labels = np.array(new_labels)
    new_scores = np.array(new_scores)
    return new_dataset, new_labels, new_scores
