import numpy as np

def filter_ScorEpochs_media_per_utente(dataset, labels, scores, idx, percentage, tot_conditions, subject_start, subject_end):
  new_dataset = []
  new_labels = []
  new_scores = []
  for condition in range(tot_conditions):
    for subject in range(subject_start, subject_end):
      if subject not in [88, 92, 100]:
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

def filter_ScorEpochs_3(dataset, labels, scores, idx, percentage, tot_conditions, subject_start, subject_end):
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
        position_random = np.arange(len(idx[bool_id]))
        position_random = np.random.shuffle(position_random)
        for i in range(new_tot_epoch_per_user):
          new_dataset = new_dataset + [dataset[subject_ep_idx[position_random[i]]]]#idx_ep[idx_best[epoch]] è l'indice dell'epoca seguendo l'ordine dei punteggi migliori per ogni utent
          new_labels = new_labels + [labels[subject_ep_idx[position_random[i]]]]
          new_scores = new_scores + [scores[subject_ep_idx[position_random[i]]]]
      else:
        continue
  new_dataset = np.array(new_dataset)
  new_labels = np.array(new_labels)
  new_scores = np.array(new_scores)
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_togli_n_ep_peggiori(dataset, labels, scores, idx, percentage):
  #elimino % epoche peggiori a prescindere dall'utente
  new_tot_epoch = int(percentage * len(idx))
  new_dataset = np.zeros((new_tot_epoch, len(dataset[0])))
  new_labels = np.zeros(new_tot_epoch)
  new_scores = np.zeros(new_tot_epoch)
  indeces_best_scores = np.argsort(scores)
  indeces_best_scores = indeces_best_scores[::-1]
  for i in range(new_tot_epoch):
    new_dataset[i] = dataset[indeces_best_scores[i]]
    new_labels[i] = labels[indeces_best_scores[i]]
    new_scores[i] = scores[indeces_best_scores[i]]
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_togli_random(dataset, labels, scores, idx, percentage):
  #elimino % epoche a caso a prescindere dall'utente
  new_tot_epoch = int(percentage * len(idx))
  new_dataset = np.zeros((new_tot_epoch, len(dataset[0])))
  new_labels = np.zeros(new_tot_epoch)
  new_scores = np.zeros(new_tot_epoch)
  indeces_random = np.arange(len(scores))
  indeces_random = np.random.shuffle(indeces_random)
  for i in range(new_tot_epoch):
    new_dataset[i] = dataset[indeces_random[i]]
    new_labels[i] = labels[indeces_random[i]]
    new_scores[i] = scores[indeces_random[i]]
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_togli_n_ep_migliori(dataset, labels, scores, idx, percentage):
  #elimino % epoche peggiori a prescindere dall'utente
  new_tot_epoch = int(percentage * len(idx))
  new_dataset = np.zeros((new_tot_epoch, len(dataset[0])))
  new_labels = np.zeros(new_tot_epoch)
  new_scores = np.zeros(new_tot_epoch)
  indeces_best_scores = np.argsort(scores)
  for i in range(new_tot_epoch):
    new_dataset[i] = dataset[indeces_best_scores[i]]
    new_labels[i] = labels[indeces_best_scores[i]]
    new_scores[i] = scores[indeces_best_scores[i]]
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_migliori_ep_train(dataset, labels, scores, idx, percentage):
  #train = % epoche migliori a prescindere dall'utente; test il resto
  train_tot_epoch = int(percentage * len(idx))
  test_tot_epoch = len(idx) - train_tot_epoch
  train_dataset = np.zeros((train_tot_epoch, len(dataset[0])))
  train_labels = np.zeros(train_tot_epoch)
  test_dataset = np.zeros((test_tot_epoch, len(dataset[0])))
  test_labels = np.zeros(test_tot_epoch)
  indeces_best_scores = np.argsort(scores)
  indeces_best_scores = indeces_best_scores[::-1]
  for i in range(train_tot_epoch):
    train_dataset[i] = dataset[indeces_best_scores[i]]
    train_labels[i] = labels[indeces_best_scores[i]]
  for i in range(test_tot_epoch):
    test_dataset[i] = dataset[indeces_best_scores[i + train_tot_epoch]]
    test_labels[i] = labels[indeces_best_scores[i + train_tot_epoch]]
  return train_dataset, train_labels, test_dataset, test_labels

def filter_ScorEpochs_peggiori_ep_train(dataset, labels, scores, idx, percentage):
  #train = % epoche peggiori a prescindere dall'utente; test il resto
  train_tot_epoch = int(percentage * len(idx))
  test_tot_epoch = len(idx) - train_tot_epoch
  train_dataset = np.zeros((train_tot_epoch, len(dataset[0])))
  train_labels = np.zeros(train_tot_epoch)
  test_dataset = np.zeros((test_tot_epoch, len(dataset[0])))
  test_labels = np.zeros(test_tot_epoch)
  indeces_best_scores = np.argsort(scores)
  for i in range(train_tot_epoch):
    train_dataset[i] = dataset[indeces_best_scores[i]]
    train_labels[i] = labels[indeces_best_scores[i]]
  for i in range(test_tot_epoch):
    test_dataset[i] = dataset[indeces_best_scores[i + train_tot_epoch]]
    test_labels[i] = labels[indeces_best_scores[i + train_tot_epoch]]
  return train_dataset, train_labels, test_dataset, test_labels