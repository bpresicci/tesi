import numpy as np

def filter_ScorEpochs(dataset, labels, scores, idx, percentage, f, kf, classifiers, user_specific, tot_conditions, subject_start, subject_end):
  scores_kfold = np.zeros((len(classifiers), kf.get_n_splits(dataset, labels)))
  X, y, scores = f(dataset, labels, scores, idx, percentage, user_specific, tot_conditions, subject_start, subject_end)
  for idx_clf in range (len(classifiers)):
    clf = classifiers[idx_clf]
    idx_score = 0
    for train_index, test_index in kf.split(X, y):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      clf.fit(X_train, y_train)
      score = clf.score(X_test, y_test)
      scores_kfold[idx_clf][idx_score] = score
      idx_score += 1
  return scores_kfold

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

def filter_ScorEpochs_togli_n_ep_peggiori(dataset, labels, scores, idx, percentage, user_specific, tot_conditions, subject_start, subject_end):
  #elimino % epoche peggiori a prescindere dall'utente
  if user_specific == False:
    new_tot_epoch = int(percentage * len(labels))
    new_dataset = np.zeros((new_tot_epoch, len(dataset[0])))
    new_labels = np.zeros(new_tot_epoch)
    new_scores = np.zeros(new_tot_epoch)
    indeces_best_scores = np.argsort(scores)
    indeces_best_scores = indeces_best_scores[::-1]
    for i in range(new_tot_epoch):
      new_dataset[i] = dataset[indeces_best_scores[i]]
      new_labels[i] = labels[indeces_best_scores[i]]
      new_scores[i] = scores[indeces_best_scores[i]]
  else: #elimino % epoche peggiori tenendo conto dell'utente
    new_tot_epoch_per_user = int(percentage * len(idx[idx == 1]))
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
          idx_ep = list(zip(*np.where(idx == id_des))) #lista degli indici relativi a quel soggetto nella matrice grande
          indeces_best_scores_per_user = np.argsort(scores[bool_id]) #indici da 0 a tot_epoche-1 che indicano la posizione dei punteggi migliori di quell'utente
          indeces_best_scores_per_user = indeces_best_scores_per_user[::-1]
          for i in range(new_tot_epoch_per_user):
            new_dataset = new_dataset + [dataset[idx_ep[indeces_best_scores_per_user[i]]]]
            new_labels = new_labels + [labels[idx_ep[indeces_best_scores_per_user[i]]]]
            new_scores = new_scores + [scores[idx_ep[indeces_best_scores_per_user[i]]]]
        else:
          continue
    new_dataset = np.array(new_dataset)
    new_labels = np.array(new_labels)
    new_scores = np.array(new_scores)
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_togli_n_ep_migliori(dataset, labels, scores, idx, percentage, user_specific, tot_conditions, subject_start, subject_end):
  #elimino % epoche peggiori a prescindere dall'utente
  if user_specific == False:
    new_tot_epoch = int(percentage * len(idx))
    new_dataset = np.zeros((new_tot_epoch, len(dataset[0])))
    new_labels = np.zeros(new_tot_epoch)
    new_scores = np.zeros(new_tot_epoch)
    indeces_best_scores = np.argsort(scores)
    for i in range(new_tot_epoch):
      new_dataset[i] = dataset[indeces_best_scores[i]]
      new_labels[i] = labels[indeces_best_scores[i]]
      new_scores[i] = scores[indeces_best_scores[i]]
  else: #elimino % epoche migliori tenendo conto dell'utente
    new_tot_epoch_per_user = int(percentage * len(idx[idx == 1]))
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
          idx_ep = list(zip(*np.where(idx == id_des))) #lista degli indici relativi a quel soggetto nella matrice grande
          indeces_best_scores_per_user = np.argsort(scores[bool_id]) #indici da 0 a tot_epoche-1 che indicano la posizione dei punteggi peggiori di quell'utente
          for i in range(new_tot_epoch_per_user):
            new_dataset = new_dataset + [dataset[idx_ep[indeces_best_scores_per_user[i]]]]
            new_labels = new_labels + [labels[idx_ep[indeces_best_scores_per_user[i]]]]
            new_scores = new_scores + [scores[idx_ep[indeces_best_scores_per_user[i]]]]
        else:
          continue
    new_dataset = np.array(new_dataset)
    new_labels = np.array(new_labels)
    new_scores = np.array(new_scores)
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_togli_random(dataset, labels, scores, idx, percentage, user_specific, tot_conditions, subject_start, subject_end):
  #elimino % epoche a caso a prescindere dall'utente
  if user_specific == False:
    new_tot_epoch = int(percentage * len(idx))
    new_dataset = np.zeros((new_tot_epoch, len(dataset[0])))
    new_labels = np.zeros(new_tot_epoch)
    new_scores = np.zeros(new_tot_epoch)
    indeces_random = np.arange(len(scores))
    np.random.shuffle(indeces_random)
    for i in range(new_tot_epoch):
      new_dataset[i] = dataset[indeces_random[i]]
      new_labels[i] = labels[indeces_random[i]]
      new_scores[i] = scores[indeces_random[i]]
  else: #elimino % epoche random tenendo conto dell'utente
    new_tot_epoch_per_user = int(percentage * len(idx[idx == 1]))
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
          idx_ep = list(zip(*np.where(idx == id_des)))
          indeces_random_per_user = np.arange(len(scores[bool_id]))
          np.random.shuffle(indeces_random_per_user)
          for i in range(new_tot_epoch_per_user):
            new_dataset = new_dataset + [dataset[idx_ep[indeces_random_per_user[i]]]]
            new_labels = new_labels + [labels[idx_ep[indeces_random_per_user[i]]]]
            new_scores = new_scores + [scores[idx_ep[indeces_random_per_user[i]]]]
        else:
          continue
    new_dataset = np.array(new_dataset)
    new_labels = np.array(new_labels)
    new_scores = np.array(new_scores)
  return new_dataset, new_labels, new_scores

def filter_ScorEpochs_migliori_ep_train(dataset, labels, scores, idx, percentage, user_specific, tot_conditions, subject_start, subject_end):
  #train = % epoche migliori a prescindere dall'utente; test il resto
  if user_specific == False:
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
  else: #tenendo conto dell'utente
    tot_epoch = len(idx[idx == 1])
    train_tot_epoch_per_user = int(percentage * tot_epoch)
    train_dataset = []
    train_labels = []
    test_dataset = []
    test_labels = []
    for condition in range(tot_conditions):
      for subject in range(subject_start, subject_end):
        if subject not in [88, 92, 100]:
          if condition == 0:
            id_des = subject
          else:
            id_des = -subject
          bool_id = idx == id_des
          idx_ep = list(zip(*np.where(idx == id_des))) #lista degli indici relativi a quel soggetto nella matrice grande
          indeces_best_scores_per_user = np.argsort(scores[bool_id]) #indici da 0 a tot_epoche-1 che indicano la posizione dei punteggi migliori di quell'utente
          indeces_best_scores_per_user = indeces_best_scores_per_user[::-1]
          for i in range(tot_epoch):
            if i < train_tot_epoch_per_user:
              train_dataset = train_dataset + [dataset[idx_ep[indeces_best_scores_per_user[i]]]]
              train_labels = train_labels + [labels[idx_ep[indeces_best_scores_per_user[i]]]]
            else:
              test_dataset = test_dataset + [dataset[idx_ep[indeces_best_scores_per_user[i]]]]
              test_labels = test_labels + [labels[idx_ep[indeces_best_scores_per_user[i]]]]
        else:
          continue
    train_dataset = np.array(train_dataset)
    train_labels = np.array(train_labels)
    test_dataset = np.array(test_dataset)
    test_labels = np.array(test_labels)
  return train_dataset, train_labels, test_dataset, test_labels
