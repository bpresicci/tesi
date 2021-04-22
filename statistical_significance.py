import numpy as np
import mne
from tesi.find_name_file_and_label import * #specific for this dataset
from tesi.feature_extraction import feature_extraction_1
from scorepochs.Python.scorEpochs import scorEpochs
from tesi.filter_ScorEpochs import filter_ScorEpochs
from tesi.filter_ScorEpochs import filter_ScorEpochs_togli_n_ep_peggiori
from tesi.filter_ScorEpochs import filter_ScorEpochs_togli_n_ep_migliori
from tesi.filter_ScorEpochs import filter_ScorEpochs_togli_random
from tesi.filter_ScorEpochs import filter_ScorEpochs_migliori_ep_train
from tesi.calculate_acc import accuracy


def create_dataset(conditions, subject_start, subject_end, cfg):
    X_user_specific = []
    y_user_specific = []
    scores_user_specific = []
    for condition in range(conditions):
        for subject in range(subject_start, subject_end):
            if subject not in [88, 92, 100]:
                print(subject)
                file, label = find_name_file_and_label(subject, condition)
                data = mne.io.read_raw_edf(file)
                raw_data = data.get_data()
                nCh = len(raw_data)
                idx_best, epoch, scores = scorEpochs(cfg, raw_data)
                if label:
                    y_per_epoch = np.ones(len(epoch))
                else:
                    y_per_epoch = np.zeros(len(epoch))
                psd = feature_extraction_1(cfg, cfg['freqRange'], nCh, epoch)
                X_user_specific = X_user_specific + [psd]
                y_user_specific = y_user_specific + [y_per_epoch]
                scores_user_specific = scores_user_specific + [scores]
                if subject == subject_start and condition == 0:
                    id = np.array(subject * np.ones(len(epoch)))
                elif condition == 0:
                    id = np.append(id, np.array(subject * np.ones(len(epoch))))
                elif condition != 0:
                    id = np.append(id, np.array(-subject * np.ones(len(epoch))))
            else:
                continue
    X_user_specific = np.array(X_user_specific)
    y_user_specific = np.array(y_user_specific)
    scores_user_specific = np.array(scores_user_specific)
    total_subjects = len(X_user_specific)
    tot_epochs = len(X_user_specific[0])
    tot_channels = len(X_user_specific[0][0])
    tot_samples = len(X_user_specific[0][0][0])
    X_all = np.reshape(X_user_specific, [total_subjects * tot_epochs, tot_channels * tot_samples])
    y_all = np.reshape(y_user_specific, [total_subjects * tot_epochs])
    scores_all = np.reshape(scores_user_specific, [total_subjects * tot_epochs])
    return X_all, y_all, scores_all, id

def kfold_nofiltri(names, classifiers, kf, X_all, y_all):
    scores_no_scorEpochs_kfold = np.zeros((len(names), kf.get_n_splits(X_all, y_all)))
    accuracy_no_scorEpochs_kfold_mean = np.zeros(len(names))
    accuracy_no_scorEpochs_kfold_var = np.zeros(len(names))
    idx_clf = 0
    for name, clf in zip(names, classifiers):
        idx_score = 0
        print(name)
        for train_index, test_index in kf.split(X_all, y_all):
            X_train, X_test = X_all[train_index], X_all[test_index]
            y_train, y_test = y_all[train_index], y_all[test_index]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores_no_scorEpochs_kfold[idx_clf][idx_score] = score
            idx_score += 1
        accuracy_no_scorEpochs_kfold_mean[idx_clf] = np.mean(scores_no_scorEpochs_kfold[idx_clf])
        accuracy_no_scorEpochs_kfold_var[idx_clf] = np.var(scores_no_scorEpochs_kfold[idx_clf])
        print("ACCURACY mean: ", accuracy_no_scorEpochs_kfold_mean[idx_clf])
        print("ACCURACY variance: ", accuracy_no_scorEpochs_kfold_var[idx_clf])
        print("\n")
        idx_clf += 1
    return accuracy_no_scorEpochs_kfold_mean, accuracy_no_scorEpochs_kfold_var

def hold_out_nofiltri(names, classifiers, X_A, y_A, X_B, y_B):
    scores_no_scorEpochs_ho = np.zeros((len(names), 2))
    accuracy_no_scorEpochs_ho_mean = np.zeros(len(names))
    accuracy_no_scorEpochs_ho_var = np.zeros(len(names))
    idx_clf = 0
    for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(X_A, y_A)
        score = clf.score(X_B, y_B)
        scores_no_scorEpochs_ho[idx_clf][0] = score
        clf.fit(X_B, y_B)
        score = clf.score(X_A, y_A)
        scores_no_scorEpochs_ho[idx_clf][1] = score
        accuracy_no_scorEpochs_ho_mean[idx_clf] = np.mean(scores_no_scorEpochs_ho[idx_clf])
        accuracy_no_scorEpochs_ho_var[idx_clf] = np.var(scores_no_scorEpochs_ho[idx_clf])
        print("ACCURACY mean: ", accuracy_no_scorEpochs_ho_mean[idx_clf])
        print("ACCURACY variance: ", accuracy_no_scorEpochs_ho_var[idx_clf])
        print("\n")
    return accuracy_no_scorEpochs_ho_mean, accuracy_no_scorEpochs_ho_var

def applica_filtri(names, classifiers, kf, n_random, percentage, conditions, subject_start, subject_end, X_all, y_all, scores_all, id):
    for user_specific in [False, True]:
        scores_togli_random_kfold = np.zeros((n_random, len(names), kf.get_n_splits()))
        if user_specific == False:
            scores_togli_n_ep_peggiori_kfold = filter_ScorEpochs(X_all, y_all, scores_all, id, percentage, filter_ScorEpochs_togli_n_ep_peggiori, kf, classifiers, user_specific, conditions, subject_start, subject_end)
            scores_togli_n_ep_migliori_kfold = filter_ScorEpochs(X_all, y_all, scores_all, id, percentage, filter_ScorEpochs_togli_n_ep_migliori, kf, classifiers, user_specific, conditions, subject_start, subject_end)
            accuracy_togli_n_ep_peggiori_kfold_mean, accuracy_togli_n_ep_peggiori_kfold_var = accuracy(scores_togli_n_ep_peggiori_kfold)
            accuracy_togli_n_ep_migliori_kfold_mean, accuracy_togli_n_ep_migliori_kfold_var = accuracy(scores_togli_n_ep_migliori_kfold)
            for i in range(n_random):
                scores_togli_random_kfold[i] = filter_ScorEpochs(X_all, y_all, scores_all, id, percentage, filter_ScorEpochs_togli_random, kf, classifiers, user_specific, conditions, subject_start, subject_end)
            accuracy_togli_random_kfold_mean, accuracy_togli_random_kfold_var = accuracy(scores_togli_random_kfold)
        else:
            scores_togli_random_kfold_us = []
            scores_togli_n_ep_peggiori_kfold_us = filter_ScorEpochs(X_all, y_all, scores_all, id, percentage, filter_ScorEpochs_togli_n_ep_peggiori, kf, classifiers, user_specific, conditions, subject_start, subject_end)
            scores_togli_n_ep_migliori_kfold_us = filter_ScorEpochs(X_all, y_all, scores_all, id, percentage, filter_ScorEpochs_togli_n_ep_migliori, kf, classifiers, user_specific, conditions, subject_start, subject_end)
            accuracy_togli_n_ep_peggiori_kfold_mean_us, accuracy_togli_n_ep_peggiori_kfold_var_us = accuracy(scores_togli_n_ep_peggiori_kfold_us)
            accuracy_togli_n_ep_migliori_kfold_mean_us, accuracy_togli_n_ep_migliori_kfold_var_us = accuracy(scores_togli_n_ep_migliori_kfold_us)
            for i in range(n_random):
                scores_togli_random_kfold_us = scores_togli_random_kfold_us + [filter_ScorEpochs(X_all, y_all, scores_all, id, percentage, filter_ScorEpochs_togli_random, kf, classifiers, user_specific, conditions, subject_start, subject_end)]
            scores_togli_random_kfold_us = np.array(scores_togli_random_kfold_us)
            accuracy_togli_random_kfold_mean_us, accuracy_togli_random_kfold_var_us = accuracy(scores_togli_random_kfold_us)
    return scores_togli_n_ep_peggiori_kfold, accuracy_togli_n_ep_peggiori_kfold_mean, accuracy_togli_n_ep_peggiori_kfold_var, scores_togli_n_ep_migliori_kfold, accuracy_togli_n_ep_migliori_kfold_mean, accuracy_togli_n_ep_migliori_kfold_var, scores_togli_random_kfold, accuracy_togli_random_kfold_mean, accuracy_togli_random_kfold_var, scores_togli_n_ep_peggiori_kfold_us, accuracy_togli_n_ep_peggiori_kfold_mean_us, accuracy_togli_n_ep_peggiori_kfold_var_us, scores_togli_n_ep_migliori_kfold_us, accuracy_togli_n_ep_migliori_kfold_mean_us, accuracy_togli_n_ep_migliori_kfold_var_us, scores_togli_random_kfold_us, accuracy_togli_random_kfold_mean_us,accuracy_togli_random_kfold_var_us

def create_KNN_matrices(perc, names, n_random, kf):
    KNN_p = np.zeros((len(perc), len(names)))
    KNN_m = np.zeros((len(perc), len(names)))
    KNN_r = np.zeros((len(perc), n_random, len(names)))
    KNN_us_p = np.zeros((len(perc), len(names)))
    KNN_us_m = np.zeros((len(perc), len(names)))
    KNN_us_r = np.zeros((len(perc), n_random, len(names)))

    KNN_p_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_m_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_r_scores = np.zeros((len(perc), n_random, len(names), kf.get_n_splits()))
    KNN_us_p_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_us_m_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_us_r_scores = np.zeros((len(perc), n_random, len(names), kf.get_n_splits()))
    return KNN_p_scores, KNN_p, KNN_m_scores, KNN_m, KNN_r_scores, KNN_r, KNN_us_p_scores, KNN_us_p, KNN_us_m_scores, KNN_us_m, KNN_us_r_scores, KNN_us_r
