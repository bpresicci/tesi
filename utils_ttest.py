from math import ceil
import numpy as np
import pandas as pd
from tesi.corrected_ttest import corrected_dependent_ttest_kfold

def lista_t_p(percentage):
    n1 = 'Rimozione peggiori contro rimozione casuale, con %s' % percentage
    n2 = 'Rimozione peggiori contro rimozione migliori, con %s' % percentage
    n3 = 'Rimozione migliori contro rimozione casuale, con %s' % percentage
    lista_prove = [n1, n2, n3]
    return lista_prove

def compute_n_folds(labels, percentage, kf):
    n_splits = kf.get_n_splits()
    n_training_folds = ceil(len(labels) * percentage * (n_splits - 1) / n_splits)
    n_test_folds = int((len(labels) * percentage) - n_training_folds)
    return n_training_folds, n_test_folds

def compute_errors(scores1, scores2, scores3):
    error1 = np.around((1 - scores1) * 100, 2)
    error2 = np.around((1 - scores2) * 100, 2)
    error3 = np.around((1 - scores3) * 100, 2)
    return error1, error2, error3

def crea_liste(names):
    t_value1 = np.zeros(len(names))
    t_value2 = np.zeros(len(names))
    t_value3 = np.zeros(len(names))
    t_value1_us = np.zeros(len(names))
    t_value2_us = np.zeros(len(names))
    t_value3_us = np.zeros(len(names))
    p_value1 = np.zeros(len(names))
    p_value2 = np.zeros(len(names))
    p_value3 = np.zeros(len(names))
    p_value1_us = np.zeros(len(names))
    p_value2_us = np.zeros(len(names))
    p_value3_us = np.zeros(len(names))
    return t_value1, t_value2, t_value3, p_value1, p_value2, p_value3, t_value1_us, t_value2_us, t_value3_us, p_value1_us, p_value2_us, p_value3_us

def crea_KNN(perc, names, kf, n_random):
    KNN_p_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_m_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_r_scores_all = np.zeros((len(perc), n_random, len(names), kf.get_n_splits()))
    KNN_us_p_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_us_m_scores = np.zeros((len(perc), len(names), kf.get_n_splits()))
    KNN_us_r_scores_all = np.zeros((len(perc), n_random, len(names), kf.get_n_splits()))
    KNN_p = np.zeros((len(perc), len(names)))
    KNN_m = np.zeros((len(perc), len(names)))
    KNN_r = np.zeros((len(perc), n_random, len(names)))
    KNN_us_p = np.zeros((len(perc), len(names)))
    KNN_us_m = np.zeros((len(perc), len(names)))
    KNN_us_r = np.zeros((len(perc), n_random, len(names)))
    return KNN_p_scores, KNN_m_scores, KNN_r_scores_all, KNN_us_p_scores, KNN_us_m_scores, KNN_us_r_scores_all, KNN_p, KNN_m, KNN_r, KNN_us_p, KNN_us_m, KNN_us_r

def do_corrected_ttest(error_p, error_m, error_r, kf, alpha):
    t_value1, df, cv, p_value1 = corrected_dependent_ttest_kfold(error_p, error_r, kf.get_n_splits(), alpha)
    t_value2, df, cv, p_value2 = corrected_dependent_ttest_kfold(error_p, error_m, kf.get_n_splits(), alpha)
    t_value3, df, cv, p_value3 = corrected_dependent_ttest_kfold(error_m, error_r, kf.get_n_splits(), alpha)
    return t_value1, t_value2, t_value3, p_value1, p_value2, p_value3, df, cv

def concatena_current(t_value1, t_value2, t_value3, t_value1_us, t_value2_us, t_value3_us, p_value1, p_value2, p_value3, p_value1_us, p_value2_us, p_value3_us):
    tabella_t_current = pd.concat([pd.DataFrame([t_value1]), pd.DataFrame([t_value2]), pd.DataFrame([t_value3])], axis=0)
    tabella_p_current = pd.concat([pd.DataFrame([p_value1]), pd.DataFrame([p_value2]), pd.DataFrame([p_value3])], axis=0)
    tabella_t_us_current = pd.concat([pd.DataFrame([t_value1_us]), pd.DataFrame([t_value2_us]), pd.DataFrame([t_value3_us])], axis=0)
    tabella_p_us_current = pd.concat([pd.DataFrame([p_value1_us]), pd.DataFrame([p_value2_us]), pd.DataFrame([p_value3_us])], axis=0)
    return tabella_t_current, tabella_p_current, tabella_t_us_current, tabella_p_us_current

def if_idx_perc_0(tabella_t_current, tabella_p_current, tabella_t_us_current, tabella_p_us_current, lista_prove):
    tabella_t = tabella_t_current
    tabella_p = tabella_p_current
    tabella_t_us = tabella_t_us_current
    tabella_p_us = tabella_p_us_current
    tabella_t.index = lista_prove
    tabella_p.index = lista_prove
    tabella_t_us.index = lista_prove
    tabella_p_us.index = lista_prove
    return tabella_t, tabella_p, tabella_t_us, tabella_p_us

def crea_tabelle(tabella_t_current, tabella_p_current, tabella_t_us_current, tabella_p_us_current, lista_prove, tabella_t, tabella_p, tabella_t_us, tabella_p_us):
    tabella_t = pd.concat([tabella_t, tabella_t_current], axis=0)
    tabella_p = pd.concat([tabella_p, tabella_p_current], axis=0)
    tabella_t_us = pd.concat([tabella_t_us, tabella_t_us_current], axis=0)
    tabella_p_us = pd.concat([tabella_p_us, tabella_p_us_current], axis=0)
    tabella_t.index = lista_prove
    tabella_p.index = lista_prove
    tabella_t_us.index = lista_prove
    tabella_p_us.index = lista_prove
    return tabella_t, tabella_p, tabella_t_us, tabella_p_us

def metti_colonne(tabella_t, tabella_p, tabella_t_us, tabella_p_us, names):
    tabella_t.columns = names
    tabella_p.columns = names
    tabella_t_us.columns = names
    tabella_p_us.columns = names
    return tabella_t, tabella_p, tabella_t_us, tabella_p_us

def salva(tabella_t, tabella_p, tabella_t_us, tabella_p_us):
    tabella_t.to_excel(r'tabella_t.xlsx', index=True, header=True)
    tabella_p.to_excel(r'tabella_p.xlsx', index=True, header=True)
    tabella_t_us.to_excel(r'tabella_t_us.xlsx', index=True, header=True)
    tabella_p_us.to_excel(r'tabella_p_us.xlsx', index=True, header=True)

