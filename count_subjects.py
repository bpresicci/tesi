import numpy as np

def count_subjects(subject_start, subject_end):
    n_subjects = subject_end - subject_start
    check_subjects = np.arange(subject_start, subject_end)
    if 88 in check_subjects:
        n_subject -= 1
    if 92 in np.arange(subject_start, subject_end):
        n_subject -= 1
    if 100 in check_subjects:
        n_subject -= 1
    return n_subjects*2
