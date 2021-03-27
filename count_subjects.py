import numpy as np

def count_subjects(subject_start, subject_end):
  check_subjects = np.arange(subject_start, subject_end)
  check_88 = 88 in check_subjects
  check_92 = 92 in check_subjects
  check_100 = 100 in check_subjects
  if  check_88 or check_92 or check_100:
    if 88 in check_subjects:
      n_subjects = subject_end - subject_start -1
    if 92 in np.arange(subject_start, subject_end):
      n_subjects = subject_end - subject_start -1
    if 100 in check_subjects:
        n_subjects = subject_end - subject_start -1
  else:
    n_subjects = subject_end - subject_start     
  return n_subjects*2
