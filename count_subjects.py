import numpy as np

def count_subjects(subject_start, subject_end):
  check_subjects = np.arange(subject_start, subject_end)
  check_88 = 88 in check_subjects
  check_92 = 92 in check_subjects
  check_100 = 100 in check_subjects
  if  check_88 or check_92 or check_100:
    if check_88:
      if check_92:
        if check_100:
          n_subjects = subject_end - subject_start - 3
        else:
          n_subjects = subject_end - subject_start - 2
      else:
        n_subjects = subject_end - subject_start -1
    else:
      if check_92:
        if check_100:
          n_subjects = subject_end - subject_start - 2
        else:
          n_subjects = subject_end - subject_start -1
      else:
        if check_100:
          n_subjects = subject_end - subject_start -1          
  else:
    n_subjects = subject_end - subject_start     
  return n_subjects*2
