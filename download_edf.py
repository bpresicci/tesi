import os

def download_edf(subject_start, subject_end, condition):
    for subject in range(subject_start, subject_end):
        if subject not in [88, 92, 100]:
            if subject < 10:
                url = 'https://physionet.org/files/eegmmidb/1.0.0/S00%s/S00%sR0%s.edf' % (subject, subject, condition)
            if subject > 9 and subject < 100:
                url = 'https://physionet.org/files/eegmmidb/1.0.0/S0%s/S0%sR0%s.edf' % (subject, subject, condition)
            if subject > 99:
                url = 'https://physionet.org/files/eegmmidb/1.0.0/S%s/S%sR0%s.edf' % (subject, subject, condition)
            os.system('wget %s' % url)
        else:
            continue

