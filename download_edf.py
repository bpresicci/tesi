import os

def download_edf(subject_start, subject_end, task):
    subject = subject_start
    while subject < subject_end:
        if subject < 10:
            url = 'https://physionet.org/files/eegmmidb/1.0.0/S00%s/S00%sR0%s.edf' % (subject, subject, task)
        if subject > 9 and subject < 100:
            url = 'https://physionet.org/files/eegmmidb/1.0.0/S0%s/S0%sR0%s.edf' % (subject, subject, task)
        if subject > 99:
            url = 'https://physionet.org/files/eegmmidb/1.0.0/S%s/S%sR0%s.edf' % (subject, subject, task)
        os.system('wget %s' % url)
        subject += 1

