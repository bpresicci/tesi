import os

def download_edf(subject_start, subject_end):
    subject = subject_start
    while subject < subject_end:
        if subject < 10:
            url = 'https://physionet.org/files/eegmmidb/1.0.0/S00%s/S00%sR01.edf' % (subject, subject)
        if subject > 9 and subject < 100:
            url = 'https://physionet.org/files/eegmmidb/1.0.0/S0%s/S0%sR01.edf' % (subject, subject)
        if subject > 99:
            url = 'https://physionet.org/files/eegmmidb/1.0.0/S%s/S%sR01.edf' % (subject, subject)
        os.system('wget %s' % url)
        subject += 1

