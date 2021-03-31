def find_name_file_and_label(subject, condition):
    if condition == 1:
        label = 0
    if condition == 2:
        label = 1
    if subject < 10:
        file = 'S00%sR0%s.edf' % (subject, condition)
    if subject > 9 and subject < 100:
        file = 'S0%sR0%s.edf' % (subject, condition)
    if subject > 99:
        file = 'S%sR0%s.edf' % (subject, condition)
    return file, label