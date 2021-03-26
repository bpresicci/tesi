def find_name_file(subject, task):
    if subject < 10:
        file = 'S00%sR0%s.edf' % (subject, task)
    if subject > 9 and subject < 100:
        file = 'S0%sR0%s.edf' % (subject, task)
    if subject > 99:
        file = 'S%sR0%s.edf' % (subject, task)
    return file