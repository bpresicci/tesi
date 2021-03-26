def find_name_file(subject, save_pic, name_folder, task):
    if save_pic and name_folder != 'none':
        if subject < 10:
            file = 'S00%sR0%s.edf' % (subject, task)
            name_image = "/content/%s/subject00%sR0%s" % (name_folder, subject, task)
        if subject > 9 and subject < 100:
            file = 'S0%sR0%s.edf' % (subject, task)
            name_image = "/content/%s/subject0%sR0%s" % (name_folder, subject, task)
        if subject > 99:
            file = 'S%sR0%s.edf' % (subject, task)
            name_image = "/content/%s/subject%sR0%s" % (name_folder, subject, task)
    else:
        if save_pic:
            if subject < 10:
                file = 'S00%sR0%s.edf' % (subject, task)
                name_image = '/content/subject00%sR0%s.png' % (subject, task)
            if subject > 9 and subject < 100:
                file = 'S0%sR0%s.edf' % (subject, task)
                name_image = '/content/subject0%sR0%s.png' % (subject, task)
            if subject > 99:
                file = 'S%sR0%s.edf' % (subject, task)
                name_image = '/content/subject%sR0%s.png' % (subject, task)
        else:
            if subject < 10:
                file = 'S00%sR0%s.edf' % (subject, task)
            if subject > 9 and subject < 100:
                file = 'S0%sR0%s.edf' % (subject, task)
            if subject > 99:
                file = 'S%sR0%s.edf' % (subject, task)
    if not save_pic:
        name_image = '/content/none.png'
    return file, name_image