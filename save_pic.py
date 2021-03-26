def save_pic(subject, name_folder, task):
    if name_folder != 'none':
        if subject < 10:
            name_image = "/content/%s/subject00%sR0%s" % (name_folder, subject, task)
        if subject > 9 and subject < 100:
            name_image = "/content/%s/subject0%sR0%s" % (name_folder, subject, task)
        if subject > 99:
            name_image = "/content/%s/subject%sR0%s" % (name_folder, subject, task)
    else:
        if subject < 10:
            name_image = '/content/subject00%sR0%s.png' % (subject, task)
        if subject > 9 and subject < 100:
            name_image = '/content/subject0%sR0%s.png' % (subject, task)
        if subject > 99:
            name_image = '/content/subject%sR0%s.png' % (subject, task)
    return name_image