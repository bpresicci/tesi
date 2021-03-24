def find_name_file(subject, save_pic, name_folder):
    if save_pic and name_folder:
        if subject < 10:
            file = 'S00%sR01.edf' % subject
            name_folder = '/content/%s' % name_folder
            name_subject = '/subject00%s.png' % subject
            name_image = name_folder + name_subject
        if subject > 9 and subject < 100:
            file = 'S0%sR01.edf' % subject
            name_folder = '/content/%s' % name_folder
            name_subject = '/subject0%s.png' % subject
            name_image = name_folder + name_subject
        if subject > 99:
            file = 'S%sR01.edf' % subject
            name_folder = '/content/%s' % name_folder
            name_subject = '/subject%s.png' % subject
            name_image = name_folder + name_subject
    else:
        if save_pic:
            if subject < 10:
                file = 'S00%sR01.edf' % subject
                name_image = '/content/subject00%s.png' % subject
            if subject > 9 and subject < 100:
                file = 'S0%sR01.edf' % subject
                name_image = '/content/subject0%s.png' % subject
            if subject > 99:
                file = 'S%sR01.edf' % subject
                name_image = '/content/subject%s.png' % subject
        else:
            if subject < 10:
                file = 'S00%sR01.edf' % subject
            if subject > 9 and subject < 100:
                file = 'S0%sR01.edf' % subject
            if subject > 99:
                file = 'S%sR01.edf' % subject
    if save_pic:
            return file, name_image
    else:
        return file