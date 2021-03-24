def find_name_file(subject, pic_cfg):
    save_pic = 'pic' in pic_cfg.keys()
    if save_pic:
        if subject < 10:
            file = 'S00%sR01.edf' % subject
            name_folder = '/content/%s' % pic_cfg['pic']
            name_subject = '/subject00%s.png' % subject
            name_image = name_folder + name_subject
        if subject > 9 and subject < 100:
            file = 'S0%sR01.edf' % subject
            name_folder = '/content/%s' % pic_cfg['pic']
            name_subject = '/subject0%s.png' % subject
            name_image = name_folder + name_subject
        if subject > 99:
            file = 'S%sR01.edf' % subject
            name_folder = '/content/%s' % pic_cfg['pic']
            name_subject = '/subject%s.png' % subject
            name_image = name_folder + name_subject
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