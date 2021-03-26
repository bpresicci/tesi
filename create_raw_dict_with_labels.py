def create_raw_dict_with_labels(data, name_file):
    if "R01" in name_file:
        y = 0 #eyes open
    if "R02" in name_file:
        y = 1 #eyes closed
    subj_name = name_file.replace("name_file[-1:-9]", " ")
    element = {'subject': subj_name, 'data_raw': data, 'label': y}
    return element