def find_name_and_label(name_file):
    if "R01" in name_file:
        label = 0 #eyes open
    if "R02" in name_file:
        label = 1 #eyes closed
    del_task = name_file[-1:-8:-1]
    del_task = del_task[::-1]
    subj_name = name_file.replace(del_task, " ")
    return subj_name, label