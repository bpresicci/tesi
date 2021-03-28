def find_idx_ep(cfg, raw_data):
    epLen = cfg['windowL'] * cfg['fs']  # Number of samples of each epoch (for each channel)
    dataLen = len(raw_data[0])  # Total number of samples
    isOverlap = 'wOverlap' in cfg.keys()  # isOverlap = True if the user wants a sliding window; the user will assign the value in seconds of the overlap desired to the key 'wOverlap'
    if isOverlap:
        idx_jump = (cfg['windowL'] - cfg['wOverlap']) * cfg['fs']  # idx_jump is the number of samples that separates the beginning of an epoch and the following one
    else:
        idx_jump = epLen
    idx_ep = range(0, dataLen - epLen + 1, idx_jump)  # Indexes from which start each epoch
    return idx_ep