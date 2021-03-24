import matplotlib.pyplot as plt

def plot_and_save(data_to_plot, data, n, ch_start, samples, fs, t, delta, save_pic, name_image='/content/none.png'):
    for i in range(n):
        plt.plot(data_to_plot[i + ch_start, :] + delta * i * 2)
    plt.yticks(range(n) * delta * 2, data.ch_names[ch_start:ch_start + n])
    plt.xticks(range(0, samples + 1, fs), range(t + 1))
    if save_pic:
        image = plt.gcf()
        image.set_size_inches(11, 8)
        image.savefig(name_image, dpi=200)
        image.clear()
        plt.close(image)
    else:
        plt.show()
