import matplotlib.pyplot as plt

def display_close(fig):
    display(fig)
    plt.close(fig)



def display_text(text, fontsize=18):
    fig = plt.figure(figsize=(12,0.1))
    plt.title(text, fontsize=fontsize)
    plt.axis('off')
    display(fig)
    plt.close(fig)
    