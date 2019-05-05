import matplotlib.pyplot as plt

def change_fig_theme(fig, ax, background_color, axis_color):
    """
    Change the color scheme for a figure.

    Paramaters
    ----------
    fig : figure handle

    ax : list of axis handles
        If a single axis, pass it as an element in a list.

    background_color : color
        Color of figure background.

    axis_color : color
        Color of axes, labels, tick labels, and titles.

    Returns
    -------
    None - changes are made in place.
    """
    fig.patch.set_facecolor(background_color)

    for i in range(len(ax)):
        ax[i].set_facecolor(background_color)
        ax[i].spines['bottom'].set_color(axis_color)
        ax[i].spines['top'].set_color(axis_color)
        ax[i].spines['right'].set_color(axis_color)
        ax[i].spines['left'].set_color(axis_color)
        ax[i].tick_params(axis='x', colors=axis_color)
        ax[i].tick_params(axis='y', colors=axis_color)
        ax[i].yaxis.label.set_color(axis_color)
        ax[i].xaxis.label.set_color(axis_color)
        ax[i].title.set_color(axis_color)
