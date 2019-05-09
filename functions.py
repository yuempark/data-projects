import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

def make_colormap(seq):
    """
    Return a LinearSegmentedColormap.

    Parameters
    ----------
    seq : list
        a sequence of floats and RGB-tuples. The floats should be increasing and in the interval (0,1).

    Returns
    -------
    colormap : LinearSegmentedColormap

    Example
    -------
    color_converter = matplotlib.colors.ColorConverter().to_rgb
    colormap = make_colormap([color_converter('C0'), color_converter('C1'), 0.33,
                              color_converter('C1'), color_converter('C2'), 0.66,
                              color_converter('C3')])
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return colors.LinearSegmentedColormap('CustomMap', cdict)
