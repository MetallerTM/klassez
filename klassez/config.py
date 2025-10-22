#! /usr/bin/env python3

import seaborn as sns
from datetime import datetime


def cron(func, *args, **kwargs):
    r""" Decorator: use it to monitor the runtime of a function.  """
    def new_func(*args, **kwargs):
        start_time = datetime.now()

        return_values = func(*args, **kwargs)

        end_time = datetime.now()
        run_time = end_time - start_time
        print(f'Runtime: {run_time}\n')
        return return_values
    return new_func


# Use seaborn's colormaps and save it to a dictionary
CMapsNames = [
    'Accent', r'Accent_r', r'Blues', r'Blues_r', r'BrBG', r'BrBG_r', r'BuGn', r'BuGn_r', r'BuPu', r'BuPu_r',
    r'CMRmap', r'CMRmap_r', r'Dark2', r'Dark2_r', r'GnBu', r'GnBu_r', r'Greens', r'Greens_r', r'Greys', r'Greys_r',
    r'OrRd', r'OrRd_r', r'Oranges', r'Oranges_r', r'PRGn', r'PRGn_r', r'Paired', r'Paired_r', r'Pastel1', r'Pastel1_r',
    r'Pastel2', r'Pastel2_r', r'PiYG', r'PiYG_r', r'PuBu', r'PuBuGn', r'PuBuGn_r', r'PuBu_r', r'PuOr', r'PuOr_r', r'PuRd',
    r'PuRd_r', r'Purples', r'Purples_r', r'RdBu', r'RdBu_r', r'RdGy', r'RdGy_r', r'RdPu', r'RdPu_r', r'RdYlBu', r'RdYlBu_r',
    r'RdYlGn', r'RdYlGn_r', r'Reds', r'Reds_r', r'Set1', r'Set1_r', r'Set2', r'Set2_r', r'Set3', r'Set3_r', r'Spectral',
    r'Spectral_r', r'Wistia', r'Wistia_r', r'YlGn', r'YlGnBu', r'YlGnBu_r', r'YlGn_r', r'YlOrBr', r'YlOrBr_r', r'YlOrRd',
    r'YlOrRd_r', r'afmhot', r'afmhot_r', r'autumn', r'autumn_r', r'binary', r'binary_r', r'bone', r'bone_r', r'brg', r'brg_r',
    r'bwr', r'bwr_r', r'cividis', r'cividis_r', r'cool', r'cool_r', r'coolwarm', r'coolwarm_r', r'copper', r'copper_r', r'cubehelix',
    r'cubehelix_r', r'flag', r'flag_r', r'gist_earth', r'gist_earth_r', r'gist_gray', r'gist_gray_r', r'gist_heat', r'gist_heat_r',
    r'gist_ncar', r'gist_ncar_r', r'gist_rainbow', r'gist_rainbow_r', r'gist_stern', r'gist_stern_r', r'gist_yarg', r'gist_yarg_r',
    r'gnuplot', r'gnuplot2', r'gnuplot2_r', r'gnuplot_r', r'gray', r'gray_r', r'hot', r'hot_r', r'hsv', r'hsv_r', r'icefire',
    r'icefire_r', r'inferno', r'inferno_r', r'magma', r'magma_r', r'mako', r'mako_r', r'nipy_spectral', r'nipy_spectral_r',
    r'ocean', r'ocean_r', r'pink', r'pink_r', r'plasma', r'plasma_r', r'prism', r'prism_r', r'rainbow', r'rainbow_r', r'rocket', r'rocket_r',
    r'seismic', r'seismic_r', r'spring', r'spring_r', r'summer', r'summer_r', r'tab10', r'tab10_r', r'tab20', r'tab20_r', r'tab20b',
    r'tab20b_r', r'tab20c', r'tab20c_r', r'terrain', r'terrain_r', r'twilight', r'twilight_r', r'twilight_shifted',
    r'twilight_shifted_r', r'viridis', r'viridis_r', r'vlag', r'vlag_r', r'winter', r'winter_r'
    ]

global CM, COLORS
CM = {}
for key in CMapsNames:
    CM[key] = sns.color_palette(key, as_cmap=True)
del CMapsNames

# Cmaps for 2D spectra display
CM_2D = {key: CM[key] for key in ['Greys_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Oranges_r', 'Purples_r', 'copper']}


# List of colors

colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:cyan',
          'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown', 'tab:olive',
          'salmon', 'indigo', 'm', 'c', 'g', 'r', 'b', 'k',
          ]

for w in range(10):
    colors += colors
COLORS = tuple(colors)
