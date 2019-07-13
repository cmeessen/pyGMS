# coding=utf-8
###############################################################################
#                     Copyright (C) 2019 by Christian Meeßen                  #
#                                                                             #
#                          This file is part of pyGMS                         #
#                                                                             #
#       GMTScripts is free software: you can redistribute it and/or modify    #
#     it under the terms of the GNU General Public License as published by    #
#           the Free Software Foundation version 3 of the License.            #
#                                                                             #
#      GMTScripts is distributed in the hope that it will be useful, but      #
#          WITHOUT ANY WARRANTY; without even the implied warranty of         #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      #
#                   General Public License for more details.                  #
#                                                                             #
#      You should have received a copy of the GNU General Public License      #
#       along with Scripts. If not, see <http://www.gnu.org/licenses/>.       #
###############################################################################
import os
from sys import stdout
import numpy as np
import matplotlib as mpl


def show_progress(i_step=None, i_max=None):
    """Show the progress of a process.

    Show progress of a calculation. To initialise use i=ShowProgress()
    where i is the loop counter. To display the progress during the loop
    and to add counts to i use i=ShowProgress(i, i_max) during the loop.

    Parameters
    ----------
    i_step : int
        Calculation step.
    i_max : int
        Maximum calculation step.

    Returns
    -------
    i : int
        Initialises i or returns i_step + 1.

    """
    if i_step is None:
        i_step = 0
        stdout.write("Progress: 0%")
    elif i_step <= i_max:
        progress = np.round((float(i_step) / float(i_max)) * 100, 2)
        stdout.write('\rProgress: %d%%' % progress)
    if i_step == i_max:
        stdout.write("\n")
    stdout.flush()
    return i_step + 1


def check_ipython():
    """Check whether code is executed in an ipython environment."""
    if not os.environ.get('DISPLAY'):
        print('DISPLAY variable not set. Switching to agg')
        try:
            mpl.pyplot.switch_backend('agg')
        except:
            mpl.use('agg')


def afrikakarte(kind='listed'):
    """Get a pretty colormap.

    Returns the afrikakarte colormap (`source <http://soliton.vm.bytemark.co.uk/pub/cpt-city/wkp/lilleskut/tn/afrikakarte.png.index.html>`_)

    Parameters
    ----------
    kind :{'listed', 'linear'}
        'listed' returns a ListedColormap, 'linear' returns a
        LinearSegmentedColormap

    """
    vals = [[127, 168, 203],
            [146, 181, 213],
            [160, 194, 222],
            [173, 203, 230],
            [185, 213, 237],
            [196, 223, 244],
            [206, 231, 249],
            [218, 240, 253],
            [172, 208, 165],
            [168, 198, 143],
            [209, 215, 171],
            [239, 235, 192],
            [222, 214, 163],
            [202, 185, 130],
            [192, 154, 83],
            [236, 236, 236]]
    vals = np.asarray(vals)/256.0
    if kind == 'listed':
        from matplotlib.colors import ListedColormap
        return ListedColormap(vals.tolist(), 'afrikakarte')
    elif kind == 'linear':
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap('afrikakarte', vals.tolist())


if __name__ == '__main__':
    pass
