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
import sys
import os

def read_args():
    """How to read and handle command line arguments."""
    if len(sys.argv) < 3:
        print("Error: not enough arguments!")
        print("Usage: GMS.py GMS_FEM FILE1 [FILE2] [...]")
        print("")
        print("Plot one or multiple YSE-profiles from YSE_profile")
        print("")
        print("       Parameters")
        print("       ----------")
        print("       GMS_FEM    The GMS *.fem file in ASCII format")
        print("       FILE1 ..   Output of YSE_profile with this internal structure")
        print("")
        print("                  Column Property                Unit ")
        print("                  ------ ----------------------  ----")
        print("                  0      Distance along profile  m")
        print("                  1      Depth                   m")
        print("                  2      DSIGMA compression      MPa")
        print("                  3      X coordinate            m")
        print("                  4      Y coordinate            m")
        print()
        sys.exit()
    return sys.argv[1], sys.argv[2::]


if __name__ == '__main__':
    from pyGMS.utils import check_ipython
    check_ipython()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import tri
    from matplotlib.ticker import FuncFormatter
    from pyGMS import GMS
    f_fem, f_profiles = read_args()
    # Assume that the structure of each profile file is
    # Dist[m] Z[m] DSIGMA_C[MPa] X[m] Y[m]
    model = GMS(f_fem)
    for f in f_profiles:
        print('Processing', f)
        raw_data = np.loadtxt(f, skiprows=1, delimiter=' ')
        x0, y0 = raw_data[0, 3:5]
        x1, y1 = raw_data[-1, 3:5]
        d = raw_data[:, 0]*0.001              # Distance / km
        z = raw_data[:, 1]*0.001              # Depth / km
        dsigma = np.abs(raw_data[:, 2])*0.001  # Diff. stress / GPa

        # Plot
        gs_kwd = dict(width_ratios=(5, 1), wspace=0.01)
        fig, axes = plt.subplots(1, 2, gridspec_kw=gs_kwd, figsize=(18, 5))

        # Profile
        ax = axes[0]
        triangulation = tri.Triangulation(d, z)
        mappable = ax.tricontourf(triangulation, dsigma, 100)
        model.plot_layer_bounds(x0, y0, x1, y1, lw=2, only='unique',
                                unit='km', ax=ax)
        plt.colorbar(mappable=mappable, ax=ax, label='Yield strength / GPa')
        ax.set_xlabel('Distance / km')
        ax.set_ylabel('Elevation / km')
        ax.set_title('y = ' + str(y0) + 'm')

        ax = axes[1]
        model.plot_topography(ax=ax)
        ax.plot([x0, x1], [y0, y1], color='red', linewidth=2)
        ax.yaxis.tick_right()

        format_km = FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        ax.xaxis.set_major_formatter(format_km)
        ax.yaxis.set_major_formatter(format_km)

        savetitle = os.path.splitext(f)[0] + '.png'
        fig.savefig(savetitle, dpi=100, bbox_inches='tight', facecolor='w')
        print('>', savetitle, 'saved!')
    print('Done!')
