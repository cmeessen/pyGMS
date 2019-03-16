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
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from matplotlib import tri
from collections import OrderedDict
import sys
import os


def show_progress(i_step=None, i_max=None):
    """
    Show progress of a calculation. To initialise use i=ShowProgress()
    where i is the loop counter. To display the progress during the loop
    and to add counts to i use i=ShowProgress(i, i_max) during the loop.

    Parameters:

    * i_step : int
        Calculation step

    * i_max : int
        Maximum calculation step

    Returns:

    * i : int
        Initialises i or returns i_step + 1
    """
    from sys import stdout
    import numpy as np
    if i_step is None:
        i_step = 0
        stdout.write("Progress: 0%")
    elif i_step < i_max - 1:
        progress = np.round((float(i_step) / float(i_max)) * 100, 2)
        stdout.write('\rProgress: %d%%' % progress)
    else:
        stdout.write("\rProgress: 100%\n")
    # TODO: fix bug where `Progress: 100%`` is shown twice
    stdout.flush()
    return i_step + 1


def check_ipython():
    from packaging.markers import Value
    if not os.environ.get('DISPLAY'):
        print('DISPLAY variable not set. Switching to agg')
        try:
            mpl.pyplot.switch_backend('agg')
        except:
            mpl.use('agg')


def afrikakarte(type='listed'):
    """
    Returns the afrikakarte colormap (source:
    http://soliton.vm.bytemark.co.uk/pub/cpt-city/wkp/lilleskut/tn/afrikakarte.png.index.html)

    Parameters
    ----------
    type : string
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
    if type == 'listed':
        from matplotlib.colors import ListedColormap
        return ListedColormap(vals.tolist(), 'afrikakarte')
    elif type == 'linear':
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap('afrikakarte', vals.tolist())


class Layer:
    """
    Base class representing one layer. Takes lists of x, y, and z points as
    well as the layer id. Automatically triangulates the layer when
    initialised. Calling a layer(x,y) will return the z value at the
    specified coordinates.
    """

    def __init__(self, x, y, z, layer_id, name=None):
        self.x = x
        self.y = y
        self.z = z
        self.layer_id = layer_id
        self.name = name
        self.triangulation = None
        self.interpolators = {}
        self.triangulate()
        self.add_var('z', self.z)

    def __call__(self, x, y, var='z'):
        if var not in list(self.interpolators.keys()):
            raise AttributeError(var, 'not in layer')
        return self.interpolators[var](x, y)[()]

    def triangulate(self):
        self.triangulation = tri.Triangulation(self.x, self.y)

    def add_var(self, name, values):
        ip = tri.LinearTriInterpolator(self.triangulation, values)
        self.interpolators[name] = ip


class Well:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = np.asarray(z)
        self.vars = {}

    def __call__(self):
        return self.z

    def __getitem__(self, varname):
        return self.vars[varname]

    def add_var(self, varname, data):
        self.vars[varname] = data

    def grad(self, varname, interp=None):
        """
        Returns the vertical gradient of the variable `varname`.

        Parameters
        ----------
        varname : str

        Returns
        -------
        if interp = None
            np.array of unit 'var_unit/m'
        else
            gradients, depths
        """
        if interp:
            ip = interpolate.interp1d(self.z, self.vars[varname])
            z = np.linspace(self.z.max(), self.z.min(), interp)
            vars = ip(z)
            return np.gradient(vars, z), z
        else:
            z = self.z
            vars = self.vars[varname]
            return np.gradient(vars, z)

    def plot_var(self, varname):
        plt.plot(self.vars[varname], self.z, label=varname)

    def plot_grad(self, varname, scale=1, return_array=False, abs=False):
        """
        Plot the gradient of a variable.

        Parameters
        ----------
        varname : str
        scale : float
            Multiply the gradient by this value.
        return_array : bool
            if `true` will return the gradient array
        """
        grad = self.grad(varname)
        if abs:
            grad = np.abs(grad)
        plt.plot(grad*scale, self.z, label='grad('+varname+')')
        if return_array:
            return grad*scale


class GMS:
    """
    This class provides a basic module to handle ASCII GMS Models in Python.
    Load a GMS model by

    model = GMS(filename)

    Calls
    -----
    * model(x,y)
      will return a virtual well at the specified coordinates

    GetItem
    -------
    * model[request]
      will return a Layer object for the layer with the corresponding ID or name

    Parameters
    ----------
    triangulate : bool
        Will not automatically triangulate the layers if `False`.
    verbosity : int
        Can be 0, 1 or 2. Level 2 is only for debugging.
    """

    def __init__(self, filename, triangulate=True, verbosity=0):
        self.gms_fem = filename

        self.verbose_level = verbosity
        self.data_raw = None
        self.n_points = None
        self.nx = None
        self.ny = None
        self.nz = None
        self._xlim = None
        self._ylim = None
        self._zlim = None
        self.field_dict = None         # Field name -> Field ID
        self.n_layers = None
        self.n_layers_unique = None
        self.layer_dict = None         # Layer ID -> Layer Name
        self.layer_dict_unique = None
        self.layers = None
        self.points_per_layer = None
        self.layer_points = None
        self.surface_heat_flow = None
        self.wells = {}
        self._info_df = None

        print('Loading', self.gms_fem)
        self.read_header()
        self.load_points()
        print('Done!')
        if triangulate:
            self.make_layers()

    def __call__(self, x, y, z=None, var=None):
        if z is None:
            return self.get_well(x, y, var)
        else:
            well = self.get_well(x, y)
            # Get the layer id
            lay_id = np.where(well.z == well.z[well.z >= z][-1])[0][0]
            return [lay_id, self.layer_dict[lay_id]]

    def __getitem__(self, request):
        if isinstance(request, int):
            return self.layers[request]
        elif isinstance(request, str):
            for key in self.layer_dict.keys():
                if self.layer_dict[key] == request:
                    return self.layers[key]
            raise ValueError('Could not find', request)

    def _get_lims_(self):
        xmin, ymin, zmin = self.data_raw[:,0:3].min(axis=0)
        xmax, ymax, zmax = self.data_raw[:,0:3].max(axis=0)
        self._xlim = (xmin, xmax)
        self._ylim = (ymin, ymax)
        self._zlim = (zmin, zmax)

    @staticmethod
    def _in_ipynb_():
        """
        Use this to check whether code runs in a notebook.
        """
        try:
            from IPython import get_ipython
            env = get_ipython().__class__.__name__
            if env == 'ZMQInteractiveShell':
                # Jupyter notebook or qtconsole
                return True
            else:
                return False
        except NameError:
            return False

    @staticmethod
    def _is_unique_layer_(name):
        splitted = name.split('_')
        try:
            int(splitted[-1])
        except ValueError:
            return True
        else:
            return False

    @staticmethod
    def _points_and_dist_(x0, y0, x1, y1, n, scale=1):
        px = np.linspace(x0, x1, num=n)
        py = np.linspace(y0, y1, num=n)
        d = np.sqrt((x0 - px)**2 + (y0 - py)**2)*scale
        return px, py, d

    def _info_(self):
        """
        Make a pretty display of the fundamental model facts.
        """
        if self._info_df is None:
            self._make_info_df_()
        if self._in_ipynb_():
            from IPython.display import display, HTML
            # noinspection PyTypeChecker
            display((HTML(self._info_df.to_html())))
        else:
            print(self._info_df)

    def _make_info_df_(self):
        """
        Create a pandas dataframe for pretty plotting in jupyter
        """
        columns = [self.gms_fem]
        indices = ['Number of layers',
                   'Number of unique layers',
                   'Points per layer',
                   'Number of points',
                   'xlim', 'ylim', 'zlim']
        data = [self.n_layers,
                self.n_layers_unique,
                self.points_per_layer,
                self.n_points,
                self.xlim, self.ylim, self.zlim]
        self._info_df = pd.DataFrame(data=data, index=indices, columns=columns)

    def _v_(self, message, level=1):
        """
        Prints verbose messages
        """
        if self.verbose_level >= level:
            out = ''
            if isinstance(message, tuple):
                for i in message:
                    out += str(i) + ' '
            else:
                out = message
            if level > 0:
                indent = '>'*level + ' '
            else:
                indent = ''
            print(indent + out)

    # noinspection PyPropertyDefinition
    @property
    def info(self):
        self._info_()
        pass

    @property
    def xlim(self):
        if self._xlim is None:
            self._get_lims_()
        return self._xlim

    @property
    def ylim(self):
        if self._ylim is None:
            self._get_lims_()
        return self._ylim

    @property
    def zlim(self):
        if self._zlim is None:
            self._get_lims_()
        return self._zlim

    def compute_surface_heat_flow(self, return_tcond=False, spacing=None,
                                  force=False):
        """

        Parameters
        ----------
        return_tcond : bool
        spacing : None or float
            Use custom lateral spacing for sampling

        Returns
        -------

        """
        if self.surface_heat_flow is not None and force is False:
            print('Already computed. Use `force=True` to re-compute.')
            return
        self._v_('Computing surface heat flow', 0)
        self.layer_add_var('Tcond')
        if spacing is None:
            x_coords = np.unique(self.data_raw[:,0])
            y_coords = np.unique(self.data_raw[:,1])
        else:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            x_coords = np.arange(xmin, xmax+spacing, spacing)
            y_coords = np.arange(ymin, ymax+spacing, spacing)
        x_points, y_points = np.meshgrid(x_coords, y_coords)
        x_points = x_points.flatten()
        y_points = y_points.flatten()
        hflow = []
        if return_tcond:
            tconds = []
        i = show_progress()
        i_max = x_points.shape[0]
        for x, y in zip(x_points, y_points):
            well = self.get_well(x, y, 'T')  # type: Well
            grad = well.grad('T', interp=1000)[0][0]
            # Get the first layer that has a specified thickness
            lay_id = self.get_uppermost_layer(x, y, tmin=1.0)
            # Obtain the thermal conductivity at the point
            t_cond = self.layers[lay_id](x, y, 'Tcond')
            if return_tcond:
                tconds.append(t_cond)
            hflow.append(-t_cond*grad)
            i = show_progress(i, i_max)
        self.surface_heat_flow = np.empty([x_points.shape[0], 3])
        self.surface_heat_flow[:, 0] = x_points
        self.surface_heat_flow[:, 1] = y_points
        self.surface_heat_flow[:, 2] = np.asarray(hflow)
        if return_tcond:
            return tconds

    def get_uppermost_layer(self, x, y, tmin=1.0):
        well = self.get_well(x, y)
        z0 = well.z[0]
        lay_id = 0
        for z in well.z:
            if z0-z > tmin:
                return lay_id
            else:
                z0 = z
                lay_id += 1
        msg = 'Could not find layer at x='+str(x)+', y='+str(y)
        raise RuntimeError(msg)

    def get_well(self, x, y, var=None, store=True):
        """
        By default returns an array of layer depth values at the specified
        coordinates. If var is given, additionally the value of the variable
        will be returned.

        Parameters
        ----------
        x : float
        y : float
        var : str
        store : bool
            If `True` will store the well object in self.wells.

        Returns
        -------
        Well instance
        """
        if (x, y) not in self.wells:
            values = []
            i = 0
            for layer in self.layers:
                self._v_(('Obtaining values for layer', self.layer_dict[i]))
                values.append(layer(x, y))
                i += 1
            well = Well(x, y, np.asarray(values))
            if var is not None:
                values = []
                i = 0
                for layer in self.layers:
                    self._v_(('Obtaining',var,'for layer', self.layer_dict[i]))
                    values.append(layer(x, y, var))
                    i += 1
                well.add_var(var, np.asarray(values))
            if store:
                self.wells[(x, y)] = well
        return self.wells[(x, y)]

    def read_header(self):
        self._v_('Reading headers')
        gms_fem_f = open(self.gms_fem, 'r')
        i = 0            # Line counter
        n_layers = 0
        layer_d = OrderedDict()
        layer_d_unique = OrderedDict()
        field_d = {}
        # Go through header
        for l in gms_fem_f:
            # Make dict for field_name:field_id
            if i == 0 and not l.startswith('# Type: GMS GridPoints'):
                msg = 'Error:', self.gms_fem, 'is not a GMS model.'
                raise RuntimeError(msg)
            if l.startswith('# Field'):
                field_id = int(l.split(' ')[2])
                field_name = l.split(' ')[3][:-1]
                field_d[field_name] = field_id
                self._v_(('Found field', field_name))
            # Get grid size
            elif l.startswith('# Grid_size'):
                self.nx = int(l.split(' ')[2])
                self.ny = int(l.split(' ')[4])
                self.nz = int(l.split(' ')[6])
                self._v_(('Grid size:', self.nx, self.ny, self.nz))
            # Make dict of layer_id:layer_name
            elif l.startswith('# LayerName'):
                layer_id = int(l.split(' ')[2])
                layer_name = l.split(' ')[3][:-1]
                layer_d[layer_id] = layer_name
                n_layers += 1
                self._v_(('Found layer', layer_name))
                if self._is_unique_layer_(layer_name):
                    layer_d_unique[layer_id] = layer_name
            # Stop if header ends
            if not l.startswith('#'):
                break
            i += 1
        gms_fem_f.close()
        self.layer_dict = layer_d
        self.layer_dict_unique = layer_d_unique
        self.field_dict = field_d
        self.n_layers = n_layers
        self.n_layers_unique = len(list(layer_d_unique.keys()))
        self.points_per_layer = self.nx*self.ny
        self._v_('Number of layers:' + str(n_layers))

    def load_points(self):
        self.data_raw = np.loadtxt(self.gms_fem)
        self.n_points = self.data_raw.shape[0]

    def make_layers(self):
        """
        Extracts points of the self.data_raw array that belong to one layer,
        triangulates them, and adds them to the self.layers array.
        """
        print('Triangulating layers')
        i_end = 0
        layers = []
        layer_points_array = []
        for layer_id in range(self.n_layers):
            self._v_(('Triangulating layer', layer_id), 1)
            # Define subset bounds
            i_start = i_end
            i_end = i_start + self.points_per_layer
            points = self.data_raw[i_start:i_end]
            layer_points_array.append(points)
            new_layer = Layer(points[:, 0], points[:, 1], points[:, 2],
                              layer_id, self.layer_dict[layer_id])
            layers.append(new_layer)
        self.layer_points = layer_points_array
        self.layers = layers
        print('Done!')

    def layer_add_var(self, varname):
        fid = self.field_dict[varname]
        i = 0
        self._v_(('Adding', varname, 'to layers'))
        self._v_(('Field id', fid), 2)
        for l in self.layers:
            self._v_(('layer', i), 2)
            i_start = i*self.points_per_layer
            i_end = i_start + self.points_per_layer
            self._v_((i_start, 'to', i_end), 3)
            values = self.data_raw[i_start:i_end][:, fid]
            l.add_var(varname, values)
            i += 1

    def plot_surface_heat_flow(self, **kwds):
        if self.surface_heat_flow is None:
            self.compute_surface_heat_flow()
        x = self.surface_heat_flow[:, 0]
        y = self.surface_heat_flow[:, 1]
        hflow = self.surface_heat_flow[:, 2]*1000
        plt.tricontourf(x, y, hflow, **kwds)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)

    def plot_layer_bounds(self, x0, y0, x1, y1, num=100, lc='black', lw=1,
                          unit='m', only_unique=True, ax=None, xaxis='dist'):
        if unit == 'm':
            scale = 1
        elif unit == 'km':
            scale = 0.001
        else:
            raise ValueError('Unknown unit', unit)

        if only_unique:
            layer_d = self.layer_dict_unique
        else:
            layer_d = self.layer_dict

        if ax is None:
            import matplotlib.pyplot as plt
        else:
            plt = ax

        px, py, dist = self._points_and_dist_(x0, y0, x1, y1, num, scale)
        if xaxis == 'dist':
            d = dist
        elif xaxis == 'x':
            d = px*scale
        elif xaxis == 'y':
            d = py*scale
        else:
            raise ValueError('Unknown xaxis', xaxis)

        for i in list(layer_d.keys()):
            layer = self.layers[i]  # type: Layer
            z = []
            for x, y in zip(px, py):
                z.append(layer(x, y))
            z = np.asarray(z)*scale
            plt.plot(d, z, color=lc, lw=lw)

    def plot_topography(self, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
        else:
            plt = ax
        zmin = np.abs(self.layers[0].z.min())
        zmax = np.abs(self.layers[0].z.max())
        vmin = -1*max(zmin, zmax)
        vmax = max(zmin, zmax)
        return plt.tricontourf(self.layers[0].triangulation, self.layers[0].z,
                               cmap=afrikakarte(), vmin=vmin, vmax=vmax)

    def plot_profile(self, x0, y0, x1, y1, var='T', num=100, ax=None,
                     unit='m', type='filled', xaxis='dist', **kwds):
        """
        Plot a profile of variable `var` along the specified coordinates.

        Parameters
        ----------
        x0, y0, x1, y1 : float
            Coordinates of start end end point
        var : str
            Name of the variable in the GMS file. Can also be 'litho'
        num : int
            Number of sampling points in horizontal direction
        ax : matplotlib.axis
            The axis to plot to
        unit : str
            Defines z unit. 'm' or 'km'
        type : str
            'filled' for a filled contour plot or 'lines' for contours
        xaxis : str
            The dimension to show along the x-axis. 'dist' for distance
            where 0km is at (x0,y0), 'x' or 'y' for the respective axis.
        kwds : dict
           Keywords sent to matplotlib.pyplot.tricontour() or
           matplotlib.pyplot.contourf(), depoending on `type`

        Returns
        -------
        mappable
        """
        valid_types = ['filled', 'lines']
        if type not in valid_types:
            raise ValueError('Invalid type', type)
        if unit == 'km':
            scale = 0.001
        elif unit == 'm':
            scale = 1.0
        else:
            raise ValueError('Unknown unit', unit)
        if ax is None:
            import matplotlib.pyplot as plt
        else:
            plt = ax

        layer_d = self.layer_dict

        # Make the points where to sample the model
        px, py, dist = self._points_and_dist_(x0, y0, x1, y1, num, scale)
        if xaxis == 'dist':
            d = dist
        elif xaxis == 'x':
            d = px*scale
        elif xaxis == 'y':
            d = py*scale
        else:
            raise ValueError('Unknown xaxis', xaxis)
        d = d.tolist()

        # Check if variables are loaded
        if var not in list(self.layers[0].interpolators.keys()):
            self.layer_add_var(var)

        z = []
        v = []
        # Go through layers
        for i in list(layer_d.keys()):
            layer = self.layers[i]  # type: Layer
            # Obtain the depths and values
            for x, y in zip(px, py):
                z.append(layer(x, y))
                v.append(layer(x, y, var))
        z = np.asarray(z)*scale
        v = np.asarray(v)

        # Make a triangulation
        t = tri.Triangulation(d*self.n_layers, z)

        # Print the contours
        if type ==  'filled':
            return plt.tricontourf(t, v, **kwds)
        elif type == 'lines':
            return plt.tricontour(t, v, **kwds)


def read_args():
    """
    How to read and handle command line arguments
    """
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
    check_ipython()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
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
        model.plot_layer_bounds(x0, y0, x1, y1, lw=2, only_unique=True,
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
