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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import matplotlib as mpl
from matplotlib import tri
from collections import OrderedDict
from .utils import show_progress
from .utils import afrikakarte


class Layer:
    """A class representing a layer.

    Base class representing one layer. Takes lists of x, y, and z points as
    well as the layer id. Automatically triangulates the layer when
    initialised. Calling a layer(x,y) will return the z value at the
    specified coordinates.
    """

    def __init__(self, x, y, z, layer_id, name=None):
        """Initialise the object.

        Initialise a layer object with the given parameters. The length of `x`,
        `y` and `z` must be equal.

        Parameters
        ----------
        x : list, np.array
            List or 1D array of x values.
        y : list, np.array
            List or 1D array of y values.
        z : list, np.array
            List or 1D array of z values.
        layer_id : int
            ID of the layer.
        name : str, optional
            Name of the layer.

        """
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
        """Return an interpolated value of a variable at a coordinate.

        Returns the value of a variable at the specified coordinate.

        Parameters
        ----------
        x : float
            The x-coordinate.
        y : float
            The y-coordinate.
        var : str
            The name of the variable, by default `z`.

        Returns
        -------
        float

        """
        if var not in list(self.interpolators.keys()):
            raise AttributeError(var, 'not in layer')
        return self.interpolators[var](x, y)[()]

    def triangulate(self):
        """Perform the triangulation."""
        self.triangulation = tri.Triangulation(self.x, self.y)

    def add_var(self, name, values):
        """Add a variable to the layer.

        Adds a variable to the layer. Interpolated values of the variable can
        be accessed using the `interpolators[name]` variable of the layer.

        Parameters
        ----------
        name : str
            Name of the variable.
        values : list, np.ndarray
            List or 1D-array of values that are ordered like `x` and `y`
            coordinates.

        """
        ip = tri.LinearTriInterpolator(self.triangulation, values)
        self.interpolators[name] = ip


class Well:
    """A class representing a vertical well in a GMS model.

    This class represents a virtual well that can be extracted from a GMS
    model.

    """

    def __init__(self, x, y, z):
        """Inititalise well.

        Parameters
        ----------
        x : float
            x-coordinate of the well.
        y : float
            y-coordinate of the well.
        z : list, np.ndarray
            List or 1D array of z values at the coordinate (x, y).

        """
        self.x = x
        self.y = y
        self.z = np.asarray(z)
        self.vars = {}
        self.zero_threshold = 0.1  # Minimum layer thickness

    def __call__(self):
        """Return the z-values array.

        Returns
        -------
        self.z : np.array

        """
        return self.z

    def __getitem__(self, varname):
        """Obtain a variable.

        Parameters
        ----------
        varname : str
            The name of the variable.

        Returns
        -------
        self.vars[varname] : np.ndarray
            The values of the variable.

        """
        return self.vars[varname]

    def add_var(self, varname, data):
        """Add a variable.

        Adds a variable to the well and stores it in `self.vars[varname]`.

        Parameters
        ----------
        varname : str
            Name of the variable.
        data : list, np.ndarray
            List or 1D array with the variable values.

        """
        self.vars[varname] = data

    def get_interpolated_var(self, nz, varname):
        """Interpolate a variable.

        Computes values of variable `varname` at `nz` equally spaced points
        along the well.

        Parameters
        ----------
        nz : int
            Number of points incl. start point and end point.
        varname : str
            Variable name.

        Returns
        -------
        z : np.array
            The depth values.
        vars : np.array
            The values of the variable at z.
        layer_ids : np.array
            The corresponding layer ids.

        """
        ip = interpolate.interp1d(self.z, self.vars[varname])
        z = np.linspace(self.z.max(), self.z.min(), nz)
        var_values = ip(z)
        # Obtain the layer ids for each point
        g_layer_tops, g_z = np.meshgrid(self.z, z)
        cond = g_z <= g_layer_tops
        layer_ids = np.sum(cond, axis=1) - 1
        # Check the thicknesses of the corresponding layer ids and replace if
        # thickness is less than a given threshold
        zero_t_lay_ids = np.argwhere(-1*np.diff(self.z) <= self.zero_threshold)
        # incrementally increase those ids
        for i in zero_t_lay_ids:
            layer_ids[layer_ids == i[0]] = i[0] + 1
        return z, var_values, layer_ids

    def grad(self, varname, interp=None):
        """Obtain the gradient of a variable.

        Returns the vertical gradient of the variable `varname`.

        Parameters
        ----------
        varname : str
            Variable name.
        interp : int, optional
            Number of equally spaced points where grad should be calculated.

        Returns
        -------
        if interp = None:
            np.array of unit 'var_unit/m'
        else
            gradients, depths

        """
        if interp:
            z, values, layer_ids = self.get_interpolated_var(interp, varname)
            return np.gradient(values, z), z
        else:
            z = self.z
            values = self.vars[varname]
            return np.gradient(values, z)

    def plot_var(self, varname, ax=None, **kwds):
        """Plot a variable.

        Plots a variable over depth.

        Parameters
        ----------
        varname : str
            Name of the variable.
        ax : matplotlib.axes, optional
            An existing axes object.
        kwds : dict, optional
            Keywords forwarded to `matplotlib.axes.plt`.

        """
        kwds.setdefault('label', varname)
        ax = ax or plt.axes()
        ax.plot(self.vars[varname], self.z, **kwds)

    def plot_grad(self, varname, scale=1, return_array=False, absolute=False,
                  ax=None, **kwds):
        """Plot the gradient of a variable.

        Plot the gradient of a variable.

        Parameters
        ----------
        varname : str
            Name of the variable.
        scale : float
            Multiply the gradient by this value.
        return_array : bool
            If `true` will return the gradient array.
        absolute : bool
            Plot the absolute value of `varname`.
        ax : matplotlib.axes, optional
            An existing matplotlib.axes.
        kwds :
            Keywords that go to `matplotlib.axes.plot`.

        Returns
        -------
        if `return_array = False`:
            Nothing
        else:
            The gradient array scaled by `scale`.

        """
        ax = ax or plt.axes()
        grad = self.grad(varname)
        if absolute:
            grad = np.abs(grad)
        kwds.setdefault('label', 'grad('+varname+')')
        ax.plot(grad*scale, self.z, **kwds)
        if return_array:
            return grad*scale


class GMS:
    """The main class of pyGMS.

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
      will return a Layer object for the layer with the corresponding ID or
      name

    Parameters
    ----------
    triangulate : bool
        Will not automatically triangulate the layers if `False`.
    verbosity : int
        Can be 0, 1 or 2. Level 2 is only for debugging.

    """

    # TODO: Make a Material class which contains all relevant data
    # TODO: Make a Body class
    # TODO: Plotting uses different ways to handle unit scaling. Rewrite the
    #  methods in a way that they use mpl.ticker.FuncFormatter (see plot_yse)

    def __init__(self, filename=None, triangulate=True, verbosity=0):
        """Initialise GMS class."""
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
        self.layer_map = None  # Relate refined ID to unique ID
        self.layers = None
        self.points_per_layer = None
        self.layer_points = None
        self.surface_heat_flow = None
        self._info_df = None
        self.t_ductile = None
        self.t_brittle = None

        # Storage
        self.wells = {}
        self._strength_profiles = {}
        self.integrated_strength = None
        self.materials_db = None     # Materials databse
        self.body_materials = None   # Materials for each body
        self.elastic_thickness = {}  # effective elastic thickness

        # Load the model
        self.load_model()
        if triangulate and filename:
            self.make_layers()

    def __call__(self, x, y, z=None, var=None):
        """Define what happens if calling using coordinates."""
        if z is None:
            return self.get_well(x, y, var)
        else:
            well = self.get_well(x, y)
            # Get the layer id
            lay_id = np.where(well.z == well.z[well.z >= z][-1])[0][0]
            return [lay_id, self.layer_dict[lay_id]]

    def __getitem__(self, request):
        """Getitem returns layers if id or names are given."""
        if isinstance(request, int):
            return self.layers[request]
        elif isinstance(request, str):
            for key in self.layer_dict.keys():
                if self.layer_dict[key] == request:
                    return self.layers[key]
            raise ValueError('Could not find', request)

    def _get_lims_(self):
        xmin, ymin, zmin = self.data_raw[:, 0:3].min(axis=0)
        xmax, ymax, zmax = self.data_raw[:, 0:3].max(axis=0)
        self._xlim = (xmin, xmax)
        self._ylim = (ymin, ymax)
        self._zlim = (zmin, zmax)

    @staticmethod
    def _in_ipynb_():
        """Use this to check whether code runs in a notebook."""
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
        except AttributeError:
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
        """Make a pretty display of the fundamental model facts."""
        if self._info_df is None:
            self._make_info_df_()
        if self._in_ipynb_():
            from IPython.display import display, HTML
            # noinspection PyTypeChecker
            display((HTML(self._info_df.to_html())))
        else:
            print(self._info_df)

    def _make_info_df_(self):
        """Create a pandas dataframe for pretty plotting in jupyter."""
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
        """Print verbose messages."""
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
        """Display info about the pyGMS object."""
        self._info_()

    @property
    def xlim(self):
        """Obtain the x-limits as tuple."""
        if self._xlim is None:
            self._get_lims_()
        return self._xlim

    @property
    def ylim(self):
        """Obtain the y-limits as tuple."""
        if self._ylim is None:
            self._get_lims_()
        return self._ylim

    @property
    def zlim(self):
        """Obtain the z-limits as tuple."""
        if self._zlim is None:
            self._get_lims_()
        return self._zlim

    @staticmethod
    def sigma_byerlee(material, z, mode):
        """Compute the brittle differential stress.

        Compute the byerlee differential stress. Requires the material
        properties friction coefficient (f_f), pore fluid factor (f_p) and
        the bulk density (rho_b).

        Parameters
        ----------
        material : dict
            Dictionary with the material properties in SI units. The
            required keys are 'f_f_e', 'f_f_c', 'f_p' and 'rho_b'
        z : float
            Depth below surface in m
        mode : str
            'compression' or 'extension'

        Returns
        -------
            sigma_d : float

        """
        if mode == 'compression':
            f_f = material['f_f_c'][0]
        elif mode == 'extension':
            f_f = material['f_f_e'][0]
        else:
            raise ValueError('Invalid parameter for mode:', mode)
        f_p = material['f_p'][0]
        rho_b = material['rho_b'][0]
        g = 9.81  # m/s2
        return f_f*rho_b*g*z*(1.0 - f_p)

    @staticmethod
    def sigma_diffusion(material, temp, strain_rate):
        """Compute differential stress for diffusion creep.

        Computes differential stress for diffusion creept at specified
        temperature and strain rate. Material properties require grain size
        'd', grain size exponent 'm', preexponential scaling factor for
        diffusion creep 'a_f', and activation energy 'q_f'.

        For diffusion creep, n=1.

        Parameters
        ----------
        material : dict
            Dictionary with the material properties in SI units. Required
            keys are 'd', 'm', 'a_f', 'q_f'
        temp : float
            Temperature in Kelvin
        strain_rate : float
            Reference strain rate in 1/s

        Returns
        -------
            sigma_diffusion : float

        """
        R = 8.314472  # m2kg/s2/K/mol
        d = material['d'][0]
        m = material['m'][0]
        a_f = material['a_f'][0]
        q_f = material['q_f'][0]
        return d**m*strain_rate/a_f*np.exp(q_f/R/temp)

    @staticmethod
    def sigma_dislocation(material, temp, strain_rate):
        """Compute differential stress for dislocation creep.

        Compute differential stress envelope for dislocation creep at
        certain temeprature and strain rate. Requires preexponential scaling
        factor 'a_p', power law exponent 'n' and activation energy 'q_p'.

        Parameters
        ----------
        material : dict
            Dictionary with the material properties in SI units. Required
            keys are 'a_p', 'n' and 'q_p'
        temp : float
            Temperature in Kelvin
        strain_rate : float
            Reference strain rate in 1/s

        Returns
        -------
            sigma_d : float

        """
        R = 8.314472  # m2kg/s2/K/mol
        a_p = material['a_p'][0]
        n = material['n'][0]
        q_p = material['q_p'][0]
        return (strain_rate/a_p)**(1.0/n)*np.exp(q_p/n/R/temp)

    @staticmethod
    def sigma_dorn(material, temp, strain_rate):
        """Compute differential stress for Dorn's law.

        Compute differential stress for solid state creep with Dorn's law.
        Requires Dorn's law stress 'sigma_d', Dorn's law activation energy
        'q_d' and Dorn's law strain rate 'A_p'.

        Dorn's creep is a special case of Peierl's creep with q=2

        sigma_delta = sigma_d*(1-(-R*T/Q*ln(strain_rate/A_d))^(1/q))

        Parameters
        ----------
        material : dict
            Dictionary with the material properties in SI units. Required
            keys are 'sigma_d', 'q_d' and 'A_p'
        temp : float
            Temperature in Kelvin
        strain_rate : float
            Reference strain rate in 1/s

        Returns
        -------
            sigma_d : float

        """
        R = 8.314472  # m2kg/s2/K/mol
        sigma_d = material['sigma_d'][0]
        q_d = material['q_d'][0]
        a_d = material['a_d'][0]
        if q_d == 0 or a_d == 0:
            return np.nan
        dorn = sigma_d*(1.0 - np.sqrt(-1.0*R*temp/q_d*np.log(strain_rate/a_d)))
        if dorn < 0.0:
            dorn = 0
        return dorn

    def sigma_d(self, material, z, temp, strain_rate=None,
                compute=None, mode=None, output=None):
        """Compute the differential stress for a given material.

        Computes differential stress for a material at given depth, temperature
        and strain rate. Returns the minimum of Byerlee's law, dislocation
        creep or dorn's creep.

        Parameters
        ----------
        material : dict
            Dict containing material properties required by sigma_byerlee() and
            sigma_dislocation()
        z : float
            Positive depth im m below surface
        temp : float
            Temperature in K
        strain_rate : float
            Reference strain rate in 1/s. If `None` will use self.strain_rate
        compute : list
            List of processes to compute: 'dislocation', 'diffusion', 'dorn'.
            Default is ['dislocation', 'dorn'].
        mode : str
            'compression' or 'extension'
        output : list of str
            Decide which diff. stress output should be retrieved. Can be either
            of `compute`.

        Returns
        -------
        out : dict
            Dictionary with the computed parameters. If `output` is `None`,
            will only contain key `dsigma_max`. Parameters of `output` will be
            added as keys to out.

        """
        # TODO: looping through each datapoint is inefficient. This should
        #  be implemented to work with numpy arrays
        if z < 0:
            raise ValueError('Depth must be positive. Got z =', z)
        if strain_rate is None:
            e_prime = self.strain_rate
        else:
            e_prime = strain_rate

        compute_processes = ['dislocation', 'diffusion', 'dorn']
        if compute is None:
            compute = ['dislocation', 'dorn']
        else:
            for kwd in compute:
                if kwd not in compute_processes:
                    raise ValueError('Unknown compute keyword', kwd)

        s_byerlee = self.sigma_byerlee(material, z, mode)

        if 'diffusion' in compute:
            s_diff = self.sigma_diffusion(material, temp, e_prime)
        else:
            s_diff = np.nan

        if 'dislocation' in compute:
            s_disloc = self.sigma_dislocation(material, temp, e_prime)
        else:
            s_disloc = np.nan
        if 'dorn' in compute:
            s_dorn = self.sigma_dorn(material, temp, e_prime)
        else:
            s_dorn = np.nan

        if (s_disloc > 200e6) and (s_dorn > 0):
            s_creep = s_dorn
        else:
            s_creep = s_disloc

        min_sigma = min([s_byerlee, s_creep, s_diff])
        factor = 1
        if mode == 'compression':
            factor = -1

        out = dict()
        out['dsigma_max'] = min_sigma*factor

        if output is not None:
            if 'byerlee' in output:
                out['byerlee'] = s_byerlee*factor
            if 'dislocation' in output:
                val = self.sigma_dislocation(material, temp, e_prime)
                out['dislocation'] = val*factor
            if 'dorn' in output:
                out['dorn'] = self.sigma_dorn(material, temp, e_prime)*factor
            if 'diffusion' in output:
                out['diffusion'] = s_diff*factor

        return out

    def compute_integrated_strength(self, nz=1000, spacing=None, force=False,
                                    mode='compression', strain_rate=None):
        """Compute the integrated strength of the model.

        Computes the integrated strength of the model and stores it as a numpy
        array with columns x, y, z in `self.integrated_strength`.

        Parameters
        ----------
        nz : int
            Number of vertical points used for the integration.
        spacing : float, optional
            The desired point spacing. Will use the original coordinates if
            `None`.
        force : bool
            If `True`, will compute the integrated strength regardless whether
            it was already computed before.
        mode : str
            Set to `compression` or `extension`.
        strain_rate : float, optional
            The strain rate in 1/s to apply. If `None` will use the one stored
            in `self.strain`.

        """
        if self.integrated_strength is not None and force is False:
            print('Already computed. Use `force=True` to re-compute.')
            return
        self._v_('Computing integrated strength', 0)

        if spacing is None:
            x_coords = np.unique(self.data_raw[:, 0])
            y_coords = np.unique(self.data_raw[:, 1])
        else:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            x_coords = np.arange(xmin, xmax+spacing, spacing)
            y_coords = np.arange(ymin, ymax+spacing, spacing)
        x_points, y_points = np.meshgrid(x_coords, y_coords)
        x_points = x_points.flatten()
        y_points = y_points.flatten()

        int_str = []
        if self.verbose_level >= 0:
            i = show_progress()
        i_max = x_points.shape[0]
        for x, y in zip(x_points, y_points):
            well = self.get_well(x, y, var='T')
            results = self.compute_yse(well, mode=mode,
                                       strain_rate=strain_rate, nz=nz,
                                       return_params=['integrate'])
            int_str.append(results['dsigma_int'])
            if self.verbose_level >= 0:
                i = show_progress(i, i_max)

        out = np.empty([x_points.shape[0], 3])
        out[:, 0] = x_points
        out[:, 1] = y_points
        out[:, 2] = np.asarray(int_str)
        self.integrated_strength = out

    def compute_surface_heat_flow(self, return_tcond=False, spacing=None,
                                  force=False, min_thickness=1.0):
        """Compute the surface heat flow of the model.

        This method computes the surface heat flow of the model. The surface
        heat flow is defined as

        .. math::
            Q_s = -\\lambda\\nabla T

        pyGMS computes the geothermal gradient :math:`\\nabla T` at the
        uppermost layer and uses the thermal conductivity :math:`\\lambda` of
        the first layer from the top that has a thickness greater than
        ``min_thickness``.

        Parameters
        ----------
        return_tcond : bool
            Return the thermal conductivities.
        spacing : float, optional
            Use custom lateral spacing for sampling.
        force : bool
            If `True`, will compute the surface heat flow regardless whether
            it was computed before or not.
        min_thickness : float
            The minimum thickness of a layer in meters to be regarded when
            computing the surface heat flow.

        Returns
        -------
        tconds : list
            A list of thermal conductivities.

        """
        if self.surface_heat_flow is not None and force is False:
            print('Already computed. Use `force=True` to re-compute.')
            return
        self._v_('Computing surface heat flow', 0)
        self.layer_add_var('Tcond')
        if spacing is None:
            x_coords = np.unique(self.data_raw[:, 0])
            y_coords = np.unique(self.data_raw[:, 1])
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
        if self.verbose_level >= 0:
            i = show_progress()
        i_max = x_points.shape[0]
        for x, y in zip(x_points, y_points):
            well = self.get_well(x, y, 'T')  # type: Well
            grad = well.grad('T', interp=1000)[0][0]
            # Get the first layer that has a specified thickness
            lay_id = self.get_uppermost_layer(x, y, tmin=min_thickness)
            # Obtain the thermal conductivity at the point
            t_cond = self.layers[lay_id](x, y, 'Tcond')
            if return_tcond:
                tconds.append(t_cond)
            hflow.append(-t_cond*grad)
            if self.verbose_level >= 0:
                i = show_progress(i, i_max)
        self.surface_heat_flow = np.empty([x_points.shape[0], 3])
        self.surface_heat_flow[:, 0] = x_points
        self.surface_heat_flow[:, 1] = y_points
        self.surface_heat_flow[:, 2] = np.asarray(hflow)
        if return_tcond:
            return tconds

    def compute_bd_thickness(self, dx=None, nz=500, mode='compression',
                             strain_rate=None):
        """Compute the brittle and ductile thicknesses of all layers.

        Compute the ductile thicknesses of each layer and store them in
        self.t_ductile and self.t_brittle as
        `[xpoints, ypoints, dict_of_thicknesses]`. The keys of
        `dict_of_thicknesses` are the unique layer ids.

        Parameters
        ----------
        dx : float, optional
            Horizontal resolution of the output. Will use the original
            coordinates if set to `None`.
        nz : int
            Number of vertical points.
        mode : str
            `compression` or `extension`.
        strain_rate : float, optional
            Strain rate in 1/s. Will use self.strain_rate if None.

        """
        return_params = ['t_brittle', 't_ductile']

        if dx is None:
            xpoints = self.data_raw[:, 0]
            ypoints = self.data_raw[:, 1]
            dx = xpoints[1] - xpoints[0]
        else:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            pointsx = np.arange(xmin, xmax+dx, dx)
            pointsy = np.arange(ymin, ymax+dx, dx)
            xgrid, ygrid = np.meshgrid(pointsx, pointsy)
            xpoints = xgrid.flatten()
            ypoints = ygrid.flatten()

        t_brittle = OrderedDict()
        t_ductile = OrderedDict()
        for uid in self.layer_dict_unique:
            t_brittle[uid] = []
            t_ductile[uid] = []

        self._v_(('Computing brittle/ductile thicknesses'), 0)
        self._v_(('> Strain rate:', strain_rate or self.strain_rate, '1/s'), 0)
        self._v_(('> Resolution :', dx, 'm'), 0)
        self._v_(('> nz         :', nz))

        if self.verbose_level >= 0:
            p = show_progress()
        pmax = xpoints.shape[0]
        for x, y in zip(xpoints, ypoints):
            w = self.get_well(x, y, 'T')
            w_r = self.compute_yse(w, mode=mode, nz=nz,
                                   strain_rate=strain_rate,
                                   return_params=return_params)
            w_t_brittle = w_r['t_brittle']
            w_t_ductile = w_r['t_ductile']
            for t_list, t_val in zip(t_brittle.values(), w_t_brittle.values()):
                t_list.append(t_val)
            for t_list, t_val in zip(t_ductile.values(), w_t_ductile.values()):
                t_list.append(t_val)
            if self.verbose_level >= 0:
                p = show_progress(p, pmax)

        # Convert lists to numpy arrays
        for layer_id in t_brittle:
            list_brittle = t_brittle[layer_id]
            list_ductile = t_ductile[layer_id]
            t_brittle[layer_id] = np.asarray(list_brittle)
            t_ductile[layer_id] = np.asarray(list_ductile)

        self.t_ductile = [xpoints, ypoints, t_ductile]
        self.t_brittle = [xpoints, ypoints, t_brittle]

    def compute_elastic_thickness(self, dx=None, nz=500, mode='compression',
                                  strain_rate=None, crit_sigma=20,
                                  crit_plitho=0.05):
        """Compute the effective elastic thickness of the model.

        Compute the effect elastic thickness after Burov and Diament (1995).
        First, the mechanical thickness is computed. It is defined as the depth
        from the layer top to the point where the differential stress is less
        than 1% - 5% of lithostatic pressure or it is below 10 - 20 MPa.

        Parameters
        ----------
        dx : float
            Horizontal resolution of the output
        nz : int
            Number of points used to compute Te at each point
        mode : str
            'compression' or 'extension'
        strain_rate : float, optional
            If None will use self.strain_rate
        crist_sigma : float
            Diff. stress criterion in MPa
        crit_plitho : float
            Lower limit for lithostatic pressure

        """
        strain_rate = strain_rate or self.strain_rate
        if dx is None:
            xpoints = self.data_raw[:, 0]
            ypoints = self.data_raw[:, 1]
        else:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            pointsx = np.arange(xmin, xmax+dx, dx)
            pointsy = np.arange(ymin, ymax+dx, dx)
            xgrid, ygrid = np.meshgrid(pointsx, pointsy)
            xpoints = xgrid.flatten()
            ypoints = ygrid.flatten()
        eff_Te = []
        competent_layers = []
        print('Computing elastic thickness')
        print('> Mode                      :', mode)
        print('> Strain rate               :', strain_rate, '1/s')
        print('> Horizontal resolution     :', dx, 'm')
        print('> Number of vertical points :', nz)
        print('> Min. sigma criterion      :', crit_sigma, 'MPa')
        print('> Lithostatic pressure crit.:', crit_plitho*100, '% of Plitho')
        if self.verbose_level >= 0:
            n = show_progress()
            nmax = xpoints.shape[0]
        yse_return = ['is_competent']
        for x, y in zip(xpoints, ypoints):
            well = self.get_well(x, y, var='T')
            result = self.compute_yse(well, mode, nz, strain_rate, crit_plitho,
                                      crit_sigma, return_params=yse_return)
            competent_layers.append(result['n_layers'])
            eff_Te.append(result['Te'])
            if self.verbose_level >= 0:
                n = show_progress(n, nmax)
        result = np.empty([xpoints.shape[0], 3])
        result[:, 0] = xpoints
        result[:, 1] = ypoints
        result[:, 2] = np.asarray(eff_Te)
        self.elastic_thickness[strain_rate] = [competent_layers, result]

    def compute_yse(self, well, mode='compression', nz=500, strain_rate=None,
                    crit_plitho=0.05, crit_sigma=20.0, return_params=None,
                    compute=None):
        """Compute yield strength envelopes and associated variables.

        Compute the yield strength envelope for a specific mode at a well
        instance. Also computes the effect elastic thickness after Burov and
        Diament (1995): First, the mechanical thickness is computed. It is
        defined as the depth from the layer top to the point where the
        differential stress is less than 1-5% of lithostatic pressure or where
        the it is below 10-20 MPa.

        Parameters
        ----------
        well : Well
            A Well instance.
        mode : str
            `compression` or `extension`.
        nz : int
            The number of sampling points in depth.
        strain_rate : float, optional
            Strain rate in 1/s. If `None`, will use strain rate provided with
            `set_rheology()`.
        crit_plitho : float
            Lithostatic pressure criterion betwen 0 and 1.
        crit_sigma : float
            Lower limit for mechanical thickness of absolute differential
            stress.
        return_params : list, optional
            A list of parameters that should be returned. See `Notes` for the
            full description.
        compute : list of str, optional
            List of processes to be computed. Is directly passed to
            :meth:`.GMS.sigma_d`.

        Returns
        -------
        results : dict
            Dictionary the containing keys
                - ``'dsigma_max'`` : the yield strength envelope
                - ``'layer_ids'``  : the corresponding layer ids
                - ``'z'``          : the depth values

        Notes
        -----
        Valid values for ``return_params``:

            * ``'integrate'``
                `dsigma_int`, the integrated strength in Pa m
            * ``'is_competent'``
                Returns multiple parameters: the effective elastic thickness
                ``'Te'``, the number of decoupled layers ``'n_layers'``, a bool
                array where layers in the YSE are competent ``'is_competent'``,
                and the tops and bottoms of the competent layers
                ``'competent_depths'``.
            * ``'diffusion'``
                differential stress for diffusion at all depths.
            * ``'dislocation'``
                differential stress for dislocation at all depths.
            * ``'dorn'``
                differential stress for Dorn's law at all depths.
            * ``'P_litho'``
                Lithostatic pressure.
            * ``'t_brittle'``
                Brittle thickness of each layer.
            * ``'t_ductile'``
                Ductile thickness of each layer.

        """
        results = dict()
        output = []
        return_params = return_params or []
        return_vals = ['is_competent', 'P_litho', 't_ductile', 't_brittle',
                       'integrate']
        output_vals = ['byerlee', 'dislocation', 'diffusion', 'dorn']
        for param in return_params:
            if param in output_vals:
                output.append(param)
                results[param] = []
            elif param in return_vals:
                continue
            else:
                msg = 'Unknown return parameter', param
                raise ValueError(msg)
        output = None if len(output) == 0 else output
        strain_rate = strain_rate or self.strain_rate

        zs, T, lay_ids = well.get_interpolated_var(nz, 'T')
        ztopo = zs[0]
        dsigma_max = []
        P_litho = [0]
        z_prev = ztopo
        for z, lay_id, temp in zip(zs, lay_ids, T):
            mat_name = self.body_materials[lay_id]
            material = self.materials_db[self.materials_db['name'] == mat_name]
            T_K = temp + 273.15
            sigma_d = self.sigma_d(material=material, z=ztopo-z, temp=T_K,
                                   strain_rate=strain_rate, mode=mode,
                                   compute=compute, output=output)
            dsigma_max.append(sigma_d['dsigma_max'])    # Unit: Pa
            incr_thickness = z_prev - z
            if incr_thickness > 0:
                dPlitho = material['rho_b'][0]*9.81*incr_thickness
                P_litho.append(P_litho[-1] + dPlitho)
            z_prev = z
            if output is not None:
                for param in output:
                    results[param].append(sigma_d[param])

        dsigma_max = np.asarray(dsigma_max)
        if output is not None:
            for param in output:
                results[param] = np.asarray(results[param])

        if 't_ductile' in return_params or 't_brittle' in return_params:
            grad_dsigma = np.gradient(np.abs(dsigma_max), -z)  # (-) = ductile
            well_unique_ids = np.asarray([self.layer_map[i] for i in lay_ids])
            delta_t = zs[0] - zs[1]
            t_ductile = OrderedDict()
            t_brittle = OrderedDict()
            for lay_id in self.layer_dict_unique.keys():
                idx_lay_id = well_unique_ids == lay_id
                n_ductile = np.sum(grad_dsigma[idx_lay_id] < 0)
                n_brittle = np.sum(grad_dsigma[idx_lay_id] >= 0)
                t_ductile[lay_id] = n_ductile*delta_t
                t_brittle[lay_id] = n_brittle*delta_t
            if 't_ductile' in return_params:
                results['t_ductile'] = t_ductile
            if 't_brittle' in return_params:
                results['t_brittle'] = t_brittle

        if 'integrate' in return_params:
            results['dsigma_int'] = np.trapz(dsigma_max, zs)

        if 'P_litho' in return_params:
            results['P_litho'] = P_litho

        if 'is_competent' in return_params:
            # Compute the mechanical thickness of each layer
            strength = np.abs(np.asarray(dsigma_max))
            P_mech = np.asarray(P_litho)*crit_plitho
            # Use simplification by Burov and Diament (1995), p. 3915: get a
            # bool array, where layers are mechanical with respect to PLitho
            is_competent_P = strength > P_mech
            # .. and with respect to absolute Sigma
            is_competent_S = strength > crit_sigma*1e6
            is_competent = np.logical_and(is_competent_P, is_competent_S)

            # Detect the individual mechanical thicknesses
            z_layer_top = ztopo
            h_mechs = []
            competent_depths = []
            wait_for_next_layer = False
            delta_z = zs[0] - zs[1]
            for i in range(is_competent.shape[0]):
                is_weak = is_competent[i] == False
                was_strong_before = is_competent[i-1] == True
                if i+1 == is_competent.shape[0]:
                    is_continuous = True
                else:
                    # Check whether the following point is also not competent
                    # required because gradient can be false at layer bounds
                    is_continuous = is_competent[i+1] == False
                if wait_for_next_layer and not is_weak:
                    wait_for_next_layer = False
                    z_layer_top = zs[i]
                if is_weak & is_continuous & was_strong_before:
                    h = z_layer_top - zs[i]
                    # Do not add if only one point is competent
                    if h >= 2*delta_z:
                        h_mechs.append(h)
                        competent_depths.extend([z_layer_top, zs[i]])
                    wait_for_next_layer = True

            if len(h_mechs) > 1:
                # In case of multiple h_mechs, i.e. when decoupled layers exist
                h_mech = 0
                for h in h_mechs:
                    h_mech += h**3
                h_mech = h_mech**(1.0/3.0)
            elif len(h_mechs) == 1:
                h_mech = h_mechs[0]
            else:
                h_mech = ztopo - well.z[-1]
                h_mechs = [h_mech]
                competent_depths = [ztopo, well.z[-1]]
            results['Te'] = h_mech
            results['n_layers'] = len(h_mechs)
            results['is_competent'] = is_competent
            results['competent_depths'] = competent_depths

        results['layer_ids'] = lay_ids
        results['dsigma_max'] = dsigma_max
        results['z'] = zs

        return results

    def get_uppermost_layer(self, x, y, tmin=1.0):
        """Get the layer id of the uppermost layer.

        Returns the layer id of a layer at the specified coordinates that has
        a thickness greater than ``'tmin'``.

        Parameters
        ----------
        x : float
            x-coordainte.
        y : float
            y-coordinate.
        tmin : float
            Minimum thickness in m.

        """
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
        """Get a virtual well at the specified coordinates.

        .. todo::

            Raise error when outside of model bounds! This should actually
            happen in the Layer class.

        By default returns a Well instance at the specified coordinates. If
        var is given, additionally the value of the variable will be returned.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        var : str, optional
            Name of the variable that should be attached to the well.
        store : bool
            If `True` will store the well object in self.wells.

        Returns
        -------
        well : Well

        """
        if (x, y, var) not in self.wells:
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
                    msg = 'Obtaining ' + str(var) + ' for layer'
                    self._v_((msg, self.layer_dict[i]))
                    values.append(layer(x, y, var))
                    i += 1
                well.add_var(var, np.asarray(values))
            if store:
                self.wells[(x, y, var)] = well
            return well
        else:
            return self.wells[(x, y, var)]

    def read_header(self):
        """Read the headers of the GMS file."""
        self._v_('Reading headers')
        gms_fem_f = open(self.gms_fem, 'r')
        i = 0            # Line counter
        n_layers = 0
        unique_id = 0
        layer_d = OrderedDict()
        layer_d_unique = OrderedDict()
        layer_map = OrderedDict()
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
                    unique_id = layer_id
                    layer_d_unique[layer_id] = layer_name
                layer_map[layer_id] = unique_id
            # Stop if header ends
            if not l.startswith('#'):
                break
            i += 1
        gms_fem_f.close()
        self.layer_dict = layer_d
        self.layer_dict_unique = layer_d_unique
        self.layer_map = layer_map
        self.field_dict = field_d
        self.n_layers = n_layers
        self.n_layers_unique = len(list(layer_d_unique.keys()))
        self.points_per_layer = self.nx*self.ny
        self._v_('Number of layers:' + str(n_layers))

    def load_model(self, filename=None):
        """Load a GMS model.

        Loads a GMS model specified by the argument. If `filename=None` will
        load the file specified in `self.gms_fem`.

        Parameters
        ----------
            filename : str
                The path and file name of the GMS model.

        """
        filename = filename or self.gms_fem
        if filename is None:
            self._v_("No GMS file name specified.")
        else:
            print('Loading', self.gms_fem)
            self.read_header()
            self.load_points()
            print('Done!')

    def load_points(self):
        """Load the data points of the GMS model."""
        self.data_raw = np.loadtxt(self.gms_fem)
        self.n_points = self.data_raw.shape[0]

    def make_layers(self):
        """Create the layers of the GMS model.

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
        """Add a variable to a layer.

        Loads a variable from the GMS model into the pyGMS model.

        Parameters
        ----------
        varname : str
            The name of the variable inside the GMS file.

        """
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
        """Plot a surface heat flow map.

        Create a plot of the surface heat flow. Will also compute the surface
        heat flow if it hasn't been computed yet.

        Parameters
        ----------
        kwds : dict
            Keywords that are passed to `matplotlib.pyplot.tricontourf`.

        Returns
        -------

        """
        if self.surface_heat_flow is None:
            self.compute_surface_heat_flow()
        ax = kwds['ax'] or plt.axes()
        if 'ax' in kwds:
            kwds.pop('ax')
        x = self.surface_heat_flow[:, 0]
        y = self.surface_heat_flow[:, 1]
        hflow = self.surface_heat_flow[:, 2]*1000
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        return ax.tricontourf(x, y, hflow, **kwds)

    def plot_layer_bounds(self, x0, y0, x1, y1, num=100, lc='black', lw=1,
                          unit='km', only='unique', ax=None, xaxis='dist',
                          cmap=None, fill_kwds=None, **kwds):
        """Plot the layer tops in profile view.

        Plot the layer tops as lines along the given profile. By default only
        plots unqiue layers, i.e. not the refined layers. Use `only_unique` to
        change this behaviour.

        Parameters
        ----------
        x0, y0, x1, y1 : float
            Start and end point of the profile.
        num : int
            Number of sampling points along the profile.
        lc : str
            Line colours.
        lw : float
            Line width.
        unit : str
            Distance units in 'm' or 'km'.
        only : str, int, list
            Define which layers should be plotted. `unique` does not plot the
            refinements, `all` plots all layers incl. refinements, stating an
            int or list of int will plot the layers with the given ids.
        ax : matplotlib.axes, optional
            The axes to plot into.
        xaxis : str
            The coordinate to show along the horizontal axis in the plot. Can
            be 'dist' (starts at 0), 'x' or 'y'.
        lay_id : None, int or list
            If stated will only plot layers with the given id(s).
        cmap : colormap, optional
            If given will fill the bodys.
        ** kwds
            Keywords going to matplotlib.pyplot.plot

        """
        length_units = {'km': 1e-3, 'm': 1}
        _ = length_units[unit]
        len_fmt = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*_))

        if only == 'unique':
            layer_d = self.layer_dict_unique
        elif only == 'all':
            layer_d = self.layer_dict
        elif isinstance(only, int):
            layer_d = {only: self.layer_dict[only]}
        elif isinstance(only, list):
            layer_d = {}
            for i in only:
                layer_d[i] = self.layer_dict[i]
        else:
            layer_d = self.layer_dict

        ax = ax or plt.axes()
        ax.yaxis.set_major_formatter(len_fmt)
        ax.xaxis.set_major_formatter(len_fmt)

        px, py, dist = self._points_and_dist_(x0, y0, x1, y1, num, scale=1)
        if xaxis == 'dist':
            d = dist
        elif xaxis == 'x':
            d = px
        elif xaxis == 'y':
            d = py
        else:
            raise ValueError('Unknown xaxis', xaxis)

        if cmap is not None:
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            zmin = self.zlim[0]
            if fill_kwds is None:
                fill_kwds = dict()

        j = 0
        for i in layer_d.keys():
            layer = self.layers[i]  # type: Layer
            z = []
            for x, y in zip(px, py):
                z.append(layer(x, y))
            z = np.asarray(z)
            if cmap is not None:
                color = cmap.colors[j]
                zmin = []
                if i < list(layer_d.keys())[-1]:
                    next_layer = self.layers[list(layer_d.keys())[j+1]]
                    for x, y in zip(px, py):
                        zmin.append(next_layer(x, y))
                else:
                    zmin = self.zlim[0]
                ax.fill_between(d, z, zmin, facecolor=color, linewidth=0,
                                **fill_kwds)
                j += 1
            ax.plot(d, z, color=lc, lw=lw, **kwds)

    def plot_strength_profile(self, x0, y0, x1, y1, num=1000, num_vert=100,
                              mode='compression', compute=None,
                              strain_rate=None, force=False,
                              show_competent=False, competent_kwds=None,
                              crit_plitho=0.05, crit_sigma=20,
                              ax=None, xaxis='dist', unit='km', dsigma='Pa',
                              levels=50, cmap=None):
        """Compute and plot an arbitrary yield strength profile.

        Compute the strength for the given strain rate along an arbitrary
        profile.

        Parameters
        ----------
        x0, y0 : float
            Start point coordinates / m
        x1, y1 : float
            End point coordinates / m
        num : int
            Number of sampling points in horizontal direction
        num_vert : int
            Number of sampling points in vertical direction
        mode : str
            'compression' or 'extension'
        compute : list of str, optional
        strain_rate : float, optional
        force : bool
            If 'True` will force the new computation of the strength
        show_competent : {bool, 'only'}, optional
            Define plotting of the tops and bases of the competent layers
            according to the `crit_plitho` and `crit_sigma`. ``True``: Plot
            them additionally to the YSE profile. ``'only'``: Do not plot the
            strength values, only the competent boundary.
        competent_kwds : dict, optional
            Keywords that go straight to `ax.scatter()`
        crit_plitho : float
            Lithostatic pressure criterion betwen 0 and 1.
        crit_sigma : float
           Lower limit for mechanical thickness of absolute differential stress
        ax : matplotlib.axes
            Axes object to plot onto. If `None` will use matplotlib.pyplot
        xaxis : {`dist`, `x`, `y`}
            Which coordinates to display along the xaxis. `dist` for distance
            along the profile, `x`, or `y`.
        unit : {`km`, `m`}
            Unit of length, `km` or `m`
        dsigma : {`GPa`, `MPa`, `Pa`}
            Unit of differential stress `GPa`, `MPa`, `Pa`
        levels : int, list, np.array
            If `int` defines the number of levels for the contour plot,
            if a 1D numpy array will use these levels as contour levels.
        cmap : str, matplotlih colormap, optional
            Colormap for differential stress plot. Will either load the
            colormap with the given name or use the one handed over.

        Returns
        -------
        mappable

        """
        length_units = {'km': 1e-3, 'm': 1}
        _ = length_units[unit]
        len_fmt = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*_))

        if dsigma == 'GPa':
            dsigma_scale = 1e-9
        elif dsigma == 'MPa':
            dsigma_scale = 1e-6
        elif dsigma == 'Pa':
            dsigma_scale = 1
        else:
            msg = 'Unknown unit for dsigma:', dsigma
            raise ValueError(msg)
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        else:
            cmap = cmap or plt.cm.get_cmap('viridis')
        if mode == 'compression':
            cmap = mpl.colors.ListedColormap(cmap.colors[::-1])

        ax = ax or plt.axes()
        ax.yaxis.set_major_formatter(len_fmt)
        ax.xaxis.set_major_formatter(len_fmt)

        # Make the points where to sample the model
        px, py, dist = self._points_and_dist_(x0, y0, x1, y1, num, scale=1)
        if xaxis == 'dist':
            d = dist
        elif xaxis == 'x':
            d = px
        elif xaxis == 'y':
            d = py
        else:
            raise ValueError('Unknown xaxis', xaxis)
        d = d.tolist()

        if strain_rate is None:
            strain_rate = self.strain_rate

        return_params = []
        only_competent = False
        competent_x = None
        competent_y = None
        competent_kwds = competent_kwds or dict()
        if show_competent:
            if show_competent == 'only':
                only_competent = True
            # force = True
            competent_y = []
            competent_x = []
            return_params.append('is_competent')

        # Check whether Temperatures are loaded
        var = 'T'
        if var not in list(self.layers[0].interpolators.keys()):
            self.layer_add_var(var)

        # Check if stored
        # TODO: This must be handled better, is missing rheology and
        #  coordinate system!
        profile_stats = (x0, y0, x1, y1, strain_rate, show_competent,
                         dsigma_scale)
        if profile_stats in self._strength_profiles.keys() and not force:
            props = self._strength_profiles[profile_stats]
            t, strength, competent_x, competent_y, dsigma_scale = props
        else:
            kwds_yse = {'mode': mode,
                        'nz': num_vert,
                        'strain_rate': strain_rate,
                        'crit_plitho': crit_plitho,
                        'crit_sigma': crit_sigma,
                        'return_params': return_params,
                        'compute': compute}
            # Obtain the wells
            depths = []
            strength = []
            profile_distance = []
            self._v_('Computing strength', 0)
            self._v_(('Strength scale:', dsigma_scale), 1)
            i = show_progress()
            imax = len(px)
            for x, y, p_dist in zip(px, py, d):
                profile_distance.extend([p_dist]*num_vert)
                well = self.get_well(x, y, var)  # type: Well
                results = self.compute_yse(well, **kwds_yse)
                dsigma_scaled = results['dsigma_max']*dsigma_scale
                strength.extend(dsigma_scaled.tolist())
                depths.extend(results['z'].tolist())
                if show_competent:
                    c_depths = results['competent_depths']
                    competent_y.extend(c_depths)
                    competent_x.extend([p_dist]*len(c_depths))
                i = show_progress(i, imax)
            t = tri.Triangulation(x=np.asarray(profile_distance),
                                  y=np.asarray(depths))
            self._strength_profiles[profile_stats] = (t, strength, competent_x,
                                                      competent_y,
                                                      dsigma_scale)

        self._v_(('Computed strength:', min(strength), ' -', max(strength)), 1)
        if isinstance(levels, int):
            levels = np.linspace(min(strength), max(strength), levels)

        if not only_competent:
            obj = ax.tricontourf(t, strength, levels=levels, cmap=cmap)
        else:
            obj = None

        if show_competent:
            competent_x = np.asarray(competent_x)
            competent_y = np.asarray(competent_y)
            kwds = {'c': 'white', 's': 1}
            if 'color' in competent_kwds:
                kwds['c'] = competent_kwds['color']
                competent_kwds.pop('color')
            if competent_kwds is not None:
                for key in competent_kwds.keys():
                    kwds[key] = competent_kwds[key]
            ax.scatter(competent_x, competent_y, **kwds)

        return obj

    def plot_topography(self, ax=None):
        """Plot the topography.

        Plots the topography either to the given axis or in its own plot.

        Parameters
        ----------
        ax : matplotlib.axis
            The axis to plot into.

        Returns
        -------
        matplotlib.tricontourf object

        """
        if ax is None:
            ax = plt.axes()
            ax.set_aspect('equal')
        zmin = np.abs(self.layers[0].z.min())
        zmax = np.abs(self.layers[0].z.max())
        vmin = -1*max(zmin, zmax)
        vmax = max(zmin, zmax)
        return ax.tricontourf(self.layers[0].triangulation, self.layers[0].z,
                              cmap=afrikakarte(), vmin=vmin, vmax=vmax)

    def plot_profile(self, x0, y0, x1, y1, var='T', num=100, ax=None,
                     unit='m', kind='filled', xaxis='dist', annotate=False,
                     **kwds):
        """Plot a profile.

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
        unit : {'m', 'km'}
            Defines the unit of length.
        kind : str
            'filled' for a filled contour plot or 'lines', 'contours' for
            contours
        xaxis : str
            The dimension to show along the x-axis. 'dist' for distance
            where 0km is at (x0, y0), 'x' or 'y' for the respective axis.
        annotate : bool
            If `True` will annotate the contour lines.
        kwds : dict
           Keywords sent to matplotlib.pyplot.tricontour() or
           matplotlib.pyplot.contourf(), depoending on `kind`

        Returns
        -------
        mappable

        """
        valid_kinds = ['filled', 'lines', 'contour']
        if kind not in valid_kinds:
            raise ValueError('Invalid type', kind)

        length_units = {'km': 1e-3, 'm': 1}
        _ = length_units[unit]
        len_fmt = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*_))

        ax = ax or plt.axes()
        ax.yaxis.set_major_formatter(len_fmt)
        ax.xaxis.set_major_formatter(len_fmt)

        layer_d = self.layer_dict

        # Make the points where to sample the model
        px, py, dist = self._points_and_dist_(x0, y0, x1, y1, num, scale=1)
        if xaxis == 'dist':
            d = dist
        elif xaxis == 'x':
            d = px
        elif xaxis == 'y':
            d = py
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
        z = np.asarray(z)
        v = np.asarray(v)

        # Make a triangulation
        t = tri.Triangulation(d*self.n_layers, z)

        # Print the contours
        if kind == 'filled':
            obj = ax.tricontourf(t, v, **kwds)
        elif kind == 'lines' or kind == 'contours':
            obj = ax.tricontour(t, v, **kwds)
            if annotate:
                ax.clabel(obj, colors='black')
        return obj

    def plot_yse(self, loc, strain_rate=None, mode='compression', nz=500,
                 crit_plitho=0.05, crit_sigma=20.0, title=None, ax=None,
                 strength_unit='GPa', depth_unit='km', plot_bodies=False,
                 body_cmap=None, body_names=None, body_col_width=None,
                 fill_mode='envelope', label_competent='Competent layer',
                 label_envelope=None, leg_kwds=None, compute=None,
                 plot_all_sigma=False, scale_axes=True, plot_kwds=None,
                 kwds_fill=None, kwds_fig=None, return_params=None, **kwds):
        """Plot a yield strength envelope of a well or an x,y coordinate.

        .. todo::

            Fix plotting of YSE for extensive regimes.

        Plot a yield strength envelope for the given mode at the specified
        location `loc`, which can either be a `Well` instance or a x, y
        coordinate. Computes the envelope on the fly and offers ability to plot
        competent layers, output effective elastic thickness or the lithology.

        Parameters
        ----------
        loc: Well, tuple, list
            Either a well instance, (x, y) or [x, y].
        strain_rate : float
            Strain rate in 1/s. If `None`, will use strain rate provided with
            `set_rheology()`.
        mode : str
            `compression` or `extension`.
        nz : int
            The number of sampling points in depth.
        crit_plitho : float
            Lithostatic pressure criterion betwen 0 and 1.
        crit_sigma : float
           Lower limit for mechanical thickness of absolute differential stress
        title : str, bool, optional
            The title of the plot, if None will plot the x,y coordinates. To
            deactivate use `False`.
        ax : matplotlib.axes instance, optional
            The axis to plot into.
        strength_unit : {`GPa`, `MPa`, `Pa`}
            The display unit of strength `GPa`, `MPa` or `Pa`.
        depth_unit : {`km`, `m`}
            The display unit of depths `km` or `m`.
        plot_bodies : bool
            If `True` will plot the lithologies next to the YSE.
        body_cmap : str, matplotlib.colors.ListedColormap, optional
            Either the name of a matplotlib colormap or a `ListedColormap`
            instance. Will use `Set2` by default.
        body_names : list of str, Bool, optional
            List of strings for the bodies in the model from top to bottom.
            Needs to include all bodies, not only the ones that will appear in
            the profile. If `None` the integrated names will be printed. If
            `False`, will not plot any body names.
        body_col_width : float, optional
            The width of the body column in MPa.
        fill_mode : str, optional
            `envelope` or `box` to mark the competent layers according to the
            chosen `crit_sigma` and `crit_plitho`. `envelope` fills the area
            between the YSE and 0 Pa, `box` creates boxes.
        lambel_competent : str, optional
            The label that should be used to mark the filled area.
        label_envelope : str
            Label for the line of the envelope, optional.
        leg_kwds : dict
            Keywords that will be passed to `ax.legend()`.
        compute : list of str, optional
            The processes which should be considered when computing the brittle
            YSE. Can be `diffusion`, `dislocation` and `dorn`. By default will
            use `[dislocation', 'dorn']`.
        plot_all_sigma : bool
            Will plot all diff. stresses for all processes defined in
            `compute`.
        plot_kwds : dict, optional
            Keywords that will be sent to matplotlib.axes.plot()
        return_params : list of str
            'Te' for elastic thickness.

        Keyword arguments
        -----------------
        show_Te : bool
            Display the computed effective elastic thickness.
        show_title : bool
            Show the title or not.

        """
        FuncFormatter = mpl.ticker.FuncFormatter
        show_title = True
        show_legend = True
        show_Te = True
        left_lim = 0
        right_lim = 0
        strength_units = {'GPa': 1e-9, 'MPa': 1e-6}
        depth_units = {'km': 1e-3, 'm': 1}
        yse_return_params = ['is_competent']
        return_params = return_params or []
        plot_processes = None

        # if isinstance(loc, Well): <-- this didnt work
        if type(loc).__name__ == 'Well':
            well = loc
            x = well.x
            y = well.y
        elif isinstance(loc, tuple) or isinstance(loc, list):
            x, y = loc
            well = self.get_well(x, y, var='T')
        else:
            msg = 'Unknown location', loc
            raise ValueError(msg)
        if plot_all_sigma:
            yse_return_params.append('byerlee')
            plot_processes = ['byerlee']
            if compute is None:
                yse_return_params.extend(['dislocation', 'dorn'])
            else:
                plot_processes.extend(compute)
                yse_return_params.extend(compute)
        if 'show_title' in kwds:
            show_title = kwds['show_title']
        if title is False:
            show_title = False
        if 'show_legend' in kwds:
            show_legend = kwds['show_legend']
        if 'show_Te' in kwds:
            show_Te = kwds['show_Te']
        if plot_kwds is None:
            plot_kwds = dict()
        if kwds_fill is None:
            kwds_fill = dict()
        if kwds_fig is None:
            kwds_fig = dict()

        ymax = well.z[0]
        ymin = well.z[-1]
        results = self.compute_yse(well, mode, nz, strain_rate, crit_plitho,
                                   crit_sigma, compute=compute,
                                   return_params=yse_return_params)
        strength = results['dsigma_max']
        strength_z = results['z']
        is_competent = results['is_competent']
        eff_Te = results['Te']

        x_fill = None
        if fill_mode == 'envelope':
            x_fill = np.ma.masked_where(np.invert(is_competent), strength)
        elif fill_mode == 'box':
            max_fill_x = 10*np.ones_like(strength)*np.abs(strength).max()
            if mode == 'compression':
                max_fill_x *= -1
            x_fill = np.ma.masked_where(np.invert(is_competent), max_fill_x)

        ax_new = False
        if ax is None:
            ax_new = True
            if 'figsize' not in kwds_fig:
                kwds_fig['figsize'] = (3, 3)
            if 'dpi' not in kwds_fig:
                kwds_fig['dpi'] = 150
            fig = plt.figure(**kwds_fig)
        ax = ax or plt.axes()

        if 'solid_joinstyle' not in list(plot_kwds.keys()):
            plot_kwds['solid_joinstyle'] = 'miter'

        ax.plot(strength, strength_z, label=label_envelope, **plot_kwds)

        if x_fill is not None:
            kwds_fill_def = {'linewidth': 0,
                             'alpha': 0.2,
                             'label': label_competent}
            for kwd in kwds_fill_def:
                if kwd not in kwds_fill:
                    kwds_fill[kwd] = kwds_fill_def[kwd]
            ax.fill_betweenx(strength_z, x_fill, 0, **kwds_fill)

        if plot_all_sigma:
            for process in plot_processes:
                _val = results[process]
                ax.plot(_val, strength_z, label=process, lw=1, ls='--')

        if plot_bodies:
            if body_cmap is None:
                body_cmap = plt.get_cmap('Set2')
            if body_col_width:
                column_width = body_col_width
            else:
                column_width = 0.2*(strength.max() - strength.min())
            layer_ids = results['layer_ids']
            unique_ids = list(self.layer_dict_unique.keys())
            if body_names is None:
                body_names = [self.layer_dict_unique[i] for i in unique_ids]
            elif body_names is False:
                body_names = [None for i in unique_ids]
            for i in range(len(unique_ids) - 1):
                this_id = unique_ids[i]
                next_id = unique_ids[i + 1]
                condition = (layer_ids >= this_id) & (layer_ids < next_id)
                if not condition.any():
                    # skip if this body doesn't exist in the plot
                    continue
                xcoords = np.zeros_like(layer_ids, dtype=float)
                xcoords[condition] += column_width
                body_color = body_cmap.colors[i]
                ax.fill_betweenx(strength_z, xcoords, 0, linewidth=0.5,
                                 facecolor=body_color, edgecolor='black',
                                 label=body_names[i], zorder=-1)
            if mode == 'compression':
                right_lim = column_width
            else:
                left_lim = column_width

        if label_competent or label_envelope or plot_bodies:
            if leg_kwds is None:
                leg_kwds = dict()
            if show_legend:
                ax.legend(**leg_kwds)

        if scale_axes:
            du = depth_units[depth_unit]
            y_fmt = FuncFormatter(lambda y, pos: '{0:g}'.format(y*du))
            ax.yaxis.set_major_formatter(y_fmt)

            su = strength_units[strength_unit]
            x_fmt = FuncFormatter(lambda x, pos: '{0:g}'.format(x*su))
            ax.xaxis.set_major_formatter(x_fmt)

        if mode == 'compression':
            left_lim = 1.1*strength.min()
        else:
            right_lim = 1.1*strength.max()

        ax.set_xlim(left=left_lim, right=right_lim)
        ax.set_ylim(ymin, ymax)

        if show_Te:
            ax.annotate('Te = '+str(np.round(eff_Te/1000, 1))+' km', xy=(1, 0),
                        xycoords='axes fraction', horizontalalignment='right',
                        verticalalignment='bottom')

        if show_title:
            title = title or 'x = '+str(x)+', y = '+str(y)
            ax.set_title(title)

        if ax_new:
            ax.set_xlabel('$\Delta\sigma_{max}$ / MPa')
            ax.set_ylabel('Elevation / km')
            # fig.show()

        if 'Te' in return_params:
            return eff_Te

    def set_rheology(self, strain_rate=None, rheologies=None, bodies=None):
        """Define the rheological parameters.

        Define the rheological parameters of the model. Sets the strain rate,
        the material properties, and assigns materials to the bodies.

        Parameters
        ----------
        strain_rate : float
            The strain rate in 1/s.

        rheologies : list
            List of dictionaries with rock properties. The following
            properties must be assigned as keys:

            * name    : the name of the material
            * f_f_c   : friction coefficient for compression
            * f_f_e   : friction coefficient for extension
            * f_p     : pore fluid factor
            * rho_b   : bulk density
            * a_p     : pre-exponential factor
            * n       : power law exponent
            * q_p     : activation energy
            * sigma_d : Dorn's law stress
            * q_d     : Dorn's law activation energy
            * a_d     : Dorn's law strain rate

            If a property should not be used, assign `None`.

        bodies : dict
            Dictionary with {layer_name: material}: where material is the name
            of the material in the `rheologies` dictionary, and layer_name
            is the **unique** layer name, i.e. the layer names in
            self.layer_dict_unique.

            bodies = {'LayerName1':'diorite_dry', 'LayerName2:'olivine_dry'}

        """
        #  Assign strain rate and body materials
        if strain_rate:
            self.strain_rate = strain_rate
        if bodies:
            self.body_materials = OrderedDict()
            # Assign the materials to the layers
            for layer_name in bodies.keys():
                material = bodies[layer_name]
                i = 0
                for id in self.layer_dict.keys():
                    name = self.layer_dict[id].split('_')[0]
                    if name == layer_name:
                        self.body_materials[i] = material
                    i += 1
        if rheologies:
            # Define the material properties
            n_rocks = len(rheologies)
            dtypes = [('name', object),
                      # Beyerlees properties
                      ('f_f_c', float), ('f_f_e', float), ('f_p', float),
                      ('rho_b', float),
                      # Dislocation creep parameter
                      ('a_p', float), ('n', float), ('q_p', float),
                      # Diffusion creep parameters
                      ('a_f', float), ('q_f', float), ('d', float),
                      ('m', float),
                      # Dorns law parameters
                      ('sigma_d', float), ('q_d', float), ('a_d', float),
                      # Metadata
                      ('source', object), ('via', object), ('altname', object)]
            # Make a list of props for later usages
            props = []
            for d in dtypes:
                props.append(d[0])
            self.materials_db = np.zeros([n_rocks], dtype=dtypes)
            i = 0
            # Store all materials, not only those in_body_rheology
            for entry in rheologies:
                for key in entry.keys():
                    if key in props:
                        self.materials_db[key][i] = entry[key]
                i += 1


if __name__ == '__main__':
    pass
