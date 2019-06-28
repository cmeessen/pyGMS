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
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from matplotlib import tri

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

        Adds a variable to the layer. Interpolated values of the variable can be
        accessed using the `interpolators[name]` variable of the layer.

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
    """Well class.

    This class represents a virtual well that can be extracted from a GMS model.

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
        self.zero_threshold = 0.1 # Minimum layer thickness

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
            #ip = interpolate.interp1d(self.z, self.vars[varname])
            #z = np.linspace(self.z.max(), self.z.min(), interp)
            #vars = ip(z)
            z, var_values, layer_ids = self.get_interpolated_var(interp, varname)
            return np.gradient(var_values, z), z
        else:
            z = self.z
            var_values = self.vars[varname]
            return np.gradient(var_values, z)

    def plot_var(self, varname, **kwds):
        """Plot a variable.

        Plots a variable over depth.

        Parameters
        ----------
        varname : str
            Name of the variable.
        kwds : dict, optional
            Keywords forwarded to `matplotlib.pyplot.plt()`.

        """
        kwds.setdefault('label', varname)
        plt.plot(self.vars[varname], self.z, **kwds)

    def plot_grad(self, varname, scale=1, return_array=False, abs=False):
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
        abs : bool
            Plot the absolute value of `varname`.

        Returns
        -------
        if `return_array = False`:
            Nothing
        else:
            The gradient array scaled by `scale`.

        """
        grad = self.grad(varname)
        if abs:
            grad = np.abs(grad)
        plt.plot(grad*scale, self.z, label='grad('+varname+')')
        if return_array:
            return grad*scale
