import unittest
import numpy as np
from matplotlib import tri
import sys
sys.path.append('../')
from pyGMS import GMS
from pyGMS.structures import Well
from pyGMS.structures import Layer
from pandas import DataFrame


def test_material():
    dtypes = [
        ('name', object),
        # Beyerlees properties
        ('f_f_c', float), ('f_f_e', float), ('f_p', float), ('rho_b', float),
        # Dislocation creep parameter
        ('a_p', float), ('n', float), ('q_p', float),
        # Diffusion creep parameters
        ('a_f', float), ('q_f', float), ('d', float), ('m', float),
        # Dorns law parameters
        ('sigma_d', float), ('q_d', float), ('a_d', float),
        # Metadata
        ('source', object), ('via', object), ('altname', object)
    ]
    mat = np.zeros([1], dtype=dtypes)
    mat['name'][0] = 'test_material'
    # Byerlee properties
    mat['f_f_e'][0] = 0.75
    mat['f_f_c'][0] = 2.0
    mat['f_p'][0] = 0.35
    mat['rho_b'][0] = 2440.0
    # Dislocation creep
    mat['a_p'][0] = 1e-28
    mat['n'][0] = 4.0
    mat['q_p'][0] = 223e3
    # Diffusion creep
    mat['a_f'][0] = 2.570e-11
    mat['q_f'][0] = 300e3
    mat['d'][0] = 0.1e-3
    mat['m'][0] = 2.5
    # Dorn's law
    mat['sigma_d'][0] = 8.5e9
    mat['q_d'][0] = 535e3
    mat['a_d'][0] = 5.754e11
    return mat

test_material_dict = dict(
    name='test_material',
    f_f_e=0.75,
    f_f_c=2.0,
    f_p=0.35,
    rho_b=2440.0,
    a_p=1e-28,
    n=4.0,
    q_p=223e3,
    a_f=2.570e-11,
    q_f=300e3,
    d=0.1e-3,
    m=2.5,
    sigma_d=8.5e9,
    q_d=535e3,
    a_d=5.754e11,
)
depth = 10e3
temp = 1000.0
test_modes = ['compression', 'extension']
strain_rate = 1e-15
testmodel = 'tests/test.fem'
bodies = {
    'Sediments': 'test_material',
    'Crust': 'test_material',
    'Mantle': 'test_material',
    'Base': 'test_material'
}


class TestLayer(unittest.TestCase):

    def test_call_fail(self):
        L = Layer([0, 1, 2], [1, 2, 0], [0, 0, 0], 0)
        with self.assertRaises(AttributeError):
            L(0, 0, 'T')


class TestWell(unittest.TestCase):

    def test_call(self):
        m = GMS(testmodel, verbosity=-1)
        w = m.get_well(0, 0 )
        self.assertListEqual(w().tolist(), [0., -10e3, -40e3, -50e3])

    def test_getitem(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        w = m.get_well(0, 0, 'T')
        self.assertListEqual(w['T'].tolist(), [0., 200., 800., 1000.])

    def test_plot_grad(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        w = m.get_well(0, 0, 'T')
        for grad in w.plot_grad('T', return_array=True, absolute=True):
            self.assertAlmostEqual(grad, 0.02)


class TestGMS(unittest.TestCase):

    def test_call_well(self):
        m = GMS(testmodel, verbosity=-1)
        self.assertIsInstance(m(50e3, 50e3), Well)

    def test_call_depth(self):
        m = GMS(testmodel, verbosity=-1)
        result = m(50e3, 50e3, -5e3)
        target = [0, 'Sediments']
        self.assertListEqual(result, target)

    def test_getitem_laye_int(self):
        m = GMS(testmodel, verbosity=-1)
        self.assertIsInstance(m[0], Layer)

    def test_getitem_layer_str(self):
        m = GMS(testmodel, verbosity=-1)
        self.assertIsInstance(m['Sediments'], Layer)

    def test_getitem_layer_str_fail(self):
        m = GMS(testmodel, verbosity=-1)
        with self.assertRaises(ValueError):
            m['fail']

    def test_get_xlim(self):
        m = GMS(testmodel, verbosity=-1)
        self.assertTupleEqual(m.xlim, (0., 100e3))

    def test_get_ylim(self):
        m = GMS(testmodel, verbosity=-1)
        self.assertTupleEqual(m.ylim, (0., 100e3))

    def test_get_zlim(self):
        m = GMS(testmodel, verbosity=-1)
        self.assertTupleEqual(m.zlim, (-50e3, 0.))

    def test_info_df(self):
        m = GMS(testmodel, verbosity=-1)
        m.info
        self.assertIsInstance(m._info_df, DataFrame)


class TestPlot(unittest.TestCase):

    def test_plot_topography(self):
        m = GMS(testmodel, verbosity=-1)
        t = m.plot_topography()
        self.assertIsInstance(t, tri.TriContourSet)
        levels = t.levels.tolist()
        levels_target = [-2.4999999999999994e-14, 0.0, 2.4999999999999994e-14]
        self.assertListEqual(levels, levels_target)

    def test_plot_yse(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.set_rheology(strain_rate=strain_rate,
                       rheologies=[test_material_dict],
                       bodies=bodies)
        te = m.plot_yse((50e3, 50e3), return_params=['Te'])
        te_target = 21442.885771543086
        self.assertAlmostEqual(te, te_target)

    def test_plot_profile(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        obj = m.plot_profile(0, 0, 100e3, 100e3, num=10, annotate=True)
        self.assertIsInstance(obj, tri.TriContourSet)

    def test_plot_strength_profile(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.set_rheology(strain_rate=strain_rate,
                       rheologies=[test_material_dict],
                       bodies=bodies)
        obj = m.plot_strength_profile(0, 0, 100e3, 100e3, num=10,
                                      show_competent=True)
        self.assertIsInstance(obj, tri.TriContourSet)

class TestThermal(unittest.TestCase):

    def test_heat_flow(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.compute_surface_heat_flow()
        for val in m.surface_heat_flow[:, 2]:
            self.assertAlmostEqual(val, 0.042)


class TestRheology(unittest.TestCase):

    def test_sigma_byerlee_ext(self):
        m = GMS()
        material = test_material()
        mode = 'extension'
        target = 116689950.0
        result = m.sigma_byerlee(material, depth, mode)
        self.assertEqual(result, target)

    def test_sigma_byerlee_comp(self):
        m = GMS()
        material = test_material()
        mode = 'compression'
        target = 311173200.0,
        result = m.sigma_byerlee(material, depth, mode)
        self.assertEqual(result, target)

    def test_sigma_byerlee_fail(self):
        m = GMS()
        material = test_material()
        mode = 'c'
        with self.assertRaises(ValueError):
            m.sigma_byerlee(material, depth, mode)

    def test_sigma_diffusion(self):
        m = GMS()
        material = test_material()
        target = 18.20267567303978
        result = m.sigma_diffusion(material, temp, strain_rate)
        self.assertEqual(result, target)

    def test_sigma_dislocation(self):
        m = GMS()
        material = test_material()
        target = 1452181.9550299197
        result = m.sigma_dislocation(material, temp, strain_rate)
        self.assertEqual(result, target)

    def test_sigma_dorn_nan_q_d(self):
        m = GMS()
        material = test_material()
        material['q_d'][0] = 0
        target = np.nan
        result = m.sigma_dorn(material, temp, strain_rate)
        self.assertIs(result, target)

    def test_sigma_dorn_nan_a_d(self):
        m = GMS()
        material = test_material()
        material['a_d'][0] = 0
        target = np.nan
        result = m.sigma_dorn(material, temp, strain_rate)
        self.assertIs(result, target)

    def test_sigma_dorn_zero(self):
        m = GMS()
        material = test_material()
        temperature = 1500.0
        target = 0.0
        result = m.sigma_dorn(material, temperature, strain_rate)
        self.assertEqual(result, target)

    def test_sigma_dorn(self):
        m = GMS()
        material = test_material()
        target = 182170257.3512531
        result = m.sigma_dorn(material, temp, strain_rate)
        self.assertEqual(result, target)

    def test_sigma_d_ext(self):
        m = GMS()
        material = test_material()
        mode = 'extension'
        processes = ['dislocation', 'dorn']
        outputs = ['byerlee', 'dislocation', 'dorn']
        target = {
            'dsigma_max': 1452181.9550299197,
            'byerlee': 116689950.0,
            'dislocation': 1452181.9550299197,
            'dorn': 182170257.3512531
        }
        result = m.sigma_d(material, depth, temp,
                           strain_rate=strain_rate,
                           compute=processes,
                           mode=mode,
                           output=outputs)
        self.assertEqual(result, target)

    def test_sigma_d_comp(self):
        m = GMS()
        material = test_material()
        mode = 'compression'
        processes = ['dislocation', 'dorn']
        outputs = ['byerlee', 'dislocation', 'dorn']
        target = {
            'dsigma_max': -1452181.9550299197,
            'byerlee': -311173200.0,
            'dislocation': -1452181.9550299197,
            'dorn': -182170257.3512531
        }
        result = m.sigma_d(material, depth, temp,
                           strain_rate=strain_rate,
                           compute=processes,
                           mode=mode,
                           output=outputs)
        self.assertEqual(result, target)

    def test_sigma_d_fail_z(self):
        m = GMS()
        material = test_material()
        with self.assertRaises(ValueError):
            m.sigma_d(material, -100, temp)

    def test_integrated_strength_comp(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.set_rheology(strain_rate=strain_rate,
                       rheologies=[test_material_dict],
                       bodies=bodies)
        m.compute_integrated_strength(nz=10, mode='compression')
        target = 3674238431772.26
        for val in m.integrated_strength[:, 2]:
            self.assertAlmostEqual(val, target, places=1)

    def test_integrated_strength_ext(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.set_rheology(strain_rate=strain_rate,
                       rheologies=[test_material_dict],
                       bodies=bodies)
        m.compute_integrated_strength(nz=10, mode='extension')
        target = -1873467598438.9
        for val in m.integrated_strength[:, 2]:
            self.assertAlmostEqual(val, target, places=1)

    def test_compute_bd_thickness_comp(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.set_rheology(strain_rate=strain_rate,
                       rheologies=[test_material_dict],
                       bodies=bodies)
        m.compute_bd_thickness(mode='compression')
        t_duct_sediments = m.t_ductile[2][0]
        t_duct_crust = m.t_ductile[2][1]
        t_duct_mantle = m.t_ductile[2][2]
        target_sediments = 0.
        target_crust = 24949.
        target_mantle = 9919.
        for val in t_duct_sediments:
            self.assertAlmostEqual(val, target_sediments, -1)
        for val in t_duct_crust:
            self.assertAlmostEqual(val, target_crust, -1)
        for val in t_duct_mantle:
            self.assertAlmostEqual(val, target_mantle, -1)

        t_brit_sediments = m.t_brittle[2][0]
        t_brit_crust = m.t_brittle[2][1]
        t_brit_mantle = m.t_brittle[2][2]
        target_sediments = 10020.
        target_crust = 5110.
        target_mantle = 0.
        for val in t_brit_sediments:
            self.assertAlmostEqual(val, target_sediments, -1)
        for val in t_brit_crust:
            self.assertAlmostEqual(val, target_crust, -1)
        for val in t_brit_mantle:
            self.assertAlmostEqual(val, target_mantle, -1)

    def test_compute_elastic_thickness_comp(self):
        m = GMS(testmodel, verbosity=-1)
        m.layer_add_var('T')
        m.set_rheology(strain_rate=strain_rate,
                       rheologies=[test_material_dict],
                       bodies=bodies)
        m.compute_elastic_thickness()
        competent_layers, eet = m.elastic_thickness[strain_rate]
        target_competent_layers = 1
        target_eet = 21442.
        for val in competent_layers:
            self.assertEqual(val, target_competent_layers)
        for val in eet[:, 2]:
            self.assertAlmostEqual(val, target_eet, -1)

if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
