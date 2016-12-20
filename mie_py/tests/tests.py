from unittest import TestCase
import mie_py
import numpy as np
from scipy import integrate

class TestBohrenHuffman(TestCase):
    def test_compare2online_calculator(self, verbose=False, raiseError=True):
        """Test against the calculations from here: http://omlc.org/cgi-bin/mie_angles.cgi?diameter=1.0&lambda_vac=0.6328&n_medium=1.0&nr_sphere=1.5&ni_sphere=0&n_angles=100&density=0.1"""
        mc = mie_py.Bohren_Huffman()
        mc.parameters.diameter = 1
        mc.parameters.wavelength = 0.6328
        mc.parameters.refractive_index = 1.5
        #     noOfAngles = 200

        asymmetry_parameter = mc.asymmetry_parameter
        backscatter_efficiency = mc.backscatter_efficiency
        extinction_crosssection = mc.extinction_crosssection
        extinction_efficiency = mc.extinction_efficiency
        scattering_crosssection = mc.scattering_crosssection
        scattering_efficiency = mc.scattering_efficiency

        s_Size_Parameter = 4.9646
        s_asymmetry_parameter = 0.70765
        s_scattering_efficiency = 3.8962
        s_extinction_efficiency = 3.8962
        s_backscatter_efficiency = 1.9428
        s_scattering_crosssection = 3.0601  # micron2
        s_extinction_crosssection = 3.0601  # micron2
        # Backscattering Cross Section = 1.5259 #micron2
        # Scattering Coefficient = 306 #mm-1
        # Total Attenuation Coefficient = 306 #mm-1

        asymmetry_parameter_ratio = asymmetry_parameter / s_asymmetry_parameter
        backscatter_efficiency_ratio = backscatter_efficiency / s_backscatter_efficiency
        extinction_crosssection_ratio = extinction_crosssection / s_extinction_crosssection
        extinction_efficiency_ratio = extinction_efficiency / s_extinction_efficiency
        scattering_crosssection_ratio = scattering_crosssection / s_scattering_crosssection
        scattering_efficiency_ratio = scattering_efficiency / s_scattering_efficiency

        test_passed = True
        for i in [asymmetry_parameter_ratio,
                  backscatter_efficiency_ratio,
                  extinction_crosssection_ratio,
                  extinction_efficiency_ratio,
                  scattering_crosssection_ratio,
                  scattering_efficiency_ratio]:
            if verbose:
                print(i)
            if raiseError:
                self.assertTrue (abs(i - 1) < 1e-4)
            if abs(i - 1) > 1e-4:
                test_passed = False

        if verbose:
            print('---------')
            print('test', end=' ')
            if test_passed:
                print('passed')
            else:
                print('failed')
            print('---------')

        return test_passed

    def test_angular_scattering_func_integral(verbose=False, raiseError=True):
        """This test makes sure the integral over the the entire angular scattering function is equal (deviates less
         than 1e-4) to the overall scattering crossection"""

        mc = mie_py.Bohren_Huffman()
        mc.parameters.diameter = 10.
        mc.parameters.wavelength = 0.6328
        #     sp = lambda wl, d: 2 * np.pi / wl * d / 2
        #     sizeparam = sp(wavelength, diameter)
        mc.parameters.refractive_index = 1.5
        mc.parameters.no_of_angles = 2000

        #### TEST if integral is same as scattering
        asf = mc._get_angular_scatt_func()
        natural = asf.natural.values
        theta = asf.index.values
        natural = natural[theta < np.pi]  # to ensure integration from 0 to pi
        theta = theta[theta < np.pi]
        isf = integrate.simps(natural * np.sin(theta), theta) * 2 * np.pi
        if verbose:
            print(mc.scattering_crosssection / isf)
        passed = False

        test_threshold = 1e-4
        if raiseError:
            assert (((mc.scattering_crosssection / isf) - 1) < test_threshold)
        if abs((mc.scattering_crosssection / isf) - 1) > test_threshold:
            if verbose:
                print('integral test failed')
        else:
            passed = True
            if verbose:
                print('integral test passed')
        return passed
