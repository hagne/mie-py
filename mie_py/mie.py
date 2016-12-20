import numpy as np
import pandas as pd
import matplotlib.pylab as plt


def get_logDeriv(size_parameter, refractive_index, no_of_termses):
    """ Logarithmic derivative D(J) calculated by downward recurrence
        beginning with initial value (0.,0.) at J=NMX

    """
    y = size_parameter * refractive_index
    nn = int(no_of_termses[1]) - 1
    d = np.zeros(nn + 1, dtype=np.complex128)
    for n in range(0, nn):
        en = no_of_termses[1] - n
        d[nn - n - 1] = (en / y) - (1. / (d[nn - n] + en / y))
    return d

def get_no_of_terms(size_parameter, refractive_index):
    """Original comment:
    Series expansion terminated after NSTOP (noOfTerms) terms
        Logarithmic derivatives calculated from NMX on down
     BTD experiment 91/1/15: add one more term to series and compare resu<s
          NMX=AMAX1(XSTOP,YMOD)+16
     test: compute 7001 wavelen>hs between .0001 and 1000 micron
     for a=1.0micron SiC grain.  When NMX increased by 1, only a single
     computed number changed (out of 4*7001) and it only changed by 1/8387
     conclusion: we are indeed retaining enough terms in series!
     """

    ymod = abs(size_parameter * refractive_index)
    xstop = size_parameter + 4. * size_parameter**0.3333 + 10
    nmx = max(xstop, ymod) + 15.0
    nmx = np.fix(nmx)

    # Hagen: now idea what this limit is for?!?
    nmxx = 150000
    if (nmx > nmxx):
        raise ValueError("error: nmx > nmxx=%f for |m|x=%f" % (nmxx, ymod))
    return (int(xstop), nmx)

class DataFramePolar(pd.DataFrame):
    def plot(self, **kwargs):
        f,a = plt.subplots(subplot_kw=dict(projection='polar'))
        super().plot(ax = a, **kwargs)
        a.set_yscale('log')
        # a.set_ylim(top=self.values.max())
        a.grid()
        return a

class Parameters(object):
    def __init__(self, parent):
        self._parent = parent
        self._size_parameter = 5
        self._diameter = None
        self._no_of_angles = 1000
        self._refractive_index = 1.5
        self._wavelength = None
        self._something_changed = True

    def __repr__(self):
        repr = ['size parameter:   {}'.format(self.size_parameter),
                'diameter:         {}'.format(self.diameter),
                'wavelength:       {}'.format(self.wavelength),
                'refractive index: {}'.format(self.refractive_index),
                'no of angles:     {}'.format(self.no_of_angles),
                ]
        return '\n'.join(repr)

    @property
    def no_of_angles(self):
        return self._no_of_angles

    @no_of_angles.setter
    def no_of_angles(self,value):
        self._something_changed = True
        self._no_of_angles = value

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self,value):
        self._something_changed = True
        self._refractive_index = value

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self,value):
        self._something_changed = True
        self._diameter = value
        try:
            self._size_parameter = np.pi * self._diameter / self.wavelength
        except:
            self._size_parameter = None

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._something_changed = True
        self._wavelength = value
        self._size_parameter = np.pi * self.diameter / self._wavelength
        try:
            self._size_parameter = np.pi * self.diameter / self._wavelength
        except:
            self._size_parameter = None

    @property
    def size_parameter(self):
        return self._size_parameter

    @size_parameter.setter
    def size_parameter(self, value):
        self._something_changed = True
        self._size_parameter = value
        self._wavelength = None
        self._diameter = None


    def _bind_to(self, callback):
        self._observers.append(callback)

class Bohren_Huffman():
    def __init__(self):#, x, refrel, noOfAngles, diameter=False):
        self.parameters = Parameters(self)
        # self.parameters.diameter = diameter
        # self.parameters.no_of_angles = noOfAngles
        # self.parameters.size_parameter = x
        # self.parameters.refractive_index = refrel
        self._normalizer = (4 * np.pi) ** 2  # hagen: the physical origin is not clear to me right now, but this normalizer
                                            # is necessary so the integral of the scattering function is equal to the
                                            # scattering crossection and the integral over the phase function is 4 pi
        self._extinction_crosssection = None
        # self._run_mie()
        self._angular_scatt_func = None
        self._phase_func = None

    ##################################################
    def _update(self):
        if self.parameters._something_changed:
            self._run_mie()
            self._angular_scatt_func = None
            self._phase_func = None

            self.parameters._something_changed = False

    @property
    def phase_func(self):
        self._update()
        if not np.any(self._phase_func):
            self._phase_func = self._get_phase_func()
        return self._phase_func

    @property
    def angular_scatt_func(self):
        self._update()
        if not np.any(self._angular_scatt_func):
            self._angular_scatt_func = self._get_angular_scatt_func()
        return DataFramePolar(self._angular_scatt_func)


    @property
    def asymmetry_parameter(self):
        self._update()
        return self._asymmetry_parameter

    @property
    def backscatter_efficiency(self):
        self._update()
        return self._backscatter_efficiency

    @property
    def extinction_crosssection(self):
        self._update()
        return self._extinction_crosssection

    @property
    def extinction_efficiency(self):
        self._update()
        return self._extinction_efficiency

    @property
    def scattering_crosssection(self):
        self._update()
        return self._scattering_crosssection

    @property
    def scattering_efficiency(self):
        self._update()
        return self._scattering_efficiency

    # @property
    # def extinction_crosssection(self):
    #     self._update()
    #     return self._extinction_crosssection

    ###################################################

    def _run_mie(self):
        s1_1 = np.zeros(self.parameters.no_of_angles, dtype=np.complex128)
        s1_2 = np.zeros(self.parameters.no_of_angles, dtype=np.complex128)
        s2_1 = np.zeros(self.parameters.no_of_angles, dtype=np.complex128)
        s2_2 = np.zeros(self.parameters.no_of_angles, dtype=np.complex128)

        # if (self.parameters.no_of_angles > 1000):
        #     print('error: self.noOfAngles > mxself.noOfAngles=1000 in bhmie')
        #     return

        # Require NANG>1 in order to calculate scattering intensities
        if (self.parameters.no_of_angles < 2):
            self.parameters.no_of_angles = 2



        pii = 4. * np.arctan(1.)
        dang = .5 * pii / (self.parameters.no_of_angles - 1)
        amu = np.arange(0.0, self.parameters.no_of_angles, 1)
        amu = np.cos(amu * dang)
        pi0 = np.zeros(self.parameters.no_of_angles, dtype=np.complex128)
        pi1 = np.ones(self.parameters.no_of_angles, dtype=np.complex128)



        # Riccati-Bessel functions with real argument X
        # calculated by upward recurrence

        psi0 = np.cos(self.parameters.size_parameter)
        psi1 = np.sin(self.parameters.size_parameter)
        chi0 = -np.sin(self.parameters.size_parameter)
        chi1 = np.cos(self.parameters.size_parameter)
        xi1 = psi1 - chi1 * 1j
        qsca = 0.
        gsca = 0.
        p = -1

        no_of_termses = get_no_of_terms(self.parameters.size_parameter, self.parameters.refractive_index)
        logDeriv = get_logDeriv(self.parameters.size_parameter, self.parameters.refractive_index, no_of_termses)
        for n in range(0, no_of_termses[0]):
            en = n + 1.0
            fn = (2. * en + 1.) / (en * (en + 1.))

            # for given N, PSI  = psi_n        CHI  = chi_n
            #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
            #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
            # Calculate psi_n and chi_n
            psi = (2. * en - 1.) * psi1 / self.parameters.size_parameter - psi0
            chi = (2. * en - 1.) * chi1 / self.parameters.size_parameter - chi0
            xi = psi - chi * 1j
            # Store previous values of AN and BN for use
            # in computation of g=<cos(theta)>
            if (n > 0):
                an1 = an
                bn1 = bn

            '''
            These are the key parameters for the Mie calculations, an and bn,
            used to comute the amplitudes of the scattering field.
            '''
            an = (logDeriv[n] / self.parameters.refractive_index + en / self.parameters.size_parameter) * psi - psi1
            an /= ((logDeriv[n] / self.parameters.refractive_index + en / self.parameters.size_parameter) * xi - xi1)
            bn = (self.parameters.refractive_index * logDeriv[n] + en / self.parameters.size_parameter) * psi - psi1
            bn /= ((self.parameters.refractive_index * logDeriv[n] + en / self.parameters.size_parameter) * xi - xi1)

            # *** Augment sums for Qsca and g=<cos(theta)>
            qsca += (2. * en + 1.) * (abs(an) ** 2 + abs(bn) ** 2)
            gsca += ((2. * en + 1.) / (en * (en + 1.))) * (np.real(an) * np.real(bn) + np.imag(an) * np.imag(bn))

            if (n > 0):
                gsca += ((en - 1.) * (en + 1.) / en) * (
                np.real(an1) * np.real(an) + np.imag(an1) * np.imag(an) + np.real(bn1) * np.real(bn) + np.imag(
                    bn1) * np.imag(bn))


                # *** Now calculate scattering intensity pattern
                #    First do angles from 0 to 90
            pi = pi1.copy()
            tau = en * amu * pi - (en + 1.) * pi0
            s1_1 += fn * (an * pi + bn * tau)
            s2_1 += fn * (an * tau + bn * pi)
            # *** Now do angles greater than 90 using PI and TAU from
            #    angles less than 90.
            #    P=1 for N=1,3,...% P=-1 for N=2,4,...
            #   remember that we have to reverse the order of the elements
            #   of the second part of s1 and s2 after the calculation
            p = -p
            s1_2 += fn * p * (an * pi - bn * tau)
            s2_2 += fn * p * (bn * pi - an * tau)

            psi0 = psi1
            psi1 = psi
            chi0 = chi1
            chi1 = chi
            xi1 = psi1 - chi1 * 1j

            # *** Compute pi_n for next value of n
            #    For each angle J, compute pi_n+1
            #    from PI = pi_n , PI0 = pi_n-1
            pi1 = ((2. * en + 1.) * amu * pi - (en + 1.) * pi0) / en
            pi0 = pi.copy()

        # *** Have summed sufficient terms.
        #    Now compute QSCA,QEXT,QBACK,and GSCA

        #   we have to reverse the order of the elements of the second part of s1 and s2
        s1 = np.concatenate((s1_1, s1_2[-2::-1]))
        s2 = np.concatenate((s2_1, s2_2[-2::-1]))
        gsca = 2. * gsca / qsca
        qsca = (2. / (self.parameters.size_parameter ** 2)) * qsca
        #        qext = (4./ (self.sizeParameter**2))* real(s1[0])

        # more common definition of the backscattering efficiency,
        # so that the backscattering cross section really
        # has dimension of length squared
        #        qback = 4*(abs(s1[2*self.noOfAngles-2])/self.sizeParameter)**2
        # qback = ((abs(s1[2*self.noOfAngles-2])/self.sizeParameter)**2 )/pii  #old form
        self._s1 = s1
        self._s2 = s2
        #        self.qext = qext
        self._calc_qext()
        self._scattering_efficiency = qsca
        #        self.qback = qback
        self._calc_qback()
        self._asymmetry_parameter = gsca
        if self.parameters.diameter:
            self._scattering_crosssection = self._scattering_efficiency * self.parameters.diameter ** 2 * np.pi * 0.5 ** 2  # scattering crosssection
        else:
            self._scattering_crosssection = 0



    # def _get_logDeriv(self):
    #     """ Logarithmic derivative D(J) calculated by downward recurrence
    #         beginning with initial value (0.,0.) at J=NMX
    #
    #     """
    #     y = self.parameters.size_parameter * self.parameters.refractive_index
    #     nn = int(self._no_of_termses[1]) - 1
    #     d = np.zeros(nn + 1, dtype=np.complex128)
    #     for n in range(0, nn):
    #         en = self._no_of_termses[1] - n
    #         d[nn - n - 1] = (en / y) - (1. / (d[nn - n] + en / y))
    #     return d

    def _get_natural(self):
        return np.abs(self._s1) ** 2 + np.abs(self._s2) ** 2

    def _get_perpendicular(self):
        return np.abs(self._s1) ** 2

    def _get_parallel(self):
        return np.abs(self._s2) ** 2

    # def _calc_noOfTerms(self):
    #     """Original comment:
    #     Series expansion terminated after NSTOP (noOfTerms) terms
    #         Logarithmic derivatives calculated from NMX on down
    #      BTD experiment 91/1/15: add one more term to series and compare resu<s
    #           NMX=AMAX1(XSTOP,YMOD)+16
    #      test: compute 7001 wavelen>hs between .0001 and 1000 micron
    #      for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    #      computed number changed (out of 4*7001) and it only changed by 1/8387
    #      conclusion: we are indeed retaining enough terms in series!
    #      """
    #
    #     ymod = abs(self.parameters.size_parameter * self.parameters.refractive_index)
    #     xstop = self.parameters.size_parameter + 4. * self.parameters.size_parameter ** 0.3333 + 2.0
    #     # xstop = x + 4.*x**0.3333 + 10.0
    #     nmx = max(xstop, ymod) + 15.0
    #     nmx = np.fix(nmx)
    #
    #     self._no_of_termses = (int(xstop), nmx)
    #     # print('noOfTerm')
    #     # Hagen: now idea what this limit is for?!?
    #     nmxx = 150000
    #     if (nmx > nmxx):
    #         raise ValueError("error: nmx > nmxx=%f for |m|x=%f" % (nmxx, ymod))
    #     return self._no_of_termses


    def _calc_qext(self):
        """extinction efficiency. normalized real part of s1 at 0 deg (forward)"""
        self._extinction_efficiency = (4. / (self.parameters.size_parameter ** 2)) * np.real(self._s1[0])
        if self.parameters.diameter:
            self._extinction_crosssection = self._extinction_efficiency * self.parameters.diameter ** 2 * np.pi * 0.5 ** 2
        else:
            self._extinction_crosssection = 0

    def _calc_qback(self):
        """ Backscattering efficiency. Looks like it simlpy locks for the efficiency
        at 180 deg... I am surprised why they are not simpy taking the last one?
        -> it is the same!! -> fixed"""
        self._backscatter_efficiency = 4 * (abs(self._s1[-1]) / self.parameters.size_parameter) ** 2

    def _get_phase_func(self):
        """ Returns the phase functions in the interval [0,2*pi).

        Note
        ----
        The phase phase function is normalized such that the integrale over the entire sphere is 4pi
        """
        # out = self.get_angular_scatt_func() * 4 * np.pi/self.csca
        s2r = self._s2[::-1]
        s2f = np.append(self._s2, s2r[1:])
        s2s = np.abs(s2f) ** 2
        # ang = np.linspace(0, np.pi * 2, len(s2s))
        # df = pd.DataFrame(s2s, index=ang, columns=['Phase_function_parallel'])
        # df.index.name = 'Angle'

        s1r = self._s1[::-1]
        s1f = np.append(self._s1, s1r[1:])
        s1s = np.abs(s1f) ** 2

        s12s = (s1s + s2s) / 2

        ang = np.linspace(0, np.pi * 2, len(s1s))
        df = pd.DataFrame(np.array([s1s, s2s, s12s]).transpose(), index=ang,
                          columns=['perpendicular', 'parallel', 'natural'])
        df.index.name = 'angle'
        df *= 4 * np.pi / (np.pi * self.parameters.size_parameter ** 2 * self.scattering_efficiency)
        return DataFramePolar(df)

    def _get_angular_scatt_func(self):
        """
        Returns the angular scattering function for scattering geometry in the interval [0,2*pi).

        Note
        ----
        The integral of 'natural' over the entire sqhere is equal to the scattering crossection.
        >>> natural = natural[theta < np.pi] # to ensure integration from 0 to pi
        >>> theta = theta[theta < np.pi]
        >>> integrate.simps(natural * np.sin(theta) ,theta) * 2 * np.pi # this is equal to scattering crossection
        """

        df = self.phase_func.copy()
        df *= self.scattering_crosssection / (4 * np.pi)
        return df

    # # todo: deprecated
    # def return_Values_as_dict(self):
    #     return {  # 'phaseFct_S1': self.s1,
    #
    #         'extinction_efficiency': self.extinction_efficiency,
    #         'scattering_efficiency': self.scattering_efficiency,
    #         'backscatter_efficiency': self.backscatter_efficiency,
    #         'asymmetry_parameter': self._asymmetry_parameter,
    #         'scattering_crosssection': self.scattering_crosssection,
    #         'extinction_crosssection': self.extinction_crosssection}
    #
    # # todo: deprecated
    # def return_Values(self):
    #     return self._s1, self._s2, self.extinction_efficiency, self.scattering_efficiency, self.backscatter_efficiency, self._asymmetry_parameter

if __name__ == "__main__":
    #    x = 10
    x_sizePara = 5
    n_refraction = 1.5 + 0.01j
    nang_no = 10
    bhh = bhmie(x_sizePara, n_refraction, nang_no)
    s1, s2, qext, qsca, qback, gsca = bhh.return_Values()

    s1, s2, qext, qsca, qback, gsca = bhmie(x_sizePara, n_refraction, nang_no)

# def test_I():


def test_extinction_coeff():
    wl = .55
    d = .1
    ref = 1.455
    sp = lambda wl, d: 2 * np.pi / wl * d / 2
    mie_I = bhmie(sp(wl, d), ref, 100, diameter=d)
    mo_I = mie_I.return_Values_as_dict()

    wl = .55
    d = .1
    ref = 1.1
    sp = lambda wl, d: 2 * np.pi / wl * d / 2
    mie_II = bhmie(sp(wl, d), ref, 100, diameter=d)
    mo_II = mie_II.return_Values_as_dict()

    wl = .55
    d = .1
    ref = 4.
    sp = lambda wl, d: 2 * np.pi / wl * d / 2
    mie = bhmie(sp(wl, d), ref, 100, diameter=d)
    mo_III = mie.return_Values_as_dict()

    test_I_is = mo_II['extinction_crosssection'] / mo_I['extinction_crosssection']
    # test_I_is = mie.
    test_I_should = 0.0527297452683
    test_II_is = mo_III['extinction_crosssection'] / mo_I['extinction_crosssection']
    test_II_should = 14.3981634837

    print('test value 1 is/should be/diff: %s/%s/%s' % (test_I_is, test_I_should, test_I_is - test_I_should))
    print('test value 2 is/should be/diff: %s/%s/%s' % (test_II_is, test_II_should, test_II_is - test_II_should))

    passed = False
    if abs(test_I_is - test_I_should) < 1e-4:
        if abs(test_II_is - test_II_should) < 1e-4:
            passed = True

    print('----------\ntest result:', '')
    if passed:
        print('passed')
    else:
        print('failed')
    print('----------')


