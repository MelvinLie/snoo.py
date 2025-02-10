import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, PchipInterpolator
from scipy.optimize import minimize_scalar

class Permeability():

    def __init__(self, input_filename):
        '''Default constructor.
        
        :param input_filename:
            The name of the input file with the B(H) data

        :return:
            None.
        '''

        mat_data = pd.read_csv(input_filename)

        self.B_data = mat_data['B(T)'].values
        self.H_data = mat_data['H(A/m)'].values

        self.initialize_interpolation()

        return None

    def initialize_interpolation(self):
        '''Initialize the BSpline spline interpolation.
        
        :return:
            None
        '''

        # allocate mu array
        self.mu = np.zeros((len(self.H_data), ))

        # compute mu
        self.mu[1:] = self.B_data[1:]/self.H_data[1:]
        self.mu[0] = self.mu[1]

        # we fit mu(x) with x = log10(H)
        # the derivative of mu wrt H is
        # dmu_dH = dmu_dx / H / ln(10)
        self.x = np.zeros((len(self.H_data), ))
        self.x[1:] = np.log10(self.H_data[1:])
        self.x[0] = 2*self.x[1] - self.x[2]

        # interpolate
        self.spl = splrep(self.x, self.mu)

        # find the zero derivative in the first interval
        def obj_fcn(x):
            return abs(splev(x, self.spl, der=1))

        # this is the minimum x value we cut off here
        self.x_min = minimize_scalar(obj_fcn, bounds=[self.x[0], self.x[1]]).x

        # this is the corresponding H value
        self.H_min = 10.0**self.x_min

        x_hr = np.linspace(self.x_min, max(self.x), 1000)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x[1:], self.mu[1:], 'o', label='data')
        ax.plot(x_hr, splev(x_hr, self.spl), label='fit')
        ax.set_xlabel('$x$')
        ax.set_title(r'$\mu$ in Vs/Am')
        ax.legend()
        plt.show()

        return None

    def evaluate_mu(self, H_mag, relative=False):
        '''Compute the permeability given the field vector magnitude.
        
        :param H_mag:
            The field vector magnitudes.

        :param relative:
            Set this flag to true to return the relative permeability.

        :return:
            The permeability values.
        '''


        # here we avoid zeros
        H = H_mag.copy()
        H[H < self.H_min] = self.H_min

        # the derivative wrt x
        mu = splev(np.log10(H), self.spl)

        if relative:
            return mu/4/np.pi*1e7
        else:
            return mu

    def evaluate_mu_derivative(self, H_mag, relative=False):
        '''Compute the permeability derivative wrt H_mag, given the field vector magnitude.
        
        :param H_mag:
            The field vector magnitudes.

        :param relative:
            Set this flag to true to return the relative permeability.

        :return:
            The permeability derivative values.
        '''


        # here we avoid zeros
        H = H_mag.copy()
        H[H < self.H_min] = self.H_min

        d_mu_dx = splev(np.log10(H), self.spl, der=1)

        # the derivative w.r.t. H
        d_mu = d_mu_dx/H/np.log(10)

        if relative:
            return d_mu/4/np.pi*1e7
        else:
            return d_mu

    def plot_report(self):
        '''Plot the report of fit and data.

        :return:
            None
        '''
        
        # mask out potential zeros
        mask = self.H_data > 0.0

        # make a HR H array
        H_hr = np.logspace(-6, np.log10(max(self.H_data)), 100)

        # compute relative permeability
        mu_r = self.evaluate_mu(H_hr, relative=True)

        # compute also the derivative wrt H
        d_mu_r = self.evaluate_mu_derivative(H_hr, relative=True)

        # validate numerically
        d_mu_r_num = (mu_r - self.evaluate_mu(H_hr-1e-3, relative=True))/1e-3

        # compute permeability
        mu = 4*np.pi*1e-7*mu_r

        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.plot(self.H_data, self.B_data, 'o', label='data')
        ax.plot(H_hr, H_hr*mu, label='fit')
        ax.set_xlabel('$H$ in A/m')
        ax.set_title('$B$ in T')
        ax.set_xscale('log')
        ax.legend()
        
        ax = fig.add_subplot(222)
        ax.plot(self.H_data[mask], self.B_data[mask]/self.H_data[mask]/4/np.pi*1e7, 'o', label='data')
        ax.plot(H_hr, mu_r, label='fit')
        ax.set_xlabel('$H$ in A/m')
        ax.set_title(r'$\mu_{\text{r}}$ in Vs/Am')
        ax.legend()
        ax.set_xscale('log')

        
        ax = fig.add_subplot(223)
        ax.plot(H_hr, d_mu_r, label='analytical')
        ax.plot(H_hr, d_mu_r_num, '--', label='numerical')
        ax.set_xlabel('$H$ in A/m')
        ax.set_title(r'$\text{d}\mu_{\text{r}}/\text{d} H$ in Vs/A$^2$')
        ax.legend()
        ax.set_xscale('log')

        plt.show()
        
    def get_spline_information(self):
        '''Get the knots t, the control points c and the degree k of this spline.
        Get also the lower limit of the spline interpolation (H_min) and the
        permeability for H < H_min.
        
        :return:
            Knots vector t,
            Control points c,
            Spline degree k,
            The minimum H value,
            The value of mu for H < H_min.
        '''

        return self.spl[0], self.spl[1], self.spl[2], self.H_min, self.evaluate_mu(np.array([0.0]))[0]

    def get_type_spec(self):
        '''Return the type specifyer. 
        0 ConstantPermeability 
        1 Permeability (BSpline)
        
        :return:
            The type specifyer.
        '''

        return 1

class ConstantPermeability():

    def __init__(self, mu):
        '''Default constructor.
        
        :param mu:
            The value for the constant permeability.

        :return:
            None.
        '''

        self.mu = mu

        return None

    def evaluate_mu(self, H_mag, relative=False):
        '''Compute the permeability given the field vector magnitude.
        
        :param H_mag:
            The field vector magnitudes.

        :param relative:
            Set this flag to true to return the relative permeability.

        :return:
            The permeability values.
        '''

        if relative:
            return self.mu/4/np.pi*1e7 + 0.0*H_mag
        else:
            return self.mu + 0.0*H_mag

    def evaluate_mu_derivative(self, H_mag, relative=False):
        '''Compute the permeability derivative wrt H_mag, given the field vector magnitude.
        
        :param H_mag:
            The field vector magnitudes.

        :param relative:
            Set this flag to true to return the relative permeability.

        :return:
            The permeability derivative values.
        '''

        return 0.0 + 0.0*H_mag

    def get_type_spec(self):
        '''Return the type specifyer. 
        0 ConstantPermeability 
        1 Permeability (BSpline)
        
        :return:
            The type specifyer.
        '''

        return 0

class Reluctance():

    def __init__(self, input_filename):
        '''Default constructor.
        
        :param input_filename:
            The name of the input file with the B(H) data

        :return:
            None.
        '''

        mat_data = pd.read_csv(input_filename)

        self.B_data = mat_data['B(T)'].values
        self.H_data = mat_data['H(A/m)'].values

        self.initialize_interpolation()

        return None

    def initialize_interpolation(self):
        '''Initialize the BSpline spline interpolation.
        
        :return:
            None
        '''

        # compute nu
        self.nu = 0.0*self.H_data
        self.nu[1:] = self.H_data[1:]/self.B_data[1:]
        self.nu[0] = self.nu[1]

        # interpolate
        self.spl = PchipInterpolator(self.B_data, self.nu, extrapolate=True)
        # self.spl = splrep(self.B_data, self.nu)

        # B_hr = np.linspace(min(self.B_data), 1.5*max(self.B_data), 100)


        # nu_alt = self.evaluate_nu_alt(B_hr)

        # dnu = self.evaluate_nu_derivative(B_hr)
        # dnu_alt = self.evaluate_nu_derivative_alt(B_hr)

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.plot(self.B_data, self.nu, 'o', label='data')
        # ax.plot(B_hr, self.spl(B_hr), label='fit')
        # ax.plot(B_hr, nu_alt, '--', label='fit alt')
        # ax.set_xlabel('$B$ in T')
        # ax.set_title(r'$\nu$ in Am/Vs')
        # ax.legend()
        # ax = fig.add_subplot(122)
        # ax.plot(B_hr, dnu, label='python')
        # ax.plot(B_hr, dnu_alt, '--', label='my')
        # ax.set_xlabel('$B$ in T')
        # ax.set_title(r'$\nu$ derivative in Am/TVs')
        # ax.legend()
        # plt.show()

        return None

    def evaluate_nu(self, B_mag):
        '''Compute the reluctance given the B field vector magnitude.
        
        :param B_mag:
            The B field vector magnitudes.

        :return:
            The reluctance values.
        '''

        return self.spl(B_mag)
    
    def evaluate_nu_alt(self, B_mag):
        '''Compute the reluctance given the B field vector magnitude.
        
        :param B_mag:
            The B field vector magnitudes.

        :return:
            The reluctance values.
        '''
        c = self.spl.c
        x = self.spl.x

        nu = 0.0*B_mag

        for j in range(len(B_mag)):
            for i in range(len(x) - 1):
                if (x[i] <= B_mag[j]) and (B_mag[j] < x[i+1]):
                    for l in range(4):
                        nu[j] += c[l, i]*(B_mag[j] - x[i])**(3-l)

            if (x[-1] <= B_mag[j]):
                for l in range(4):

                    nu[j] += c[l, -1]*(B_mag[j] - x[-2])**(3-l)
        return nu
    
    def evaluate_nu_derivative(self, B_mag):
        '''Compute the reluctance derivative wrt B_mag, given the B field vector magnitude.
        
        :param B_mag:
            The B field vector magnitudes.

        :return:
            The reluctance derivative values.
        '''

        return self.spl.derivative()(B_mag)
    
    def evaluate_nu_derivative_alt(self, B_mag):
        '''Compute the reluctance derivative given the B field vector magnitude.
        This function performs the calculation to evaluate the fit directly.

        :param B_mag:
            The B field vector magnitudes.

        :return:
            The reluctance values.
        '''
        c = self.spl.c
        x = self.spl.x

        dnu = 0.0*B_mag

        for j in range(len(B_mag)):
            for i in range(len(x) - 1):
                if (x[i] < B_mag[j]) and (B_mag[j] <= x[i+1]):
                    for l in range(4):
                        dnu[j] += (3-l)*c[l, i]*(B_mag[j] - x[i])**(2-l)

            if (x[-1] < B_mag[j]):
                for l in range(4):

                    dnu[j] += (3-l)*c[l, -1]*(B_mag[j] - x[-2])**(2-l)
        return dnu
    
    def get_type_spec(self):
        '''Return the type specifyer. 
        0 ConstantReluctance 
        1 Reluctance (BSpline)
        
        :return:
            The type specifyer.
        '''

        return 1

    def get_spline_information(self):
        '''Get the knots t, the control points c of this spline.
        
        :return:
            The spline coefficients
            The B intervals
        '''
        
        return self.spl.c, self.spl.x


    def plot_permeability(self, ax, B_max, resol=100, relative=True, label='none'):
        '''Plot the permeability over B.

        :param ax:
            The matplotlib axes to plot into.

        :param B_max:
            The maximum B field for the plot.

        :param resol:
            The resolution. Default 100.

        :param relative:
            Set this flag true to plot the relative permeability.

        :param label:
            A plot label. If 'none' no label is used.

        :return:
            None
        '''

        # the B field values
        B = np.linspace(0, B_max, resol)

        # the reluctance
        nu = self.evaluate_nu(B)

        # the permeabiltiy
        mu = 1.0/nu

        if relative:

            if label == 'none':
                ax.plot(B, mu/4/np.pi*1e7)
            else:
                ax.plot(B, mu/4/np.pi*1e7, label=label)

        else:

            if label == 'none':
                ax.plot(B, mu)
            else:
                ax.plot(B, mu, label=label)

        return

class ConstantReluctance():

    def __init__(self, value):
        '''Default constructor.
        
        :param value:
            The reluctance value.

        :return:
            None.
        '''

        self.value = value

        return None

    def evaluate_nu(self, B_mag):
        '''Compute the reluctance given the B field vector magnitude.
        
        :param B_mag:
            The B field vector magnitudes.

        :return:
            The reluctance values.
        '''

        return self.value + 0.0*B_mag

    def evaluate_nu_derivative(self, B_mag):
        '''Compute the reluctance derivative wrt B_mag, given the B field vector magnitude.
        
        :param B_mag:
            The B field vector magnitudes.

        :return:
            The reluctance derivative values.
        '''

        return 0.0*B_mag
    
    def get_type_spec(self):
        '''Return the type specifyer. 
        0 ConstantReluctance 
        1 Reluctance (BSpline)
        
        :return:
            The type specifyer.
        '''

        return 0