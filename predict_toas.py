from astropy import units as u
from astropy.constants import c, e, m_e, eps0
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import sys
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse

# Dispersion constant
D = e.si**2 / (8 * pi**2 * eps0 * m_e * c)

# DM units
DMunits = u.pc/u.cm**3

def dm_delay(dm, flo, fhi):
    delay = D * dm * (1/flo**2 - 1/fhi**2)
    return delay

def check_model_input_units(TEPOCH, F0, F1):

    # Get everything into astropy Time and Quantity objects
    if not isinstance(TEPOCH, Time):
        TEPOCH = Time(TEPOCH, format='mjd')

    if not isinstance(F0, u.Quantity):
        F0 *= u.Hz

    if not isinstance(F1, u.Quantity):
        F1 *= u.Hz**2

    # Check that the units are correct
    if not F0.unit.is_equivalent(u.Hz):
        raise ValueError("F0 must have units of [T]^-1")

    if not F1.unit.is_equivalent(u.Hz**2):
        raise ValueError("F1 must have units of [T]^-2")

    return TEPOCH, F0, F1

def predict_TOA(N, TEPOCH, F0, F1):
    '''
    Predict the time of arrival of pulse number N from timing parameters.

    It returns a predicted time as an MJD.

    Inputs:
        N:  The pulse number (or array of pulse numbers).

        TEPOCH:
            The epoch at which the first (N=0) pulse arrives.
            This (fittable) parameter effectively absorbs the mean
            residual offset. When fitting, it is safe to initially
            set it to any initial value, and should converge to the
            nearest pulse arrival epoch.

        F0: The rotation frequency of the pulsar (Hz).

        F1: The first time derivative of F0 (Hz^2)

    Output:
        The MJD of the Nth pulse (or pulses, if N is an array).
    '''

    TEPOCH, F0, F1 = check_model_input_units(TEPOCH, F0, F1)

    if np.all(F1 == 0): # Reduce to pure-F0 case
        T = N/F0 + TEPOCH
        return T.mjd

    #A = np.sqrt(F0**2 + 2*N*F1)  # <--- 2*N*F1 is effectively lost when second term is << first term (due to finite floating point precision)
    A = F0 + N*F1/F0 # <-- This is Taylor expansion of the above
    T = (-F0 + A)/F1 + TEPOCH
    return T.mjd

def predict_TOA_jacobian(N, TEPOCH, F0, F1):

    TEPOCH, F0, F1 = check_model_input_units(TEPOCH, F0, F1)

    # Some definitions
    A = np.sqrt(F0**2 + 2*N*F1)

    # In order for everything to have the same (array) dimensions, values
    # that don't depend on N have to be broadcast to have the same
    # shape as N. This is done by multiplying by an array filled with
    # ones:
    ones = np.ones(N.shape)

    # Various partial derivatives ('d' = '∂')
    dA_dF0 = F0/A * ones
    dA_dF1 = N/A

    dT_dTEPOCH = u.Quantity(ones)
    dT_dF0 = (-1 + dA_dF0)/F1
    dT_dF1 = -(-F0 + A)/F1**2 + dA_dF1/F1

    # Arrange the results so that N is the 0th axis
    J = np.array([
        dT_dTEPOCH,
        dT_dF0.to(u.day/u.Hz).value,
        dT_dF1.to(u.day/u.Hz**2).value,
    ]).transpose()
    return J


def parse_timfile(timfile):
    '''
    timfile is an already-open file stream object

    This function returns just the contents of the .tim file needed for this application.
    In particular, the observatory location is ignored, because the timing difference it makes
    to ULPs is negligible compared to the period, and is much smaller than the errors anyway.
    '''
    freqs = []
    mjds = []
    errs = []
    lines = timfile.readlines()
    for line in lines:
        tokens = line.split()

        # ingore comment lines and
        # ignore the "FORMAT" line (and just assume it is FORMAT 1)
        if tokens[0] in ["C", "FORMAT"]:
            continue

        # parse the line and just keep the frequency, mjd, and error
        freqs.append(float(tokens[1]))
        mjds.append(float(tokens[2]))
        errs.append(float(tokens[3]))

    freqs *= u.MHz
    mjds = Time(mjds, format='mjd')
    errs *= u.us

    return freqs, mjds, errs


def parse_parfile(parfile):
    '''
    parfile is an already-open file stream object

    This blindly parses the contents of the file into a dictionary.
    Apart from comment lines (starting with '#'s), it takes the first whitespace-delimited
    word and uses it as a key in a dictionary that contains the rest of the file's contents.
    '''
    pardict = {}
    lines = parfile.readlines()
    for line in lines:
        tokens = line.split()

        # ignore comments
        if tokens[0] == '#':
            continue

        try:
            pardict[tokens[0]] = tokens[1:]
        except:
            pass

    return pardict


def predict_N(T, TEPOCH, F0, F1):
    t = (T - TEPOCH).to(u.s)
    return F0*t + 0.5*F1*t**2

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ULP TOA predictor',
        description='Basic TOA predictor for long-period objects.',
    )

    parser.add_argument('timfile', type=argparse.FileType('r'), help='The path to a \'.tim\' TOA file. Only "FORMAT 1" is supported here; this parses that file manually (to avoid being dependent on TEMPO/PINT).')
    parser.add_argument('parfile', type=argparse.FileType('r'), help='The path to a \'.par\' ephemeris file. This is used to set the initial parameters for curve_fit.')
    parser.add_argument('--nobary', action='store_true', help='Turn off barycentring (e.g. if the provided TOAs have already been barycentred).')
    parser.add_argument('--outbary', action='store_true', help='Output barycentric MJDs (default is to convert to topocentric).')
    parser.add_argument('--no_inf_freq', action='store_true', help='Turn off DM correction to infinite frequency (e.g. if the TOAs in the .tim file have already been so corrected).')
    parser.add_argument('time_range', type=float, nargs=2, help='The range of times to produce predicted TOAs for (by default, expects MJDs, but see --time_format).')
    parser.add_argument('--outfile', type=argparse.FileType('w'), default=sys.stdout, help='The file to write out the predicted TOAs to. If not provided, the TOAs are written to stdout.')
    parser.add_argument('--outplot', help='Create an "F1 residual" plot and save it to the given path. The format is selected by the extension, and must be supported by matplotlib''s savefig() function.')
    parser.add_argument('--outfreq', type=float, help='The output TOAs are delayed for this frequency (in MHz). If not supplied, no delay is added.')
    parser.add_argument('planetary_ephemeris', help='The path to the planetary ephemeris file to use (e.g. "de440.bsp")')
    parser.add_argument('--time_format', default='mjd', help='The format of the supplied times given by the "time_range" argument (default: "mjd", but can be any valid string taken by the Astropy Time constructor).')
    parser.add_argument('--out_time_format', default='mjd', help='The output time format (default: "mjd", but can be any valid string taken by the Time class''s to_value() function).')

    args = parser.parse_args()

    # Parse the tim and parfiles
    freqs, mjds, errs = parse_timfile(args.timfile)
    pardict = parse_parfile(args.parfile)

    # Do barycentering if requested
    if not args.nobary:

        # Pull out the source position
        try:
            ra = pardict['RAJ'][0]
            dec = pardict['DECJ'][0]
        except:
            raise Exception("Could not find RAJ and/or DECJ in the provided par file (needed for barycentric correction)")

        try:
            coord = SkyCoord(f"{ra} {dec}", frame='icrs', unit=(u.hourangle, u.deg))
        except:
            raise ValueError(f"Could not parse ra='{ra}' and/or dec='{dec}' as valid SkyCoord inputs")

        from bc_corr import PlanetaryEphemeris
        peph = PlanetaryEphemeris(args.planetary_ephemeris)
        mjds += TimeDelta([peph.bc_corr(coord, time).to(u.s).value for time in mjds], format='sec')

    # Do dedispersion correction to infinite frequency if requested
    if not args.no_inf_freq:

        try:
            dm = pardict['DM'][0]*DMunits
        except:
            raise Exception("Could not find DM in the provided par file (needed for dedispersion correction)")

        mjds -= TimeDelta([dm_delay(dm, freq, np.inf*u.Hz).to(u.s).value for freq in freqs], format='sec')

    # Pull out the other needed parameters
    # Errors, if present, are assumed to be the last token on the line
    try:
        F0 = float(pardict['F0'][0]) * u.Hz
        if len(pardict['F0']) > 1:
            F0_err = float(pardict['F0'][-1]) * u.Hz
    except:
        raise Exception("Could not find F0 in the provided par file")

    try:
        F1 = float(pardict['F1'][0]) * u.Hz**2
        if len(pardict['F1']) > 1:
            F1_err = float(pardict['F1'][-1]) * u.Hz**2
    except:
        raise Exception("Could not find F1 in the provided par file")

    try:
        TEPOCH = Time(pardict['PEPOCH'][0], format='mjd')
        if len(pardict['PEPOCH']) > 1:
            TEPOCH_err = float(pardict['PEPOCH'][-1]) * u.day
    except:
        raise Exception("Could not find PEPOCH in the provided par file")

    # Refit the data to improve the ephemeris
    pcov = None # This is used later to determine whether to draw an error region

    # Assume that the given F0 and F1 values are enough to get accurate pulse numbers
    initial_Ns = np.round(predict_N(mjds, TEPOCH, F0, F1).decompose().value)

    # Prepare for using curve_fit (incompatible with astropy units, so everything must be converted
    # to just the scalar value in the correct (expected) units)
    # Supply the initial guess from the ephemeris
    p0 = (
        TEPOCH.mjd,
        F0.to(u.Hz).value,
        F1.to(u.Hz**2).value,
    )

    sigma = errs.to(u.day).value

    popt, pcov = curve_fit(predict_TOA, initial_Ns, mjds.mjd, p0=p0, sigma=sigma, jac=predict_TOA_jacobian)

    TEPOCH = Time(popt[0], format='mjd')
    F0 = popt[1] * u.Hz
    F1 = popt[2] * u.Hz**2

    TEPOCH_err, F0_err, F1_err = np.sqrt(np.diag(pcov))

    TEPOCH_err *= u.day
    F0_err *= u.Hz
    F1_err *= u.Hz**2

    # As in the refitting, assume that the F0 and F1 are sufficiently good to get the pulse numbers
    T = Time(args.time_range, format=args.time_format)
    N_start, N_end = np.floor(predict_N(T, TEPOCH, F0, F1).decompose().value)
    Ns = np.arange(N_start, N_end+1)

    TOAs = Time(predict_TOA(Ns, TEPOCH, F0, F1), format='mjd')
    non_F1_TOAs = Time(predict_TOA(Ns, TEPOCH, F0, 0), format='mjd')
    diffs = (TOAs - non_F1_TOAs).to(u.s)

    # Make plots, if requested
    if args.outplot is not None:
        initial_Ns = np.round(predict_N(mjds, TEPOCH, F0, F1).decompose().value)
        initial_non_F1_TOAs = Time(predict_TOA(initial_Ns, TEPOCH, F0, 0), format='mjd')
        initial_diffs = (mjds - initial_non_F1_TOAs).to(u.s)

        smallest_N = np.min([np.min(initial_Ns), np.min(Ns)])
        largest_N = np.max([np.max(initial_Ns), np.max(Ns)])
        model_Ns = np.round(np.linspace(smallest_N, largest_N, num=10000))
        #model_Ns = np.round(np.linspace(-97000, -96900, num=101))
        Js = predict_TOA_jacobian(model_Ns, TEPOCH, F0, F1)
        Σ = np.array(pcov)
        err = np.sqrt([Js[i,:] @ Σ @ Js[i,:].T for i in range(Js.shape[0])]) * u.day

        model_TOAs = Time(predict_TOA(model_Ns, TEPOCH, F0, F1), format='mjd')
        model_non_F1_TOAs = Time(predict_TOA(model_Ns, TEPOCH, F0, 0), format='mjd')
        model_diffs = (model_TOAs - model_non_F1_TOAs).to(u.s)

        plt.fill_between(model_Ns, (model_diffs + err).to(u.s).value, (model_diffs - err).to(u.s).value, fc="gray")
        plt.plot(model_Ns, model_diffs, 'k-', label='Model')
        plt.errorbar(initial_Ns, initial_diffs, errs, fmt='o', label='Original TOAs')
        plt.plot(Ns, diffs, 'r.', label='Predicted TOAs')
        plt.xlabel(f"Pulse number since {TEPOCH}")
        plt.ylabel(f"Deviation from non-F1 solution ({initial_diffs.unit})")
        plt.legend()
        try:
            source_name = pardict['PSRJ'][0]
            plt.title(source_name)
        except:
            pass
        plt.savefig(args.outplot)
        #plt.show()

    # Print out the results to stdout/file
    f = args.outfile
    print("# Output of:", file=f)
    print(f"#     {' '.join(sys.argv)}", file=f)
    print("#\n# Model:", file=f)
    print(f"#      TEPOCH  :  {TEPOCH.mjd} ± {TEPOCH_err.to(u.day).value} {TEPOCH_err.unit}", file=f)
    print(f"#      F0      :  {F0.value} ± {F0_err.to(F0.unit).value} {F0.unit}", file=f)
    print(f"#      F1      :  {F1.value} ± {F1_err.to(F1.unit).value} {F1.unit}", file=f)

    # First, have to un-barycentre...
    if not args.outbary:
        bary_correction = TimeDelta([peph.bc_corr(coord, time).to(u.s).value for time in TOAs], format='sec')
        TOAs -= bary_correction

    # ... and un-dedisperse
    if args.outfreq:

        # Try getting DM again, because it might not have been checked before
        try:
            dm = pardict['DM'][0]*DMunits
        except:
            raise Exception("Could not find DM in the provided par file (needed for dispersion correction)")

        freq = args.outfreq*u.MHz
        dm_correction = TimeDelta(dm_delay(dm, freq, np.inf*u.Hz).to(u.s).value, format='sec')
        TOAs += dm_correction
        print(f"#\n# Added dispersion correction of {dm_correction.to(u.s):.1f} for a DM of {dm} at frequency {freq}", file=f)

    print("#\n# Columns:", file=f)
    print(f"# Pulse number | {'Barycentric' if args.outbary else 'Topocentric'} {args.out_time_format.upper()} | Difference from non-F1 prediction (s)", file=f)
    for i in range(len(Ns)):
        print(f"{Ns[i]:.0f} {TOAs[i].to_value(args.out_time_format)} {diffs[i].value:.1e}", file=f)
    
    # Finally, write out the new par file, if requested
    # At the moment, too lazy to write it out properly
    # YET TO DO...

if __name__ == '__main__':
    main()

