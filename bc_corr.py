import os
import sys
import numpy as np
import argparse
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from spiceypy.spiceypy import spkezr, furnsh, j2000, spd, unload
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(module)s:%(lineno)d:%(levelname)s %(message)s")
logger.setLevel(logging.INFO)

class PlanetaryEphemeris():

    def __init__(self, path):
        """
        Initialise planetary ephemeris object, and load it for use.
        """

        self.loaded = False

        self.path = path
        self.dir, self.filename = os.path.split(self.path)

        if len(self.dir) == 0 or len(self.filename) == 0:
            raise ValueError(f"{self.path} does not appear to be a valid (absolute) path")

        # Determine the ephemeris name
        self.name, _ = os.path.splitext(self.filename)

        # Load the SPICE kernel
        try:
            furnsh(self.path)
            self.loaded = True
        except:
            raise ValueError(f"'{self.path}' does not appear to be a valid SPICE epheremis.")


    def __del__(self):

        if self.loaded:
            unload(self.path)


    def bc_corr(self, coord, time):
        """
        Calculate the barycentric correction towards a given sky coord (COORD),
        at a given time (TIME).

        Args:
            coord (astropy SkyCoord): SkyCoord object representing the location of the source
            time (astropy Time): Time object
        """

        jd = time.jd
        et = (jd - j2000())*spd()
        r_earth = spkezr("earth", et, "j2000", "NONE", "solar system barycenter")[0][:3] # Gives answer in km
        r_src_normalised = [
            np.cos(coord.ra.rad)*np.cos(coord.dec.rad),
            np.sin(coord.ra.rad)*np.cos(coord.dec.rad),
            np.sin(coord.dec.rad)
        ]
        delay = np.dot(r_earth, r_src_normalised) * u.km / c

        return delay


def main():
    # Parse the command line
    parser = argparse.ArgumentParser(description='Calculate the barycentric correction towards a given source at specific times')
    parser.add_argument('planetary_ephemeris', help='Path to a planetary ephemeris file (e.g. de430.bsp)')
    parser.add_argument('ra', type=float, help='The RA in decimal hours')
    parser.add_argument('dec', type=float, help='The declination in decimal degrees')
    parser.add_argument('times', type=float, nargs='*', help='The times at which the barycentric correction is to be calculated')
    parser.add_argument('--time_format', default='mjd', help='The format of the supplied times (default: "mjd"). Can be anything understood by the Astropy Time constructor.')
    parser.add_argument('--version', action="store_true", help='Print the software version and exit')

    args = parser.parse_args()

    if args.version:
        print(f"v{__version__}")
        sys.exit(0)

    peph = PlanetaryEphemeris()

    times = Time(args.times, format=args.time_format)

    coord = SkyCoord(ra=args.ra*u.hr, dec=args.dec*u.deg, frame='icrs')
    corrected = [peph.bc_corr(coord, time).to(u.s).value for time in times]

    print(corrected)

if __name__ == "__main__":
    main()
