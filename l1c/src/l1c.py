# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import writeToa, readToa
from common.io.readGeodetic import readGeodetic, getCorners
import mgrs
import numpy as np
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt
from common.io.l1cProduct import writeL1c
from matplotlib import cm


class l1c(initL1c):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1C Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the L1B in Radiances
            # -------------------------------------------------------------------------------
            toa = readToa(
                self.l1bdir,
                self.globalConfig.l1b_toa + band + '.nc'
            )
            lat, lon = readGeodetic(
                self.gmdir,
                self.globalConfig.gm_geoloc
            )
            self.checkSize(lat, toa)

            # L1C reprojection onto the MGRS grid
            # -------------------------------------------------------------------------------
            lat_l1c, lon_l1c, toa_l1c = self.l1cProjtoa(
                lat, lon, toa, band
            )

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeL1c(
                self.outdir,
                self.globalConfig.l1c_toa + band,
                lat_l1c,
                lon_l1c,
                toa_l1c
            )

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1C Module!")

    def l1cProjtoa(self, lat, lon, toa, band):
        """
        Reproyectamos las radiancias L1B sobre la cuadrícula MGRS.
        :param lat: L1B latitudes [deg]
        :param lon: L1B longitudes [deg]
        :param toa: L1B radiances
        :param band: band
        :return: L1C latitudes, longitudes and radiances
        """

        m = mgrs.MGRS()
        mgrs_tiles = set()

        # Buscamos las celdas MGRS sin guardar puntos repetidos.
        for ii in range(toa.shape[0]):
            for jj in range(toa.shape[1]):
                auxi = m.toMGRS(
                    lat[ii, jj],
                    lon[ii, jj],
                    MGRSPrecision=self.l1cConfig.mgrs_tile_precision,
                )
                mgrs_tiles.add(auxi)

        mgrs_tiles = list(mgrs_tiles)

        # Reservamos los vectores de salida para las celdas encontradas.
        toa_l1c = np.zeros(len(mgrs_tiles))
        lat_l1c = np.zeros(len(mgrs_tiles))
        lon_l1c = np.zeros(len(mgrs_tiles))

        # Pasamos las entradas a vectores de doble precisión para el ajuste.
        x = np.asarray(lat, dtype=np.float64).ravel()
        y = np.asarray(lon, dtype=np.float64).ravel()
        z = np.asarray(toa, dtype=np.float64).ravel()

        # Construimos el interpolante de las radiancias originales.
        tck = bisplrep(x, y, z)

        # Recuperamos las coordenadas de cada celda e interpolamos su radiancia.
        for ii in range(len(mgrs_tiles)):
            lat_l1c[ii], lon_l1c[ii] = m.toLatLon(mgrs_tiles[ii])
            toa_l1c[ii] = bisplev(lat_l1c[ii], lon_l1c[ii], tck)

        return lat_l1c, lon_l1c, toa_l1c

    def checkSize(self, lat, toa):
        '''
        Check the sizes of the input radiances and geodetic coordinates.
        If they don't match, exit.
        :param lat: Latitude 2D matrix
        :param toa: Radiance 2D matrix
        :return: NA
        '''

        # Comprobamos que ambas entradas sean matrices 2D.
        if lat.ndim != 2 or toa.ndim != 2:
            raise ValueError(
                "La latitud y las radiancias deben ser matrices 2D."
            )

        # Revisamos que tengamos una latitud por cada píxel de radiancia.
        if lat.shape != toa.shape:
            raise ValueError(
                "Los tamaños de la geometría y las radiancias no coinciden: "
                f"latitud {lat.shape}, radiancias {toa.shape}."
            )