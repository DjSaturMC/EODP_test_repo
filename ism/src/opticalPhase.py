from ism.src.initIsm import initIsm
from ism.src.mtf import mtf
from numpy.fft import fftshift, ifft2, fft2
import numpy as np
from common.io.writeToa import writeToa
from common.io.readIsrf import readIsrf
from scipy.interpolate import interp1d
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
from common.src.auxFunc import getIndexBand


class opticalPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, sgm_toa, sgm_wv, band):
        """
        Aplicamos la integración espectral, convertimos radiancias
        a irradiancias y aplicamos el filtro espacial.
        Devolvemos la imagen en irradiancias [mW/m2].
        """
        self.logger.info("EODP-ALG-ISM-1000: Optical stage")

        # Integramos el espectro con la respuesta de esta banda.
        self.logger.info(
            "EODP-ALG-ISM-1010: Spectral modelling. ISRF"
        )

        toa = self.spectralIntegration(sgm_toa, sgm_wv, band)

        self.logPixel(toa, band, "isrf", "mW/m2/sr")

        if self.ismConfig.save_after_isrf:
            saveas_str = self.globalConfig.ism_toa_isrf + band
            writeToa(self.outdir, saveas_str, toa)

        # Convertimos las radiancias a irradiancias.
        self.logger.info(
            "EODP-ALG-ISM-1020: Radiances to Irradiances"
        )

        toa = self.rad2Irrad(
            toa,
            self.ismConfig.D,
            self.ismConfig.f,
            self.ismConfig.Tr,
        )

        self.logPixel(toa, band, "irradiance_before_mtf", "mW/m2")

        # Calculamos las contribuciones a la MTF del sistema.
        self.logger.info(
            "EODP-ALG-ISM-1030: Spatial modelling. PSF/MTF"
        )

        myMtf = mtf(self.logger, self.outdir)

        Hsys = myMtf.system_mtf(
            toa.shape[0],
            toa.shape[1],
            self.ismConfig.D,
            self.ismConfig.wv[getIndexBand(band)],
            self.ismConfig.f,
            self.ismConfig.pix_size,
            self.ismConfig.kLF,
            self.ismConfig.wLF,
            self.ismConfig.kHF,
            self.ismConfig.wHF,
            self.ismConfig.defocus,
            self.ismConfig.ksmear,
            self.ismConfig.kmotion,
            self.outdir,
            band,
        )

        # Aplicamos la misma MTF que utilizábamos anteriormente.
        toa = self.applySysMtf(toa, Hsys)

        self.logPixel(toa, band, "optical", "mW/m2")

        # Guardamos la imagen y los gráficos de la fase óptica.
        if self.ismConfig.save_optical_stage:
            saveas_str = self.globalConfig.ism_toa_optical + band
            writeToa(self.outdir, saveas_str, toa)

            title_str = "TOA after the optical phase [mW/m2]"
            xlabel_str = "ACT"
            ylabel_str = "ALT"

            plotMat2D(
                toa,
                title_str,
                xlabel_str,
                ylabel_str,
                self.outdir,
                saveas_str,
            )

            idalt = int(toa.shape[0] / 2)
            saveas_str = saveas_str + "_alt" + str(idalt)

            plotF([], toa[idalt, :], title_str, xlabel_str, 'Irradiancia [mW/m2]', self.outdir, saveas_str)

        return toa

    def logPixel(self, toa, band, etapa, unidad):
        # Registramos el píxel solicitado sin modificar los datos.
        if toa.shape[0] > 50 and toa.shape[1] > 75:
            self.logger.info(
                "PIXEL band=%s stage=%s value=%.17g unit=%s",
                band,
                etapa,
                float(toa[50, 75]),
                unidad,
            )
        else:
            self.logger.warning(
                "No podemos registrar [50,75]: imagen de tamaño %s",
                toa.shape,
            )

    def rad2Irrad(self, toa, D, f, Tr):
        """
        Convertimos radiancias [mW/m2/sr] a irradiancias [mW/m2].
        D: diámetro de la pupila [m].
        f: distancia focal [m].
        Tr: transmitancia óptica.
        """
        # Aplicamos el factor de conversión indicado en la guía.
        toa = Tr * toa * (np.pi / 4) * (D / f)**2

        return toa

    def applySysMtf(self, toa, Hsys):
        """
        Aplicamos la MTF del sistema a la imagen en irradiancias.
        La entrada y la salida están en [mW/m2].
        """
        # Transformamos la imagen al dominio de frecuencias.
        GE = fft2(toa)

        # Colocamos la frecuencia cero de la MTF en la posición de la FFT.
        GE = GE * fftshift(Hsys)

        # Recuperamos la imagen y conservamos la parte real.
        toa_ft = ifft2(GE)
        toa_ft = np.real(toa_ft)

        return toa_ft

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integramos el cubo espectral con la ISRF de la banda.
        Devolvemos una imagen 2D en radiancias [mW/m2/sr].
        """
        # Conservamos el procedimiento proporcionado por la profesora.
        isrf, wv_isrf = readIsrf(
            self.auxdir + "/" + self.ismConfig.isrffile,
            band,
        )

        toa = np.zeros((sgm_toa.shape[0], sgm_toa.shape[1]))

        # Normalizamos la respuesta espectral para que sus pesos sumen uno.
        isrf = isrf / np.sum(isrf)

        # Convertimos las longitudes de onda de la ISRF a nanómetros.
        wv_isrf = wv_isrf * 1000

        for ialt in range(sgm_toa.shape[0]):
            for iact in range(sgm_toa.shape[1]):

                # Interpolamos el espectro del píxel sobre la malla de la ISRF.
                cs = interp1d(
                    sgm_wv,
                    sgm_toa[ialt, iact, :],
                    fill_value=(0, 0),
                    bounds_error=False,
                )

                sgm_interp = cs(wv_isrf)

                # Calculamos la suma ponderada con la respuesta normalizada.
                toa[ialt, iact] = np.sum(sgm_interp * isrf)

        return toa