from ism.src.initIsm import initIsm
import numpy as np
from common.io.writeToa import writeToa
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF


class detectionPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

        # Conservamos la semilla original para reproducir PRNU y DSNU.
        np.random.seed(self.ismConfig.seed)

    def compute(self, toa, band):
        self.logger.info("EODP-ALG-ISM-2000: Detection stage")

        self.logPixel(toa, band, "irradiance_input", "mW/m2")

        # Convertimos la irradiancia a número de fotones.
        self.logger.info("EODP-ALG-ISM-2010: Irradiances to Photons")

        area_pix = self.ismConfig.pix_size * self.ismConfig.pix_size

        toa = self.irrad2Phot(
            toa,
            area_pix,
            self.ismConfig.t_int,
            self.ismConfig.wv[int(band[-1])],
        )

        self.logPixel(toa, band, "photons", "ph")

        # Convertimos los fotones a electrones y aplicamos el límite FWC.
        self.logger.info("EODP-ALG-ISM-2030: Photons to Electrons")

        toa = self.phot2Electr(toa, self.ismConfig.QE)

        self.logPixel(toa, band, "electrons", "e-")

        if self.ismConfig.save_after_ph2e:
            saveas_str = self.globalConfig.ism_toa_e + band
            writeToa(self.outdir, saveas_str, toa)

        # Aplicamos la variación de respuesta de cada detector.
        if self.ismConfig.apply_prnu:
            self.logger.info("EODP-ALG-ISM-2020: PRNU")

            toa = self.prnu(toa, self.ismConfig.kprnu)

            self.logPixel(toa, band, "prnu", "e-")

            if self.ismConfig.save_after_prnu:
                saveas_str = self.globalConfig.ism_toa_prnu + band
                writeToa(self.outdir, saveas_str, toa)

        # Añadimos la señal oscura y su variación espacial.
        if self.ismConfig.apply_dark_signal:
            self.logger.info("EODP-ALG-ISM-2040: Dark signal")

            toa = self.darkSignal(
                toa,
                self.ismConfig.kdsnu,
                self.ismConfig.T,
                self.ismConfig.Tref,
                self.ismConfig.ds_A_coeff,
                self.ismConfig.ds_B_coeff,
            )

            self.logPixel(toa, band, "dark_signal", "e-")

            if self.ismConfig.save_after_ds:
                saveas_str = self.globalConfig.ism_toa_ds + band
                writeToa(self.outdir, saveas_str, toa)

        # Aplicamos la reducción de señal de los píxeles defectuosos.
        if self.ismConfig.apply_bad_dead:
            self.logger.info("EODP-ALG-ISM-2050: Bad/dead pixels")

            toa = self.badDeadPixels(
                toa,
                self.ismConfig.bad_pix,
                self.ismConfig.dead_pix,
                self.ismConfig.bad_pix_red,
                self.ismConfig.dead_pix_red,
            )

        self.logPixel(toa, band, "detection", "e-")

        # Guardamos la imagen y los gráficos de la fase de detección.
        if self.ismConfig.save_detection_stage:
            saveas_str = self.globalConfig.ism_toa_detection + band
            writeToa(self.outdir, saveas_str, toa)

            title_str = "TOA after the detection phase [e-]"
            xlabel_str = "ACT"
            ylabel_str = "ALT"

            plotMat2D(
                toa, title_str, xlabel_str, ylabel_str,
                self.outdir, saveas_str,
            )

            idalt = int(toa.shape[0] / 2)
            saveas_str = saveas_str + "_alt" + str(idalt)

            plotF(
                [], toa[idalt, :], title_str, xlabel_str, ylabel_str,
                self.outdir, saveas_str,
            )

        return toa

    def logPixel(self, toa, band, etapa, unidad):
        # Registramos el píxel solicitado sin modificar la imagen.
        if toa.shape[0] > 50 and toa.shape[1] > 75:
            self.logger.info(
                "PIXEL band=%s stage=%s value=%.17g unit=%s",
                band, etapa, float(toa[50, 75]), unidad,
            )
        else:
            self.logger.warning(
                "No podemos registrar [50,75]: imagen de tamaño %s",
                toa.shape,
            )

    def irrad2Phot(self, toa, area_pix, tint, wv):
        """
        Convertimos la irradiancia [mW/m2] a número de fotones.
        area_pix: área del píxel [m2].
        tint: tiempo de integración [s].
        wv: longitud de onda central [m].
        """
        # Usamos las constantes físicas del proyecto.
        h = self.constants.h_planck
        c = self.constants.speed_light

        # Convertimos la irradiancia de mW/m2 a W/m2.
        toa = np.asarray(toa, dtype=float) * 1e-3

        # Calculamos la energía recibida durante la integración.
        E_in = toa * area_pix * tint

        # Dividimos por la energía de un fotón de esta banda.
        E_photon = h * c / wv
        toa_ph = E_in / E_photon

        return toa_ph

    def phot2Electr(self, toa, QE):
        """Convertimos los fotones a electrones y limitamos la carga a FWC."""
        toa = np.asarray(toa, dtype=float)

        # Aplicamos la eficiencia cuántica.
        toae = toa * QE

        # Registramos cuántos píxeles superan la capacidad del detector.
        self.logger.info(
            "Píxeles con carga superior a FWC: %d; FWC=%g e-",
            int(np.count_nonzero(toae > self.ismConfig.FWC)),
            self.ismConfig.FWC,
        )

        # Conservamos la saturación en el mismo punto de la cadena.
        toae[toae > self.ismConfig.FWC] = self.ismConfig.FWC

        return toae

    def badDeadPixels(
        self, toa, bad_pix, dead_pix, bad_pix_red, dead_pix_red
    ):
        """Reducimos la señal en las columnas de píxeles defectuosos."""
        toa = np.array(toa, dtype=float, copy=True)

        # Calculamos el número de elementos afectados.
        toa_act = toa.shape[1]
        n_bad = int(toa_act * bad_pix / 100.0)
        n_dead = int(toa_act * dead_pix / 100.0)

        idx_bad = []
        idx_dead = []

        # Distribuimos los píxeles malos desde la columna 5.
        if n_bad > 0:
            step_bad = int(toa_act / n_bad)
            idx_bad = list(range(5, toa_act, step_bad))
            toa[:, idx_bad] *= 1.0 - bad_pix_red

        # Distribuimos los píxeles muertos desde la columna 0.
        if n_dead > 0:
            step_dead = int(toa_act / n_dead)
            idx_dead = list(range(0, toa_act, step_dead))
            toa[:, idx_dead] *= 1.0 - dead_pix_red

        # Guardamos los índices para comprobar su posición.
        np.savetxt(f"{self.outdir}/idx_bad.txt", idx_bad, fmt="%d")
        np.savetxt(f"{self.outdir}/idx_dead.txt", idx_dead, fmt="%d")

        return toa

    def prnu(self, toa, kprnu):
        """Aplicamos la variación de respuesta de los detectores."""
        toa = np.asarray(toa, dtype=float)

        # Generamos una variación por columna, igual para todas sus filas.
        prnu = np.random.standard_normal(toa.shape[1]) * kprnu
        toa = toa * (1 + prnu[np.newaxis, :])

        return toa

    def darkSignal(self, toa, kdsnu, T, Tref, ds_A_coeff, ds_B_coeff):
        """Añadimos la señal oscura y su variación entre detectores."""
        toa = np.asarray(toa, dtype=float)

        # Calculamos la componente que depende de la temperatura.
        Sd = ds_A_coeff * (T / Tref)**3
        Sd = Sd * np.exp(-ds_B_coeff * (1.0 / T - 1.0 / Tref))

        # Generamos una variación positiva por columna.
        dsnu = (
            np.abs(np.random.standard_normal(toa.shape[1])) * kdsnu
        )

        # Añadimos la misma señal oscura a todas las filas de cada columna.
        DS = Sd * (1.0 + dsnu)
        toa = toa + DS

        return toa