from ism.src.initIsm import initIsm
import numpy as np
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF


class videoChainPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, toa, band):
        self.logger.info("EODP-ALG-ISM-3000: Video Chain")

        # Registramos la carga que llega desde la fase de detección.
        self.logPixel(toa, band, "video_input", "e-")

        # Convertimos los electrones a voltios y amplificamos la señal.
        self.logger.info(
            "EODP-ALG-ISM-3010: Electrons to Voltage "
            "- Read-out and Amplification"
        )

        toa = self.electr2Volt(
            toa,
            self.ismConfig.OCF,
            self.ismConfig.ADC_gain,
        )

        self.logPixel(toa, band, "voltage", "V")

        # Convertimos la tensión a cuentas digitales.
        self.logger.info(
            "EODP-ALG-ISM-3020: Voltage to Digital Numbers - Digitisation"
        )

        toa = self.digitisation(
            toa,
            self.ismConfig.bit_depth,
            self.ismConfig.min_voltage,
            self.ismConfig.max_voltage,
        )

        self.logPixel(toa, band, "digital_numbers", "DN")

        # Guardamos los gráficos de la salida de la cadena de vídeo.
        if self.ismConfig.save_vcu_stage:
            saveas_str = self.globalConfig.ism_toa_vcu + band
            title_str = "TOA after the VCU phase [DN]"
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

            plotF(
                [],
                toa[idalt, :],
                title_str,
                xlabel_str,
                ylabel_str,
                self.outdir,
                saveas_str,
            )

        return toa

    def logPixel(self, toa, band, etapa, unidad):
        # Registramos el píxel solicitado sin modificar la imagen.
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

    def electr2Volt(self, toa, OCF, gain_adc):
        """
        Convertimos electrones a voltios y aplicamos la amplificación.
        :param toa: imagen de entrada [e-]
        :param OCF: factor de conversión [V/e-]
        :param gain_adc: ganancia de amplificación
        :return: imagen en voltios [V]
        """
        # Convertimos la carga a tensión.
        toa = toa * OCF

        # Aplicamos la ganancia indicada.
        toa = toa * gain_adc

        return toa

    def digitisation(self, toa, bit_depth, min_voltage, max_voltage):
        """
        Convertimos voltios a cuentas digitales.
        :param toa: imagen de entrada [V]
        :param bit_depth: profundidad de bits
        :param min_voltage: tensión mínima [V]
        :param max_voltage: tensión máxima [V]
        :return: imagen en cuentas digitales [DN]
        """
        # Calculamos el máximo valor digital representable.
        max_dn = 2**bit_depth - 1

        # Conservamos la conversión utilizada anteriormente.
        toa_dn = toa / (max_voltage - min_voltage) * max_dn

        # Redondeamos antes de aplicar los límites, como en el código original.
        toa_dn = np.round(toa_dn)

        # Registramos cuántos valores necesitan limitarse.
        self.logger.info(
            "Valores por encima del rango ADC: %d; máximo=%g DN",
            int(np.count_nonzero(toa_dn > max_dn)),
            max_dn,
        )
        self.logger.info(
            "Valores por debajo del rango ADC: %d",
            int(np.count_nonzero(toa_dn < 0)),
        )

        # Limitamos la salida al rango de cuentas digitales permitido.
        toa_dn = np.clip(toa_dn, 0, max_dn)

        return toa_dn