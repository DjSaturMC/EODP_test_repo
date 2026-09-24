from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")

        # Combinamos las MTF de todos los componentes para obtener la del sistema.
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO
        fstepAlt = 1 / nlines / w # 333.3333333
        fstepAct = 1 / ncolumns / w # 222.2222222

        eps = 1e-6

        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        [fAltxx, fActxx] = np.meshgrid(fAlt, fAct, indexing='ij')  # Please use ‘ij’ indexing or you will get the transpose
        f2D = np.sqrt(fAltxx * fAltxx + fActxx * fActxx)

        fc = D / (lambd * focal)

        fn2D = f2D / (1 / w)
        fr2D = f2D / fc
        fnAct = fAct / (1 / w)
        fnAlt = fAlt / (1 / w)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """

        """Calcula la MTF de difracción para una apertura circular."""
        fr_clip = np.clip(fr2D, 0.0, 1.0)

        # Expresión válida hasta la frecuencia de corte.
        Hdiff = (2.0 / np.pi) * (
            np.arccos(fr_clip)
            - fr_clip * np.sqrt(1.0 - fr_clip**2)
        )

        # Por encima de la frecuencia de corte, la MTF es cero.
        Hdiff = np.where(fr2D > 1.0, 0.0, Hdiff)

        return Hdiff

    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """

        """Calcula la MTF correspondiente al desenfoque."""
        fr2D = np.asarray(fr2D, dtype=float)

        # Argumento de la función de Bessel según la guía.
        x = np.pi * defocus * fr2D * (1.0 - fr2D)

        # En x = 0, el límite de 2·J1(x)/x es 1.
        Hdefoc = np.ones_like(x)
        mask = np.abs(x) > 1e-12
        Hdefoc[mask] = 2.0 * j1(x[mask]) / x[mask]

        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """

        """Calcula la MTF debida a los errores del frente de onda."""
        fr2D = np.asarray(fr2D, dtype=float)

        # Contribuciones de los errores de baja y alta frecuencia.
        error_LF = kLF * (wLF / lambd)**2
        error_HF = kHF * (wHF / lambd)**2

        # Modelo de aberraciones indicado en la guía.
        Hwfe = np.exp(-fr2D * (1.0 - fr2D) * (error_LF + error_HF))

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """

        """Calcula la MTF del detector a partir de la frecuencia normalizada."""
        fn2D = np.asarray(fn2D, dtype=float)

        # La integración de la luz en cada píxel produce una respuesta sinc.
        Hdet = np.abs(np.sinc(fn2D))

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """

        """Calcula la MTF de smearing en la dirección ALT."""
        fnAlt = np.asarray(fnAlt, dtype=float)

        # Se calcula una respuesta para cada frecuencia ALT.
        Hsmear = np.sinc(ksmear * fnAlt)

        # Se repite en ACT para obtener una matriz de tamaño
        # (nlines, ncolumns), como las demás MTF del sistema.
        Hsmear = np.tile(Hsmear[:, np.newaxis], (1, ncolumns))

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """

        """Calcula la MTF debida al movimiento de la plataforma."""
        fn2D = np.asarray(fn2D, dtype=float)

        # El movimiento afecta a las frecuencias de ambas direcciones.
        Hmotion = np.sinc(kmotion * fn2D)

        return Hmotion

    def plotMtf(self, Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion,
                Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """Guarda los cortes ACT y ALT y el mapa 2D de la MTF del sistema."""
        os.makedirs(directory, exist_ok=True)

        # Localizamos la frecuencia cero para hacer los cortes centrales.
        centro_alt = np.argmin(np.abs(fnAlt))
        centro_act = np.argmin(np.abs(fnAct))

        # Dibujamos un gráfico para ACT y otro para ALT.
        for direccion, frecuencias, corte in (
                ("ACT", fnAct, lambda H: H[centro_alt, :]),
                ("ALT", fnAlt, lambda H: H[:, centro_act]),
        ):
            positivas = frecuencias >= 0
            fig, ax = plt.subplots(figsize=(9, 5))

            # Mismos colores que en los gráficos de la guía.
            for nombre, H, color in (
                    ("Difracción", Hdiff, "#1f77b4"),
                    ("Desenfoque", Hdefoc, "#ff7f0e"),
                    ("Aberraciones", Hwfe, "#2ca02c"),
                    ("Detector", Hdet, "#d62728"),
                    ("Smearing", Hsmear, "#9467bd"),
                    ("Movimiento", Hmotion, "#8c564b"),
                    ("Sistema", Hsys, "black"),
            ):
                ax.plot(
                    frecuencias[positivas],
                    corte(H)[positivas],
                    color=color,
                    linewidth=2 if nombre == "Sistema" else 1,
                    label=nombre,
                )

            # Nyquist está en 0,5; ampliamos ligeramente el eje para ver la línea.
            ax.axvline(0.5, color="black", linestyle="--", label="Nyquist")
            ax.set_xlim(0, 0.51)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frecuencia espacial normalizada")
            ax.set_ylabel("MTF")
            ax.set_title(f"MTF - corte {direccion} - {band}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(
                os.path.join(directory, f"mtf_{band}_{direccion}.png"),
                dpi=150,
            )
            plt.close(fig)

        # Mapa 2D de la MTF total, con la paleta usada en la guía.
        fig, ax = plt.subplots(figsize=(8, 5))
        imagen = ax.imshow(
            Hsys,
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=np.min(Hsys),
            vmax=1,
        )
        ax.set_xlabel("ACT (píxeles)")
        ax.set_ylabel("ALT (píxeles)")
        ax.set_title(f"MTF del sistema - {band}")
        fig.colorbar(imagen, ax=ax, label="MTF")

        fig.tight_layout()
        fig.savefig(
            os.path.join(directory, f"mtf_{band}_2D.png"),
            dpi=150,
        )
        plt.close(fig)


