from config.ismConfig import ismConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from common.io.readMat import writeMat
import os


class mtf:
    """Calculamos las contribuciones a la MTF del sistema."""

    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(
        self, nlines, ncolumns, D, lambd, focal, pix_size,
        kLF, wLF, kHF, wHF, defocus, ksmear, kmotion,
        directory, band,
    ):
        """Calculamos, guardamos y representamos las MTF de una banda."""
        self.logger.info(
            "Calculation of the System MTF - %s", band
        )

        # Calculamos las frecuencias con el mismo muestreo de la imagen.
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(
            nlines, ncolumns, D, lambd, focal, pix_size
        )

        # Calculamos cada contribución por separado.
        Hdiff = self.mtfDiffract(fr2D)
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)
        Hwfe = self.mtfWfeAberrations(
            fr2D, lambd, kLF, wLF, kHF, wHF
        )
        Hdet = self.mtfDetector(fn2D)
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Multiplicamos las contribuciones para obtener la MTF total.
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

        # Guardamos las siete matrices para compararlas con la referencia.
        for nombre, H in (
            ("Hdiff", Hdiff),
            ("Hdefoc", Hdefoc),
            ("Hwfe", Hwfe),
            ("Hdet", Hdet),
            ("Hsmear", Hsmear),
            ("Hmotion", Hmotion),
            ("Hsys", Hsys),
        ):
            writeMat(directory, nombre + "_" + band, H)
            self.logger.info(
                "MTF guardada: %s_%s.nc", nombre, band
            )

        self.plotMtf(
            Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion,
            Hsys, nlines, ncolumns, fnAct, fnAlt,
            directory, band,
        )

        return Hsys

    def freq2d(self, nlines, ncolumns, D, lambd, focal, w):
        """
        Calculamos las frecuencias espaciales de la imagen.
        fn2D: frecuencia radial normalizada con 1/w.
        fr2D: frecuencia radial relativa a la frecuencia de corte.
        fnAct y fnAlt: frecuencias normalizadas de cada dirección.
        """
        # Conservamos el paso y el intervalo utilizados anteriormente.
        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w
        eps = 1e-6

        fAlt = np.arange(
            -1 / (2 * w),
            1 / (2 * w) - eps,
            fstepAlt,
        )
        fAct = np.arange(
            -1 / (2 * w),
            1 / (2 * w) - eps,
            fstepAct,
        )

        # Usamos ij para conservar el orden de filas ALT y columnas ACT.
        fAltxx, fActxx = np.meshgrid(
            fAlt, fAct, indexing="ij"
        )
        f2D = np.sqrt(
            fAltxx * fAltxx + fActxx * fActxx
        )

        # Calculamos la frecuencia de corte de la óptica.
        fc = D / (lambd * focal)

        fn2D = f2D / (1 / w)
        fr2D = f2D / fc
        fnAct = fAct / (1 / w)
        fnAlt = fAlt / (1 / w)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self, fr2D):
        """Calculamos la MTF de difracción de una apertura circular."""
        # Limitamos el argumento para evaluar la expresión sin raíces inválidas.
        fr_clip = np.clip(fr2D, 0.0, 1.0)

        Hdiff = (2.0 / np.pi) * (
            np.arccos(fr_clip)
            - fr_clip * np.sqrt(1.0 - fr_clip**2)
        )

        # Anulamos la respuesta por encima de la frecuencia de corte.
        Hdiff = np.where(fr2D > 1.0, 0.0, Hdiff)

        return Hdiff

    def mtfDefocus(self, fr2D, defocus, focal, D):
        """Calculamos la contribución del desenfoque."""
        fr2D = np.asarray(fr2D, dtype=float)

        # Calculamos el argumento de la función de Bessel.
        x = np.pi * defocus * fr2D * (1.0 - fr2D)

        # Usamos el límite igual a uno cuando el argumento es cero.
        Hdefoc = np.ones_like(x)
        mask = np.abs(x) > 1e-12
        Hdefoc[mask] = 2.0 * j1(x[mask]) / x[mask]

        return Hdefoc

    def mtfWfeAberrations(
        self, fr2D, lambd, kLF, wLF, kHF, wHF
    ):
        """Calculamos la contribución de los errores de frente de onda."""
        fr2D = np.asarray(fr2D, dtype=float)

        # Calculamos las contribuciones de baja y alta frecuencia.
        error_LF = kLF * (wLF / lambd)**2
        error_HF = kHF * (wHF / lambd)**2

        # Aplicamos la expresión de la guía.
        Hwfe = np.exp(
            -fr2D * (1.0 - fr2D) * (error_LF + error_HF)
        )

        return Hwfe

    def mtfDetector(self, fn2D):
        """Calculamos la MTF debida a la integración en el píxel."""
        fn2D = np.asarray(fn2D, dtype=float)

        # Aplicamos el valor absoluto de la sinc normalizada.
        Hdet = np.abs(np.sinc(fn2D))

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """Calculamos el smearing en la dirección ALT."""
        fnAlt = np.asarray(fnAlt, dtype=float)

        # Calculamos la respuesta ALT y la repetimos en todas las columnas.
        Hsmear = np.sinc(ksmear * fnAlt)
        Hsmear = np.tile(
            Hsmear[:, np.newaxis], (1, ncolumns)
        )

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """Calculamos la contribución del movimiento de la plataforma."""
        fn2D = np.asarray(fn2D, dtype=float)

        Hmotion = np.sinc(kmotion * fn2D)

        return Hmotion

    def plotMtf(
        self, Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion,
        Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band,
    ):
        """Guardamos los cortes ACT y ALT y el mapa 2D del sistema."""
        os.makedirs(directory, exist_ok=True)

        # Localizamos la frecuencia cero para obtener los cortes centrales.
        centro_alt = np.argmin(np.abs(fnAlt))
        centro_act = np.argmin(np.abs(fnAct))

        for direccion, frecuencias, corte in (
            ("ACT", fnAct, lambda H: H[centro_alt, :]),
            ("ALT", fnAlt, lambda H: H[:, centro_act]),
        ):
            positivas = frecuencias >= 0
            fig, ax = plt.subplots(figsize=(9, 5))

            # Conservamos los colores de las gráficas de la guía.
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

            # Señalamos Nyquist y dejamos margen para ver la línea.
            ax.axvline(
                0.5,
                color="black",
                linestyle="--",
                label="Nyquist",
            )
            ax.set_xlim(0, 0.51)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frecuencia espacial normalizada")
            ax.set_ylabel("MTF")
            ax.set_title(f"MTF - corte {direccion} - {band}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            fig.tight_layout()

            name = f"mtf_{band}_{direccion}.png"
            fig.savefig(
                os.path.join(directory, name),
                dpi=150,
            )
            plt.close(fig)

            self.logger.info("Gráfica MTF guardada: %s", name)

        # Representamos la MTF total con la misma paleta anterior.
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

        name = f"mtf_{band}_2D.png"
        fig.savefig(
            os.path.join(directory, name),
            dpi=150,
        )
        plt.close(fig)

        self.logger.info("Gráfica MTF guardada: %s", name)