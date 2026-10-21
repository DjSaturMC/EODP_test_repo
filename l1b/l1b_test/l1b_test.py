# CROSS VALIDATION AND EQUALIZATION PLOTS - ALL VNIR BANDS

from pathlib import Path
from datetime import datetime
import logging
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")

# Conservamos las tolerancias del test anterior.
RTOL = 1e-5
ATOL = 1e-8

# Podemos desactivarlo para guardar las gráficas sin abrir ventanas.
SHOW_PLOTS = True


def localizar_datos():
    # Partimos de la ubicación de nuestro script.
    script_dir = Path(__file__).resolve().parent

    # Buscamos la carpeta principal del proyecto.
    root = None
    for carpeta in [script_dir] + list(script_dir.parents):
        if (
            carpeta.name == "PROCESADO DE DATOS TIERRA"
            or (carpeta / "EODP_codigo").is_dir()
        ):
            root = carpeta
            break

    if root is None:
        raise FileNotFoundError(
            "No encontramos la carpeta principal del proyecto."
        )

    carpetas = [
        carpeta
        for carpeta in root.rglob("EODP-TS-L1B")
        if carpeta.is_dir()
    ]

    if len(carpetas) != 1:
        raise FileNotFoundError(
            f"Necesitamos una única carpeta EODP-TS-L1B: {carpetas}"
        )

    return carpetas[0]


def preparar_logger(outdir):
    # Guardamos en un log la misma información que mostramos en consola.
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("L1B_TEST")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Evitamos duplicar los mensajes si ejecutamos el script otra vez.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    consola = logging.StreamHandler(sys.stdout)
    archivo = logging.FileHandler(
        logdir / "L1B_crossvalidation.log",
        mode="w",
        encoding="utf-8",
    )

    formato = logging.Formatter("%(message)s")

    for handler in (consola, archivo):
        handler.setFormatter(formato)
        logger.addHandler(handler)

    return logger


def comparar_archivo(prof_path, my_path, logger):
    # Comprobamos que existen los dos archivos.
    if not prof_path.is_file():
        logger.error("  FALTA la referencia: %s", prof_path)
        return False

    if not my_path.is_file():
        logger.error("  FALTA nuestra salida: %s", my_path)
        return False

    correcto = True

    with (
        xr.open_dataset(prof_path) as ref,
        xr.open_dataset(my_path) as mio,
    ):
        if not ref.data_vars:
            raise ValueError(
                "El archivo de referencia no contiene variables de datos."
            )

        # Comparamos cada variable sobre la matriz completa.
        for nombre in ref.data_vars:
            if nombre not in mio.data_vars:
                logger.error("  FALTA la variable %s", nombre)
                correcto = False
                continue

            if (
                ref[nombre].dims != mio[nombre].dims
                or ref[nombre].shape != mio[nombre].shape
            ):
                logger.error("  %s: dimensiones diferentes", nombre)
                correcto = False
                continue

            # Convertimos los valores antes de restar para evitar
            # desbordamientos si los archivos contienen enteros.
            v_ref = np.asarray(
                ref[nombre].values, dtype=np.float64
            )
            v_mio = np.asarray(
                mio[nombre].values, dtype=np.float64
            )

            # Comprobamos que no hay NaN ni infinitos.
            if (
                not np.all(np.isfinite(v_ref))
                or not np.all(np.isfinite(v_mio))
            ):
                logger.error(
                    "  %s: contiene NaN o infinitos", nombre
                )
                correcto = False
                continue

            diferencia = np.abs(v_mio - v_ref)
            limite = ATOL + RTOL * np.abs(v_ref)

            fuera = int(np.count_nonzero(diferencia > limite))
            exacto = np.array_equal(v_mio, v_ref)
            ok = np.allclose(
                v_mio, v_ref, rtol=RTOL, atol=ATOL
            )

            logger.info(
                "  Variable: %s; tamaño: %s",
                nombre,
                v_ref.shape,
            )
            logger.info(
                "  Diferencia máxima: %.8e",
                float(np.max(diferencia)),
            )
            logger.info(
                "  Elementos fuera de tolerancia: %d", fuera
            )

            if exacto:
                logger.info("  CORRECTO: valores idénticos")
            elif ok:
                logger.info(
                    "  CORRECTO: coincide dentro de la tolerancia"
                )
            else:
                logger.error(
                    "  DIFERENTE: supera la tolerancia"
                )
                correcto = False

    return correcto


def cross_validate(folder, logger):
    referencia = folder / "output"
    nuestros = folder / "output_test_marco"

    # Exigimos los ocho archivos de referencia de las cuatro bandas.
    esperados = {
        f"{base}{band}.nc"
        for base in ("l1b_toa_eq_", "l1b_toa_")
        for band in BANDS
    }

    # Incluimos cualquier otro .nc que hubiera en la referencia.
    disponibles = {
        str(ruta.relative_to(referencia))
        for ruta in referencia.rglob("*.nc")
    }

    nombres = sorted(esperados | disponibles)
    correctos = 0

    logger.info("\n===== CROSS VALIDATION L1B =====")
    logger.info("Referencia: %s", referencia)
    logger.info("Nuestras salidas: %s", nuestros)
    logger.info(
        "Tolerancias: rtol=%g; atol=%g", RTOL, ATOL
    )
    logger.info(
        "Criterio: abs(mio-ref) <= atol + rtol*abs(ref)"
    )

    for nombre in nombres:
        relativo = Path(nombre)

        # Comparamos la referencia con nuestro caso ecualizado TRUE.
        my_path = (
            nuestros
            / relativo.parent
            / ("TRUE_" + relativo.name)
        )

        logger.info(
            "\n%s -> %s", nombre, my_path.name
        )

        try:
            if comparar_archivo(
                referencia / relativo, my_path, logger
            ):
                correctos += 1
        except Exception:
            logger.exception(
                "  ERROR al comparar %s", nombre
            )

    logger.info("\n===== RESUMEN CROSS VALIDATION =====")
    logger.info("Archivos exigidos: %d", len(nombres))
    logger.info("Correctos: %d", correctos)
    logger.info(
        "Pendientes o incorrectos: %d",
        len(nombres) - correctos,
    )

    return correctos == len(nombres)


def generar_grafica(folder, band, logger):
    outdir = folder / "output_test_marco"

    truth_path = (
        folder / "input" / f"ism_toa_isrf_{band}.nc"
    )
    eq_path = outdir / f"TRUE_l1b_toa_{band}.nc"
    no_eq_path = outdir / f"FALSE_l1b_toa_{band}.nc"

    for ruta in (truth_path, eq_path, no_eq_path):
        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta el archivo: {ruta}"
            )

    # Leemos las tres imágenes en radiancias.
    with (
        xr.open_dataset(truth_path) as truth,
        xr.open_dataset(eq_path) as eq,
        xr.open_dataset(no_eq_path) as no_eq,
    ):
        v_truth = np.asarray(
            truth["toa"].values, dtype=np.float64
        )
        v_eq = np.asarray(
            eq["toa"].values, dtype=np.float64
        )
        v_no_eq = np.asarray(
            no_eq["toa"].values, dtype=np.float64
        )

        if not (
            truth["toa"].dims
            == eq["toa"].dims
            == no_eq["toa"].dims
        ):
            raise ValueError(
                "Las dimensiones de las tres imágenes no coinciden."
            )

    if (
        v_truth.ndim != 2
        or not (
            v_truth.shape == v_eq.shape == v_no_eq.shape
        )
    ):
        raise ValueError(
            "Necesitamos tres imágenes 2D del mismo tamaño."
        )

    if not all(
        np.all(np.isfinite(imagen))
        for imagen in (v_truth, v_eq, v_no_eq)
    ):
        raise ValueError(
            "Las imágenes contienen NaN o infinitos."
        )

    # Conservamos el perfil del test original:
    # promediamos las filas ALT para cada detector ACT.
    perfil_truth = np.mean(v_truth, axis=0)
    perfil_eq = np.mean(v_eq, axis=0)
    perfil_no_eq = np.mean(v_no_eq, axis=0)

    act_pixels = np.arange(v_truth.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))

    # Conservamos los colores de la guía.
    ax.plot(
        act_pixels,
        perfil_eq,
        "k-",
        label="TOA L1B with eq",
        linewidth=1.5,
        zorder=3,
    )
    ax.plot(
        act_pixels,
        perfil_no_eq,
        "r-",
        label="TOA L1B no eq",
        linewidth=1.2,
        zorder=2,
    )
    ax.plot(
        act_pixels,
        perfil_truth,
        "b-",
        label="TOA after the ISRF",
        linewidth=1.8,
        zorder=1,
    )

    ax.set_title(
        f"Effect of the Equalization for {band}"
    )
    ax.set_xlabel("ACT pixel [-]")
    ax.set_ylabel("TOA [mW/m2/sr]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="-", alpha=0.5)
    fig.tight_layout()

    # Guardamos la gráfica para incorporarla al report.
    destino = (
        outdir / f"equalization_comparison_{band}.png"
    )
    fig.savefig(destino, dpi=180)

    logger.info("Gráfica guardada: %s", destino)
    logger.info(
        "  Perfil: promedio ALT sobre %d filas",
        v_truth.shape[0],
    )

    if not SHOW_PLOTS:
        plt.close(fig)


def main():
    folder = localizar_datos()
    logger = preparar_logger(
        folder / "output_test_marco"
    )

    logger.info(
        "Inicio del test: %s",
        datetime.now().astimezone().isoformat(),
    )

    # Validamos primero los resultados numéricos.
    validacion_ok = cross_validate(folder, logger)

    # Generamos las gráficas de todas las bandas.
    graficas_ok = 0
    logger.info(
        "\n===== GRÁFICAS DE LAS CUATRO BANDAS ====="
    )

    for band in BANDS:
        try:
            generar_grafica(folder, band, logger)
            graficas_ok += 1
        except Exception:
            logger.exception(
                "ERROR al generar la gráfica de %s", band
            )

    logger.info("\n===== RESULTADO FINAL =====")
    logger.info(
        "Cross validation: %s",
        "CORRECTA" if validacion_ok else "INCOMPLETA",
    )
    logger.info(
        "Gráficas generadas: %d/4", graficas_ok
    )

    completo = validacion_ok and graficas_ok == 4

    logger.info(
        "Estado: %s",
        "COMPLETO" if completo else "REVISAR INCIDENCIAS",
    )
    logger.info(
        "Fin del test: %s",
        datetime.now().astimezone().isoformat(),
    )

    if SHOW_PLOTS:
        plt.show()

    return 0 if completo else 1


if __name__ == "__main__":
    raise SystemExit(main())