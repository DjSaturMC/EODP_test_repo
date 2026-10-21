from pathlib import Path
from datetime import datetime
import sys
import re
import csv
import numpy as np
import xarray as xr


# Añadimos la raíz para leer la configuración original del proyecto.
PROJECT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.ismConfig import ismConfig
from auxiliary.constants import constants


BANDS = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")
FILA = 50
COLUMNA = 75

RTOL = 1e-5
ATOL = 1e-8


def localizar_datos():
    # Buscamos la carpeta principal desde la ubicación del script.
    script_dir = Path(__file__).resolve().parent
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
        for carpeta in root.rglob("EODP-TS-ISM")
        if carpeta.is_dir()
    ]

    if len(carpetas) != 1:
        raise FileNotFoundError(
            f"Necesitamos una única carpeta EODP-TS-ISM: {carpetas}"
        )

    return carpetas[0] / "output_test_marco"


def leer_matriz(ruta, variable="toa"):
    # Leemos una matriz sin modificar el archivo original.
    if not ruta.is_file():
        raise FileNotFoundError(f"No encontramos: {ruta}")

    with xr.open_dataset(ruta) as ds:
        matriz = np.asarray(ds[variable].values, dtype=float)

    if matriz.ndim != 2:
        raise ValueError(f"Esperábamos una matriz 2D en {ruta.name}")

    if not np.all(np.isfinite(matriz)):
        raise ValueError(f"Hay NaN o infinitos en {ruta.name}")

    return matriz


def leer_registros(ruta):
    # Recuperamos los valores reales registrados durante la ejecución.
    texto = ruta.read_text(encoding="utf-8")

    patron = (
        r"PIXEL band=(VNIR-\d+) stage=(\w+) "
        r"value=(\S+) unit=(\S+)"
    )

    registros = {}

    for banda, etapa, valor, unidad in re.findall(patron, texto):
        clave = (banda, etapa)

        if clave in registros:
            raise ValueError(
                f"Hay registros repetidos para {banda}, {etapa}."
            )

        registros[clave] = float(valor)

    return registros


def main():
    cfg = ismConfig()
    fisica = constants()

    outdir = localizar_datos()
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    registros = leer_registros(logdir / "ISM_execution.log")

    filas_resultados = []
    filas_factores = []
    correctos = 0
    comprobaciones = 0

    with open(
        logdir / "ISM_analytic.log", "w", encoding="utf-8"
    ) as archivo_log:

        # Guardamos en el log lo mismo que mostramos en consola.
        def informar(texto=""):
            print(texto)
            archivo_log.write(str(texto) + "\n")
            archivo_log.flush()

        def comprobar(banda, etapa, calculado, observado, unidad, origen):
            # Comparamos valores finitos con las tolerancias indicadas.
            nonlocal correctos, comprobaciones

            comprobaciones += 1

            ok = (
                np.isfinite(calculado)
                and np.isfinite(observado)
                and np.isclose(
                    calculado, observado, rtol=RTOL, atol=ATOL
                )
            )

            diferencia = abs(calculado - observado)

            if ok:
                correctos += 1

            informar(
                f"{etapa} [{unidad}] - {origen}\n"
                f"  Calculado: {calculado:.12g}\n"
                f"  Observado: {observado:.12g}\n"
                f"  Diferencia: {diferencia:.6e}\n"
                f"  Resultado: {'CORRECTO' if ok else 'REVISAR'}"
            )

            filas_resultados.append({
                "banda": banda,
                "etapa": etapa,
                "unidad": unidad,
                "origen": origen,
                "calculado": calculado,
                "observado": observado,
                "diferencia_absoluta": diferencia,
                "correcto": bool(ok),
            })

        informar("========== COMPROBACIÓN DEL PÍXEL [50,75] ==========")
        informar(f"Inicio: {datetime.now().astimezone().isoformat()}")
        informar(f"Tolerancias: rtol={RTOL}, atol={ATOL}")
        informar(
            "Reconstruimos la MTF por FFT; calculamos las demás "
            "etapas mediante sus fórmulas."
        )

        for indice, band in enumerate(BANDS):
            informar(f"\n========== {band} ==========")

            # Partimos de nuestra imagen después de la integración espectral.
            radiancia = leer_matriz(
                outdir / f"ism_toa_isrf_{band}.nc"
            )
            Hsys = leer_matriz(
                outdir / f"Hsys_{band}.nc", "mat"
            )

            if radiancia.shape != Hsys.shape:
                raise ValueError(
                    f"La imagen y Hsys tienen tamaños diferentes en {band}."
                )

            if (
                radiancia.shape[0] <= FILA
                or radiancia.shape[1] <= COLUMNA
            ):
                raise ValueError("La imagen no contiene el píxel [50,75].")

            L0 = radiancia[FILA, COLUMNA]
            wv = cfg.wv[indice]
            area_pix = cfg.pix_size * cfg.pix_size
            energia_foton = fisica.h_planck * fisica.speed_light / wv
            max_dn = 2**cfg.bit_depth - 1

            # Calculamos los factores de la cadena lineal.
            K_rad_irrad = cfg.Tr * (np.pi / 4) * (cfg.D / cfg.f)**2
            K_irrad_ph = 1e-3 * area_pix * cfg.t_int / energia_foton
            K_ph_e = cfg.QE
            K_e_v = cfg.OCF * cfg.ADC_gain
            K_v_dn = max_dn / (cfg.max_voltage - cfg.min_voltage)

            K_total = (
                K_rad_irrad * K_irrad_ph * K_ph_e * K_e_v * K_v_dn
            )

            factores = {
                "banda": band,
                "K_rad_irrad": K_rad_irrad,
                "K_irrad_ph": K_irrad_ph,
                "K_ph_e": K_ph_e,
                "K_e_v": K_e_v,
                "K_v_dn": K_v_dn,
                "K_total_lineal": K_total,
            }

            filas_factores.append(factores)

            for nombre, valor in factores.items():
                if nombre != "banda":
                    informar(f"{nombre}: {valor:.12g}")

            informar(f"Radiancia inicial [50,75]: {L0:.12g}")
            informar(
                "DN lineales ideales, sin MTF, PRNU, señal oscura, "
                f"saturación ni redondeo: {L0 * K_total:.12g}"
            )

            # Calculamos la irradiancia antes del filtro espacial.
            irradiancia = (
                cfg.Tr * radiancia * (np.pi / 4) * (cfg.D / cfg.f)**2
            )

            # Incluimos la influencia de los píxeles vecinos mediante la MTF.
            # Usamos Hsys guardada, ya contrastada con la referencia.
            optical = np.real(
                np.fft.ifft2(
                    np.fft.fft2(irradiancia) * np.fft.fftshift(Hsys)
                )
            )

            E0 = irradiancia[FILA, COLUMNA]
            E = optical[FILA, COLUMNA]

            informar(
                "Cambio de irradiancia debido a la MTF en [50,75]: "
                f"{E - E0:.12g} mW/m2"
            )

            # Convertimos irradiancia a fotones y después a electrones.
            fotones = (E * 1e-3) * area_pix * cfg.t_int / energia_foton
            electrones_sin_limitar = fotones * cfg.QE
            electrones = min(electrones_sin_limitar, cfg.FWC)

            informar(
                f"Electrones antes del límite FWC: "
                f"{electrones_sin_limitar:.12g}"
            )
            informar(
                f"Saturación FWC en el píxel: "
                f"{electrones_sin_limitar > cfg.FWC}"
            )

            etapas = {
                "isrf": (L0, "mW/m2/sr"),
                "irradiance_before_mtf": (E0, "mW/m2"),
                "optical": (E, "mW/m2"),
                "irradiance_input": (E, "mW/m2"),
                "photons": (fotones, "ph"),
                "electrons": (electrones, "e-"),
            }

            # Reproducimos la semilla de cada nueva detectionPhase.
            # Usamos un generador local para no alterar el del módulo.
            rng = np.random.RandomState(cfg.seed)
            ncolumns = radiancia.shape[1]
            carga = electrones

            if cfg.apply_prnu:
                prnu = rng.standard_normal(ncolumns) * cfg.kprnu
                carga = carga * (1 + prnu[COLUMNA])
                etapas["prnu"] = (carga, "e-")

                informar(
                    f"Factor PRNU del detector 75: {1 + prnu[COLUMNA]:.12g}"
                )

            if cfg.apply_dark_signal:
                Sd = cfg.ds_A_coeff * (cfg.T / cfg.Tref)**3
                Sd = Sd * np.exp(
                    -cfg.ds_B_coeff * (1.0 / cfg.T - 1.0 / cfg.Tref)
                )

                dsnu = (
                    np.abs(rng.standard_normal(ncolumns)) * cfg.kdsnu
                )
                DS = Sd * (1.0 + dsnu[COLUMNA])

                carga = carga + DS
                etapas["dark_signal"] = (carga, "e-")

                informar(f"Señal oscura del detector 75: {DS:.12g} e-")

            # Comprobamos si la columna central tiene un defecto configurado.
            if cfg.apply_bad_dead:
                n_bad = int(ncolumns * cfg.bad_pix / 100.0)
                n_dead = int(ncolumns * cfg.dead_pix / 100.0)

                if n_bad > 0:
                    step_bad = int(ncolumns / n_bad)
                    if COLUMNA in range(5, ncolumns, step_bad):
                        carga = carga * (1.0 - cfg.bad_pix_red)

                if n_dead > 0:
                    step_dead = int(ncolumns / n_dead)
                    if COLUMNA in range(0, ncolumns, step_dead):
                        carga = carga * (1.0 - cfg.dead_pix_red)

            etapas["detection"] = (carga, "e-")
            etapas["video_input"] = (carga, "e-")

            # Calculamos la tensión y la digitalización.
            voltaje = carga * cfg.OCF
            voltaje = voltaje * cfg.ADC_gain

            dn_sin_limitar = np.round(voltaje * K_v_dn)
            dn = np.clip(dn_sin_limitar, 0, max_dn)

            etapas["voltage"] = (voltaje, "V")
            etapas["digital_numbers"] = (dn, "DN")

            informar(
                f"DN redondeados antes de limitar: {dn_sin_limitar:.0f}"
            )
            informar(
                "Limitación ADC en el píxel: "
                f"{dn_sin_limitar < 0 or dn_sin_limitar > max_dn}"
            )

            # Contrastamos las etapas con los valores reales del log.
            for etapa, (calculado, unidad) in etapas.items():
                clave = (band, etapa)

                if clave not in registros:
                    raise ValueError(
                        f"Falta el registro PIXEL de {band}, {etapa}."
                    )

                comprobar(
                    band, etapa, calculado,
                    registros[clave], unidad, "log",
                )

            # Contrastamos también las etapas que tienen salida NetCDF.
            archivos = {
                "isrf": "ism_toa_isrf_",
                "optical": "ism_toa_optical_",
                "electrons": "ism_toa_e_",
                "detection": "ism_toa_detection_",
                "digital_numbers": "ism_toa_",
            }

            if cfg.apply_prnu:
                archivos["prnu"] = "ism_toa_prnu_"

            if cfg.apply_dark_signal:
                archivos["dark_signal"] = "ism_toa_ds_"

            for etapa, nombre_base in archivos.items():
                matriz = leer_matriz(
                    outdir / f"{nombre_base}{band}.nc"
                )

                if matriz.shape != radiancia.shape:
                    raise ValueError(
                        f"Tamaño incorrecto en {nombre_base}{band}.nc"
                    )

                calculado, unidad = etapas[etapa]

                comprobar(
                    band, etapa, calculado,
                    matriz[FILA, COLUMNA], unidad, "NetCDF",
                )

        # Guardamos tablas para consultar e incorporar al report.
        for nombre, filas in (
            ("ISM_analytic_pixel.csv", filas_resultados),
            ("ISM_conversion_factors.csv", filas_factores),
        ):
            with open(
                outdir / nombre, "w", newline="", encoding="utf-8"
            ) as archivo:
                writer = csv.DictWriter(
                    archivo,
                    fieldnames=list(filas[0]),
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerows(filas)

            informar(f"\nTabla guardada: {outdir / nombre}")

        informar("\n========== RESUMEN ==========")
        informar(f"Comprobaciones: {comprobaciones}")
        informar(f"Correctas: {correctos}")
        informar(f"Por revisar: {comprobaciones - correctos}")
        informar(
            f"Fin: {datetime.now().astimezone().isoformat()}"
        )

    return correctos == comprobaciones


if __name__ == "__main__":
    correcto = main()
    raise SystemExit(0 if correcto else 1)