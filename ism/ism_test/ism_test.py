from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr


def cross_validate(rtol=1e-5, atol=1e-8):
    # Partimos de la ubicación de este script.
    script_dir = Path(__file__).resolve().parent

    # Buscamos la carpeta principal del proyecto.
    root_project = None

    for carpeta in [script_dir] + list(script_dir.parents):
        if (
            carpeta.name == "PROCESADO DE DATOS TIERRA"
            or (carpeta / "EODP_codigo").is_dir()
        ):
            root_project = carpeta
            break

    if root_project is None:
        raise FileNotFoundError(
            "No encontramos la carpeta principal del proyecto."
        )

    # Localizamos la carpeta de datos de ISM.
    carpetas = [
        carpeta
        for carpeta in root_project.rglob("EODP-TS-ISM")
        if carpeta.is_dir()
    ]

    if len(carpetas) != 1:
        raise FileNotFoundError(
            "Necesitamos una única carpeta EODP-TS-ISM. "
            f"Hemos encontrado: {carpetas}"
        )

    ism_folder = carpetas[0]
    prof_folder = ism_folder / "output"
    my_folder = ism_folder / "output_test_marco"

    if not prof_folder.is_dir() or not my_folder.is_dir():
        raise FileNotFoundError(
            "No encontramos las carpetas output y output_test_marco "
            f"dentro de {ism_folder}"
        )

    # Añadimos el archivo de log del test.
    logdir = my_folder / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "ISM_crossvalidation.log"

    with open(log_path, "w", encoding="utf-8") as archivo_log:

        # Mostramos cada mensaje en consola y lo guardamos también en el log.
        def informar(texto=""):
            print(texto)
            archivo_log.write(str(texto) + "\n")
            archivo_log.flush()

        informar("========== CROSS VALIDATION ISM ==========")
        informar(
            f"Inicio: {datetime.now().astimezone().isoformat()}"
        )
        informar(f"Referencia: {prof_folder}")
        informar(f"Mis resultados: {my_folder}")
        informar(f"Tolerancias: rtol={rtol}, atol={atol}")
        informar(
            "Criterio: abs(mio-ref) <= atol + rtol*abs(ref)"
        )

        # Buscamos todos los archivos numéricos de la profesora.
        archivos = {
            ruta.relative_to(prof_folder)
            for ruta in prof_folder.rglob("*.nc")
        }

        # Comprobamos también que están las 56 referencias que ya validamos.
        # Así evitamos dar el test por completo si falta alguna referencia.
        bandas = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")

        nombres_base = (
            "Hdiff_",
            "Hdefoc_",
            "Hwfe_",
            "Hdet_",
            "Hsmear_",
            "Hmotion_",
            "Hsys_",
            "ism_toa_isrf_",
            "ism_toa_optical_",
            "ism_toa_e_",
            "ism_toa_prnu_",
            "ism_toa_ds_",
            "ism_toa_detection_",
            "ism_toa_",
        )

        esperados = {
            Path(nombre + banda + ".nc")
            for nombre in nombres_base
            for banda in bandas
        }

        archivos = sorted(archivos | esperados)

        correctos = 0
        diferentes = 0
        ausentes = 0
        errores = 0

        for nombre in archivos:
            prof_path = prof_folder / nombre
            my_path = my_folder / nombre

            informar(f"\n--- {nombre} ---")

            # Comprobamos que existen ambos archivos.
            if not prof_path.is_file():
                informar("FALTA: no encontramos la referencia.")
                ausentes += 1
                continue

            if not my_path.is_file():
                informar("FALTA: no encontramos nuestro archivo.")
                ausentes += 1
                continue

            try:
                with (
                    xr.open_dataset(prof_path) as ds_prof,
                    xr.open_dataset(my_path) as ds_my,
                ):
                    archivo_ok = True

                    if not ds_prof.data_vars:
                        raise ValueError(
                            "El archivo de referencia no contiene variables."
                        )

                    # Comparamos cada variable sin asumir que se llama toa.
                    for variable in ds_prof.data_vars:
                        if variable not in ds_my.data_vars:
                            informar(
                                f"  {variable}: FALTA la variable."
                            )
                            archivo_ok = False
                            continue

                        ref = ds_prof[variable]
                        mio = ds_my[variable]

                        # Conservamos la comprobación de tamaño y dimensiones.
                        if ref.dims != mio.dims or ref.shape != mio.shape:
                            informar(
                                f"  {variable}: dimensiones diferentes."
                            )
                            informar(
                                f"    Profesora: {ref.dims}, {ref.shape}"
                            )
                            informar(
                                f"    Nosotros:  {mio.dims}, {mio.shape}"
                            )
                            archivo_ok = False
                            continue

                        # Convertimos a float para evitar errores al restar enteros.
                        v_prof = ref.values.astype(float)
                        v_my = mio.values.astype(float)

                        # Exigimos valores finitos antes de comparar.
                        if (
                            not np.all(np.isfinite(v_prof))
                            or not np.all(np.isfinite(v_my))
                        ):
                            informar(
                                f"  {variable}: contiene NaN o infinitos."
                            )
                            archivo_ok = False
                            continue

                        exacto = np.array_equal(v_my, v_prof)

                        # Conservamos las tolerancias del test original.
                        ok = np.allclose(
                            v_my,
                            v_prof,
                            rtol=rtol,
                            atol=atol,
                        )

                        diferencia = np.abs(v_my - v_prof)
                        diff = np.max(diferencia)

                        limite = atol + rtol * np.abs(v_prof)
                        fuera = np.count_nonzero(
                            diferencia > limite
                        )

                        informar(
                            f"  {variable}: tamaño = {v_prof.shape}"
                        )
                        informar(
                            f"  Diferencia máxima = {diff:.8e}"
                        )
                        informar(
                            f"  Elementos fuera de tolerancia = {fuera}"
                        )

                        if exacto:
                            informar(
                                "  CORRECTO: valores idénticos."
                            )
                        elif ok:
                            informar(
                                "  CORRECTO: coincide dentro de la tolerancia."
                            )
                        else:
                            informar(
                                "  DIFERENTE: supera la tolerancia."
                            )
                            archivo_ok = False

                    if archivo_ok:
                        correctos += 1
                    else:
                        diferentes += 1

            except Exception as error:
                informar(
                    f"ERROR al comparar el archivo: {error}"
                )
                errores += 1

        # Conservamos el resumen e indicamos el resultado global.
        informar("\n========== RESUMEN ==========")
        informar(f"Archivos exigidos:      {len(archivos)}")
        informar(f"Correctos:              {correctos}")
        informar(f"Con diferencias:        {diferentes}")
        informar(f"Ausentes:               {ausentes}")
        informar(f"Con errores:            {errores}")

        completo = correctos == len(archivos)

        if completo:
            informar(
                "\nVALIDACIÓN COMPLETA: todos los archivos .nc coinciden."
            )
        else:
            informar(
                "\nVALIDACIÓN INCOMPLETA: revisamos los casos indicados."
            )

        informar(
            f"Fin: {datetime.now().astimezone().isoformat()}"
        )
        informar(f"Log guardado en: {log_path}")

    return completo


if __name__ == "__main__":
    correcto = cross_validate()
    raise SystemExit(0 if correcto else 1)