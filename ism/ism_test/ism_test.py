from pathlib import Path
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
        raise FileNotFoundError("No encontramos la carpeta principal del proyecto.")

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

    print(f"Referencia: {prof_folder}")
    print(f"Mis resultados: {my_folder}")
    print(f"Tolerancias: rtol={rtol}, atol={atol}")

    # Buscamos todos los archivos numéricos de la profesora.
    archivos = sorted(prof_folder.rglob("*.nc"))

    if not archivos:
        raise FileNotFoundError("No encontramos archivos .nc de referencia.")

    correctos = 0
    diferentes = 0
    ausentes = 0
    errores = 0

    for prof_path in archivos:
        nombre = prof_path.relative_to(prof_folder)
        my_path = my_folder / nombre

        print(f"\n--- {nombre} ---")

        # Comprobamos que hemos generado el mismo archivo.
        if not my_path.is_file():
            print("FALTA: no encontramos nuestro archivo.")
            ausentes += 1
            continue

        try:
            with (
                xr.open_dataset(prof_path) as ds_prof,
                xr.open_dataset(my_path) as ds_my,
            ):
                archivo_ok = True

                if not ds_prof.data_vars:
                    raise ValueError("El archivo de referencia no contiene variables.")

                # Comparamos cada variable, sin asumir que siempre se llama toa.
                for variable in ds_prof.data_vars:
                    if variable not in ds_my.data_vars:
                        print(f"  {variable}: FALTA la variable.")
                        archivo_ok = False
                        continue

                    ref = ds_prof[variable]
                    mio = ds_my[variable]

                    # Comprobamos el tamaño y el orden de las dimensiones.
                    if ref.dims != mio.dims or ref.shape != mio.shape:
                        print(f"  {variable}: dimensiones diferentes.")
                        print(f"    Profesora: {ref.dims}, {ref.shape}")
                        print(f"    Nosotros:  {mio.dims}, {mio.shape}")
                        archivo_ok = False
                        continue

                    # Convertimos a float para evitar errores al restar enteros.
                    v_prof = ref.values.astype(float)
                    v_my = mio.values.astype(float)

                    exacto = np.array_equal(v_my, v_prof, equal_nan=True)
                    ok = np.allclose(
                        v_my, v_prof,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=True,
                    )

                    # Calculamos la diferencia máxima entre valores finitos.
                    finitos = np.isfinite(v_my) & np.isfinite(v_prof)
                    if np.any(finitos):
                        diff = np.max(
                            np.abs(v_my[finitos] - v_prof[finitos])
                        )
                        print(f"  {variable}: diferencia máxima finita = {diff:.6e}")
                    else:
                        print(f"  {variable}: sin valores finitos para calcular diferencias.")

                    if exacto:
                        print("    CORRECTO: valores idénticos.")
                    elif ok:
                        print("    CORRECTO: coincide dentro de la tolerancia.")
                    else:
                        print("    DIFERENTE: no coincide dentro de la tolerancia.")
                        archivo_ok = False

                if archivo_ok:
                    correctos += 1
                else:
                    diferentes += 1

        except Exception as error:
            print(f"ERROR al comparar el archivo: {error}")
            errores += 1

    # Mostramos el resultado global sin ocultar archivos ausentes o errores.
    print("\n========== RESUMEN ==========")
    print(f"Archivos de referencia: {len(archivos)}")
    print(f"Correctos:              {correctos}")
    print(f"Con diferencias:        {diferentes}")
    print(f"Ausentes:               {ausentes}")
    print(f"Con errores:            {errores}")

    if correctos == len(archivos):
        print("\nVALIDACIÓN COMPLETA: todos los archivos .nc coinciden.")
    else:
        print("\nVALIDACIÓN INCOMPLETA: revisamos los casos indicados arriba.")


if __name__ == "__main__":
    cross_validate()