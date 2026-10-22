# E2E - TOA IMAGES, HISTOGRAMS AND GEOGRAPHIC MAPS

from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from netCDF4 import Dataset
import sys
import numpy as np
import matplotlib.pyplot as plt


class salidaDoble:

    # Guardamos los mensajes y los seguimos viendo en la consola.
    def __init__(self, consola, archivo):
        self.consola = consola
        self.archivo = archivo

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)
        self.flush()
        return len(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


def leer_array(variable):

    # Conservamos los datos ausentes como NaN.
    datos = np.ma.asarray(variable[:], dtype=float).filled(np.nan)

    return datos


def buscar_variable(ds, nombres):

    # Buscamos la variable dentro del archivo y sus grupos.
    encontrados = []

    def buscar(grupo):

        for nombre, variable in grupo.variables.items():
            if nombre.lower() in nombres:
                encontrados.append((grupo.path, nombre, variable))

        for subgrupo in grupo.groups.values():
            buscar(subgrupo)

    buscar(ds)

    if len(encontrados) != 1:
        raise ValueError(
            f"Necesitamos una variable entre {nombres}; "
            f"hemos encontrado {len(encontrados)}."
        )

    return encontrados[0]


def leer_toa(ruta):

    # Leemos la radiancia de nuestro producto L1B.
    with Dataset(ruta) as ds:
        _, nombre, variable = buscar_variable(ds, ("toa",))
        toa = leer_array(variable)

    if toa.size == 0 or not np.any(np.isfinite(toa)):
        raise ValueError(f"No encontramos radiancias válidas en {ruta.name}.")

    print(f"Leemos {ruta.name}: variable {nombre}, tamaño {toa.shape}")

    return toa


def leer_l1c(ruta):

    # Leemos las radiancias junto con sus coordenadas geográficas.
    with Dataset(ruta) as ds:

        _, _, variable = buscar_variable(ds, ("toa",))
        toa = leer_array(variable)

        _, _, variable = buscar_variable(
            ds, ("lat", "latitude", "lat_l1c", "latitudes")
        )
        lat = leer_array(variable)

        _, _, variable = buscar_variable(
            ds, ("lon", "longitude", "lon_l1c", "longitudes")
        )
        lon = leer_array(variable)

    if toa.shape != lat.shape or toa.shape != lon.shape:
        raise ValueError(
            f"Radiancias y coordenadas tienen tamaños distintos en {ruta.name}."
        )

    if toa.size == 0:
        raise ValueError(f"El producto {ruta.name} está vacío.")

    if not (
        np.all(np.isfinite(toa))
        and np.all(np.isfinite(lat))
        and np.all(np.isfinite(lon))
    ):
        raise ValueError(f"Hay valores no válidos en {ruta.name}.")

    return lat, lon, toa


def guardar_figura(fig, ruta):

    # Guardamos la figura y la cerramos para liberar memoria.
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)

    print(f"Guardamos: {ruta.name}")


def plot_toa(toa, band, plots_dir):

    if toa.ndim != 2:
        raise ValueError(f"La TOA L1B de {band} debe ser una matriz 2D.")

    # Representamos ALT creciente hacia arriba, como en Panoply.
    fig, ax = plt.subplots(figsize=(12, 6))

    imagen = ax.imshow(
        np.ma.masked_invalid(toa),
        cmap="jet",
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(f"L1B TOA - {band}")
    ax.set_xlabel("ACT pixel")
    ax.set_ylabel("ALT pixel")

    barra = fig.colorbar(
        imagen,
        ax=ax,
        orientation="horizontal",
        pad=0.15,
    )
    barra.set_label("Radiance [mW/m2/sr]")

    guardar_figura(
        fig,
        plots_dir / f"l1b_toa_image_{band}.png",
    )


def plot_histograma(toa, band, plots_dir):

    # Usamos todos los píxeles válidos, sin promediar la imagen.
    valores = toa[np.isfinite(toa)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Mantenemos cien intervalos en cada banda.
    ax.hist(
        valores,
        bins=100,
        color="gray",
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title(f"L1B TOA histogram - {band}")
    ax.set_xlabel("Radiance [mW/m2/sr]")
    ax.set_ylabel("Number of pixels")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    guardar_figura(
        fig,
        plots_dir / f"l1b_toa_histogram_{band}.png",
    )

    print(f"Píxeles válidos: {valores.size}")
    print(f"Píxeles no finitos: {toa.size - valores.size}")
    print(f"Radiancia mínima: {valores.min():.6f} mW/m2/sr")
    print(f"Radiancia máxima: {valores.max():.6f} mW/m2/sr")
    print(f"Radiancia media: {valores.mean():.6f} mW/m2/sr")
    print(f"Desviación típica: {valores.std():.6f} mW/m2/sr")


def plot_mapa(lat, lon, valores, titulo, etiqueta, ruta):

    # Colocamos cada valor en su posición geográfica.
    fig, ax = plt.subplots(figsize=(10, 8))

    puntos = ax.scatter(
        lon.ravel(),
        lat.ravel(),
        c=valores.ravel(),
        cmap="viridis",
        s=3,
        linewidths=0,
    )

    ax.set_title(titulo)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.ticklabel_format(
        axis="both",
        style="plain",
        useOffset=False,
    )
    ax.grid(alpha=0.3)

    # Aproximamos la proporción geográfica usando la latitud media.
    lat_media = np.mean(lat)
    ax.set_aspect(1.0 / np.cos(np.radians(lat_media)))

    barra = fig.colorbar(puntos, ax=ax)
    barra.set_label(etiqueta)

    guardar_figura(fig, ruta)


def revisar_altitud(ruta, plots_dir):

    # Comprobamos si la geometría incluye una variable de altura.
    nombres_altitud = (
        "altitude", "height", "elevation",
        "altitudes", "heights",
    )

    with Dataset(ruta) as ds:

        alturas = []

        def revisar(grupo):

            for nombre, variable in grupo.variables.items():

                unidades = getattr(variable, "units", "sin unidades")
                descripcion = getattr(variable, "standard_name", "")

                print(
                    f"Geometría: grupo={grupo.path}, variable={nombre}, "
                    f"tamaño={variable.shape}, unidades={unidades}"
                )

                if (
                    nombre.lower() in nombres_altitud
                    or descripcion.lower() in (
                        "altitude",
                        "height",
                        "surface_altitude",
                        "height_above_reference_ellipsoid",
                    )
                ):
                    alturas.append(
                        (nombre, leer_array(variable), unidades)
                    )

            for subgrupo in grupo.groups.values():
                revisar(subgrupo)

        revisar(ds)

        if len(alturas) != 1:
            print(
                "\nMAPA DE ALTITUD PENDIENTE: no encontramos una única "
                "variable de altura identificable."
            )
            return False

        _, _, variable = buscar_variable(
            ds, ("lat", "latitude", "latitudes")
        )
        lat = leer_array(variable)

        _, _, variable = buscar_variable(
            ds, ("lon", "longitude", "longitudes")
        )
        lon = leer_array(variable)

    nombre, altitud, unidades = alturas[0]

    # Exigimos una altura por cada par de coordenadas.
    if lat.shape != lon.shape or altitud.shape != lat.shape:
        print(
            "MAPA DE ALTITUD PENDIENTE: la altura y las coordenadas "
            "no tienen las mismas dimensiones."
        )
        return False

    validos = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(altitud)
    )

    if not np.any(validos):
        print("MAPA DE ALTITUD PENDIENTE: no hay alturas válidas.")
        return False

    plot_mapa(
        lat[validos],
        lon[validos],
        altitud[validos],
        "Altitude over the map",
        f"{nombre} [{unidades}]",
        plots_dir / "altitude_map.png",
    )

    print(
        "La altura pertenece a la geometría común; "
        "guardamos un único mapa para las cuatro bandas."
    )

    return True


def generar_plots(e2e_dir, output_dir, plots_dir):

    l1b_dir = output_dir / "l1b_out_marco"
    l1c_dir = output_dir / "l1c_out_marco"

    bandas = ("VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3")

    # Comprobamos que estén todos los productos antes de empezar.
    for band in bandas:
        for ruta in (
            l1b_dir / f"FALSE_l1b_toa_{band}.nc",
            l1c_dir / f"l1c_toa_{band}.nc",
        ):
            if not ruta.is_file():
                raise FileNotFoundError(f"No encontramos: {ruta}")

    for band in bandas:

        print(f"\n========== {band} ==========")

        toa = leer_toa(
            l1b_dir / f"FALSE_l1b_toa_{band}.nc"
        )

        plot_toa(toa, band, plots_dir)
        plot_histograma(toa, band, plots_dir)

        lat, lon, toa_l1c = leer_l1c(
            l1c_dir / f"l1c_toa_{band}.nc"
        )

        plot_mapa(
            lat,
            lon,
            toa_l1c,
            f"L1C radiance over the map - {band}",
            "Radiance [mW/m2/sr]",
            plots_dir / f"l1c_radiance_map_{band}.png",
        )

        print(f"Puntos L1C válidos: {toa_l1c.size}")

    print("\n========== GEOMETRÍA Y ALTITUD ==========")

    ruta_geometria = e2e_dir / "gm_out" / "geolocation.nc"

    if not ruta_geometria.is_file():
        raise FileNotFoundError(f"No encontramos: {ruta_geometria}")

    altitud_ok = revisar_altitud(ruta_geometria, plots_dir)

    print("\n========== RESUMEN ==========")
    print("Bandas representadas: 4")
    print("Imágenes L1B: 4")
    print("Histogramas L1B: 4")
    print("Mapas de radiancia L1C: 4")
    print(f"Mapa de altitud: {'generado' if altitud_ok else 'pendiente'}")
    print(f"Carpeta de gráficas: {plots_dir}")
    print("Este script no realiza crossvalidation con la profesora.")


def main():

    # Usamos la carpeta definitiva del test E2E.
    e2e_dir = Path(
        r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP-TS-E2E-20261022T155646Z-1-001\EODP-TS-E2E"
    )

    output_dir = e2e_dir / "output_test_marco"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"

    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logpath = logs_dir / "E2E_plots.log"

    with logpath.open("w", encoding="utf-8") as archivo:

        consola = salidaDoble(sys.stdout, archivo)
        errores = salidaDoble(sys.stderr, archivo)

        with redirect_stdout(consola), redirect_stderr(errores):

            print("E2E - IMÁGENES, HISTOGRAMAS Y MAPAS")
            print(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"Datos: {e2e_dir}")

            generar_plots(e2e_dir, output_dir, plots_dir)

    print(f"\nLog guardado en: {logpath}")


if __name__ == "__main__":
    main()