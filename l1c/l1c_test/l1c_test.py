from pathlib import Path
from netCDF4 import Dataset
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt


def leer_coordenadas(ruta):
    # Buscamos las coordenadas en el archivo y en sus grupos.
    nombres_lat = ("lat", "latitude", "lat_l1c", "latitudes")
    nombres_lon = ("lon", "longitude", "lon_l1c", "longitudes")
    encontrados = []

    def buscar(grupo):
        nombres = {
            nombre.lower(): nombre
            for nombre in grupo.variables
        }

        nombre_lat = next(
            (nombres[n] for n in nombres_lat if n in nombres), None
        )
        nombre_lon = next(
            (nombres[n] for n in nombres_lon if n in nombres), None
        )

        if nombre_lat is not None and nombre_lon is not None:
            lat = np.ma.asarray(
                grupo.variables[nombre_lat][:], dtype=float
            ).filled(np.nan)
            lon = np.ma.asarray(
                grupo.variables[nombre_lon][:], dtype=float
            ).filled(np.nan)

            encontrados.append((grupo.path, lat, lon))

        for subgrupo in grupo.groups.values():
            buscar(subgrupo)

    def mostrar(grupo):
        print(f"Grupo: {grupo.path}")
        print(f"Variables: {list(grupo.variables)}")

        for subgrupo in grupo.groups.values():
            mostrar(subgrupo)

    with Dataset(ruta) as ds:
        buscar(ds)

        if len(encontrados) != 1:
            mostrar(ds)
            raise ValueError(
                f"Encontramos {len(encontrados)} pares de coordenadas "
                f"en {ruta.name}; necesitamos uno."
            )

    grupo, lat, lon = encontrados[0]
    print(f"Coordenadas leídas de {ruta.name}, grupo {grupo}")

    if lat.shape != lon.shape:
        raise ValueError("Latitud y longitud tienen tamaños diferentes.")

    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError(f"Hay coordenadas no válidas en {ruta.name}.")

    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    # Convertimos las coordenadas de grados a radianes.
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Calculamos la distancia sobre una Tierra esférica.
    radio_tierra = 6371000.0  # [m]
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0)**2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    )
    a = np.clip(a, 0.0, 1.0)

    distancia = 2.0 * radio_tierra * np.arctan2(
        np.sqrt(a), np.sqrt(1.0 - a)
    )

    return distancia


def generar_plots():
    # Partimos de la ubicación de nuestro script.
    script_dir = Path(__file__).resolve().parent

    # Localizamos la carpeta principal del proyecto.
    root_project = None
    for carpeta in [script_dir] + list(script_dir.parents):
        if (
            carpeta.name == "PROCESADO DE DATOS TIERRA"
            or (carpeta / "EODP_codigo").is_dir()
        ):
            root_project = carpeta
            break

    if root_project is None:
        raise FileNotFoundError("No encontramos la carpeta del proyecto.")

    # Buscamos la carpeta de datos de nivel 1C.
    carpetas = [
        carpeta
        for carpeta in root_project.rglob("EODP-TS-L1C")
        if carpeta.is_dir()
    ]

    if len(carpetas) != 1:
        raise FileNotFoundError(
            f"Necesitamos una única carpeta EODP-TS-L1C: {carpetas}"
        )

    l1c_folder = carpetas[0]
    outdir = l1c_folder / "output_test_marco"
    outdir.mkdir(parents=True, exist_ok=True)

    # Definimos las rutas de la geometría y de nuestra salida L1C.
    ruta_l1b = (
        l1c_folder / "input" / "gm_alt100_act_150" / "geolocation.nc"
    )
    ruta_l1c = outdir / "l1c_toa_VNIR-0.nc"

    for ruta in (ruta_l1b, ruta_l1c):
        if not ruta.is_file():
            raise FileNotFoundError(f"No encontramos el archivo: {ruta}")

    lat_l1b, lon_l1b = leer_coordenadas(ruta_l1b)
    lat_l1c, lon_l1c = leer_coordenadas(ruta_l1c)

    if lat_l1b.ndim != 2:
        raise ValueError("La geometría L1B debe ser una matriz 2D.")

    # Primer plot: representamos ambas cuadrículas mediante puntos.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dibujamos los puntos originales L1B en rojo.
    ax.scatter(
        lon_l1b.ravel(), lat_l1b.ravel(),
        color="red",
        marker="o",
        s=5,
        linewidths=0,
        label="L1B",
    )

    # Superponemos los puntos reproyectados L1C en azul.
    ax.scatter(
        lon_l1c.ravel(), lat_l1c.ravel(),
        color="blue",
        marker="o",
        s=12,
        linewidths=0,
        label="L1C MGRS",
        zorder=3,
    )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Projection on ground")
    ax.grid(alpha=0.3)
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(outdir / "l1b_vs_l1c_grid.png", dpi=200)

    # Seleccionamos la fila y la columna centrales de L1B.
    fila = lat_l1b.shape[0] // 2
    columna = lat_l1b.shape[1] // 2

    # Medimos la separación entre columnas en la fila central: ACT.
    ssd_act = haversine(
        lat_l1b[fila, :-1], lon_l1b[fila, :-1],
        lat_l1b[fila, 1:], lon_l1b[fila, 1:],
    )

    # Medimos la separación entre filas en la columna central: ALT.
    ssd_alt = haversine(
        lat_l1b[:-1, columna], lon_l1b[:-1, columna],
        lat_l1b[1:, columna], lon_l1b[1:, columna],
    )

    # Segundo plot: usamos dos paneles con escalas verticales independientes.
    # Colocamos ACT a la izquierda y ALT a la derecha.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle("L1B Spatial Sampling Distance - Haversine")

    # Mostramos la dirección ACT en rojo.
    axes[0].plot(
        np.arange(len(ssd_act)), ssd_act,
        color="red", linewidth=1.5,
    )
    axes[0].set_title(f"ACT - central row {fila}")
    axes[0].set_xlabel("ACT pixel interval i → i+1")
    axes[0].set_ylabel("SSD [m]")

    # Mostramos la dirección ALT en azul.
    axes[1].plot(
        np.arange(len(ssd_alt)), ssd_alt,
        color="blue", linewidth=1.5,
    )
    axes[1].set_title(f"ALT - central column {columna}")
    axes[1].set_xlabel("ALT pixel interval i → i+1")
    axes[1].set_ylabel("SSD [m]")

    # Mostramos suficientes decimales para apreciar la pequeña variación.
    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        ax.grid(alpha=0.3)
        ax.margins(x=0.02, y=0.1)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "l1b_ssd_haversine.png", dpi=200)

    # Comprobamos los rangos de distancia obtenidos.
    print(f"Fila central: {fila}; columna central: {columna}")
    print(f"SSD ACT: {ssd_act.min():.6f} a {ssd_act.max():.6f} m")
    print(f"SSD ALT: {ssd_alt.min():.6f} a {ssd_alt.max():.6f} m")
    print(f"Gráficas guardadas en: {outdir}")

    plt.show()


if __name__ == "__main__":
    generar_plots()