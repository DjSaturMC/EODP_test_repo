# MAIN FUNCTION TO CALL THE L1B MODULE

from pathlib import Path
import logging
import sys

# Añadimos la raíz del proyecto para poder ejecutar el archivo directamente.
PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from l1b.src.l1b import l1b


auxdir = r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_codigo\auxiliary"

indir = r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_TER_2021-20260910T154758Z-1-001\EODP_TER_2021\EODP-TS-L1B\input"

outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\output_test_marco"


def main():
    # Inicializamos el módulo y conservamos el valor original de configuración.
    myL1b = l1b(auxdir, indir, outdir)
    original = myL1b.l1bConfig.do_equalization

    # Creamos una carpeta para los logs de ejecución.
    logdir = Path(outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    try:
        # Ejecutamos las cuatro bandas con y sin ecualización.
        for equalization in (True, False):
            myL1b.l1bConfig.do_equalization = equalization
            caso = "TRUE" if equalization else "FALSE"

            # Guardamos los mensajes del módulo en un log por cada caso.
            handler = logging.FileHandler(
                logdir / f"L1B_{caso}.log",
                mode="w",
                encoding="utf-8",
            )

            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))

            myL1b.logger.addHandler(handler)

            try:
                myL1b.logger.info(
                    "Caso %s: do_equalization=%s",
                    caso,
                    equalization,
                )
                myL1b.logger.info("Entrada: %s", indir)
                myL1b.logger.info("Salida: %s", outdir)

                myL1b.processModule()

                myL1b.logger.info("Ejecución %s completada", caso)

            except Exception:
                myL1b.logger.exception(
                    "Error durante la ejecución %s", caso
                )
                raise

            finally:
                # Cerramos el log antes de pasar al siguiente caso.
                myL1b.logger.removeHandler(handler)
                handler.close()

    finally:
        # Recuperamos la configuración original al terminar.
        myL1b.l1bConfig.do_equalization = original

    print(f"Ambos casos completados. Logs guardados en: {logdir}")


if __name__ == "__main__":
    main()