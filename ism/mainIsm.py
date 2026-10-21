# MAIN FUNCTION TO CALL THE ISM MODULE

from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import sys


# Añadimos la raíz del proyecto para ejecutar el archivo directamente.
PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ism.src.ism import ism


auxdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_codigo\\auxiliary"

indir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\input\\gradient_alt100_act150"

outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-ISM\\output_test_marco"


class salidaDoble:
    """Escribimos tanto en la consola como en el archivo de ejecución."""

    def __init__(self, consola, archivo):
        self.consola = consola
        self.archivo = archivo

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)
        return len(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()

    def __getattr__(self, nombre):
        return getattr(self.consola, nombre)


def main():
    # Creamos la carpeta para los logs de esta ejecución.
    logdir = Path(outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "ISM_execution.log"

    # Guardamos una ejecución nueva, incluyendo mensajes y avisos de consola.
    with open(log_path, "w", encoding="utf-8") as archivo:
        consola_normal = salidaDoble(sys.stdout, archivo)
        consola_errores = salidaDoble(sys.stderr, archivo)

        with (
            redirect_stdout(consola_normal),
            redirect_stderr(consola_errores),
        ):
            print("===== EJECUCIÓN DEL MÓDULO ISM =====")
            print("Inicio:", datetime.now().astimezone().isoformat())
            print("Entrada:", indir)
            print("Salida:", outdir)

            try:
                # Ejecutamos la cadena original para las bandas configuradas.
                myIsm = ism(auxdir, indir, outdir)

                print("Bandas:", myIsm.globalConfig.bands)
                myIsm.processModule()

                print("Estado: EJECUCIÓN COMPLETADA")

            except Exception:
                # Registramos el error completo y mantenemos el fallo visible.
                import traceback
                traceback.print_exc()
                raise

            finally:
                print("Fin:", datetime.now().astimezone().isoformat())

    print(f"Log de ejecución guardado en: {log_path}")


if __name__ == "__main__":
    main()