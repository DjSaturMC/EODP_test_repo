# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import sys


class salidaDoble:

    # Escribimos tanto en la consola como en el archivo de ejecución.
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


def main():

    # Directory - this is the common directory for the execution of the E2E, all modules
    auxdir = r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_codigo\auxiliary"

    # GM dir + L1B dir
    indir = (
        r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_TER_2021-20260910T154758Z-1-001\EODP_TER_2021\EODP-TS-L1C\input\gm_alt100_act_150"
        ","
        r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_TER_2021-20260910T154758Z-1-001\EODP_TER_2021\EODP-TS-L1C\input\l1b_output"
    )

    # Mantenemos las barras dobles para la configuración del logger.
    outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-L1C\\output_test_marco"

    # Preparamos la carpeta donde guardamos el registro de ejecución.
    logdir = Path(outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logpath = logdir / "L1C_execution.log"

    # Guardamos los mensajes de consola y los avisos de esta ejecución.
    with logpath.open("w", encoding="utf-8") as archivo:

        consola = salidaDoble(sys.stdout, archivo)
        errores = salidaDoble(sys.stderr, archivo)

        with redirect_stdout(consola), redirect_stderr(errores):

            print(f"Auxiliares: {auxdir}")
            print(f"Entradas: {indir}")
            print(f"Resultados: {Path(outdir)}")
            print()

            # Initialise the L1C
            myL1c = l1c(auxdir, indir, outdir)
            myL1c.processModule()

    print(f"\nRegistro de ejecución guardado en: {logpath}")


if __name__ == "__main__":
    main()