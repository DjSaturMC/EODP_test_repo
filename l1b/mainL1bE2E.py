# MAIN FUNCTION TO CALL THE L1B MODULE FOR THE E2E TEST

from l1b.src.l1b import l1b
from pathlib import Path


def main():

    auxdir = r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_codigo\auxiliary"

    # Leemos los resultados generados por nuestro ISM.
    indir = r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP-TS-E2E-20261022T155646Z-1-001\EODP-TS-E2E\output_test_marco"

    # Separamos los resultados L1B en su propia subcarpeta.
    outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP-TS-E2E-20261022T155646Z-1-001\\EODP-TS-E2E\\output_test_marco\\l1b_out_marco"

    # Creamos la carpeta si todavía no existe.
    Path(outdir).mkdir(parents=True, exist_ok=True)

    myL1b = l1b(auxdir, indir, outdir)

    # Desactivamos la ecualización para este test, como indicó la profesora.
    myL1b.l1bConfig.do_equalization = False

    myL1b.processModule()


if __name__ == "__main__":
    main()