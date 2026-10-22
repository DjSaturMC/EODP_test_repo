# MAIN FUNCTION TO CALL THE L1C MODULE FOR THE E2E TEST

from l1c.src.l1c import l1c
from pathlib import Path


def main():

    auxdir = r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP_codigo\auxiliary"

    # Indicamos primero la geometría y después nuestra salida L1B.
    indir = (
        r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP-TS-E2E-20261022T155646Z-1-001\EODP-TS-E2E\gm_out"
        ","
        r"C:\Users\marco\OneDrive\Escritorio\PROCESADO DE DATOS TIERRA\EODP-TS-E2E-20261022T155646Z-1-001\EODP-TS-E2E\output_test_marco\l1b_out_marco"
    )

    # Separamos los resultados L1C en otra subcarpeta.
    outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP-TS-E2E-20261022T155646Z-1-001\\EODP-TS-E2E\\output_test_marco\\l1c_out_marco"

    Path(outdir).mkdir(parents=True, exist_ok=True)

    myL1c = l1c(auxdir, indir, outdir)

    # Leemos los archivos generados con la ecualización desactivada.
    myL1c.globalConfig.l1b_toa = "FALSE_l1b_toa_"

    myL1c.processModule()


if __name__ == "__main__":
    main()