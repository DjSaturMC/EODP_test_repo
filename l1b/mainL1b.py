
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_codigo\\auxiliary'
indir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\input"
outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\output_test_marco"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
