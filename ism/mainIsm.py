
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_codigo\\auxiliary'
indir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\input" # small scene
outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\PROCESADO DE DATOS TIERRA\\EODP_TER_2021-20260910T154758Z-1-001\\EODP_TER_2021\\EODP-TS-L1B\\output_test_marco"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
