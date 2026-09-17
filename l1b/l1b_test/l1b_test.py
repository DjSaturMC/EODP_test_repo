# CROSS VALIDATE L1B OUTPUTS EQUALIZED

# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED VERSUS THE TRUTH

# TRUTH = EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc

# ==========================================
# 3. GRÁFICA COMPARATIVA (EJE ACT PIXEL)
# ==========================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ==========================================
# 1. LOCALIZACIÓN ROBUSTA DE ARCHIVOS
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent

ROOT_PROJECT = None
for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (
        p.name == "PROCESADO DE DATOS TIERRA"
        or (p / "EODP_TER_2021").exists()
        or (p / "EODP_codigo").exists()
    ):
        ROOT_PROJECT = p
        break

if ROOT_PROJECT is None:
    ROOT_PROJECT = SCRIPT_DIR.parents[3]

matches = list(ROOT_PROJECT.rglob("EODP-TS-L1B"))

if not matches:
    raise FileNotFoundError(
        f"❌ No se encontró la carpeta 'EODP-TS-L1B' desde: {ROOT_PROJECT}"
    )

L1B_FOLDER = matches[0]
print(f"📁 Carpeta de datos localizada en: {L1B_FOLDER}")

# Archivos de datos
TRUTH_PATH = L1B_FOLDER / "input" / "ism_toa_isrf_VNIR-0.nc"
PROF_EQ = L1B_FOLDER / "output" / "l1b_toa_eq_VNIR-0.nc"

# Archivo ecualizado en DN para cross-validation
MY_EQ_DN = L1B_FOLDER / "output_test_marco" / "TRUE_l1b_toa_eq_VNIR-0.nc"

# Archivos procesados en Radiancia para la gráfica
MY_EQ_RAD = L1B_FOLDER / "output_test_marco" / "TRUE_l1b_toa_VNIR-0.nc"
MY_NO_EQ_RAD = L1B_FOLDER / "output_test_marco" / "FALSE_l1b_toa_VNIR-0.nc"

VAR_NAME = "toa"


# ==========================================
# 2. VALIDACIÓN CRUZADA (EQUALIZED DN vs PROFESORA)
# ==========================================
def cross_validate():
    print("\n--- 1. CROSS VALIDATION L1B OUTPUTS EQUALIZED ---")

    if not MY_EQ_DN.exists():
        print(f"❌ Falta tu archivo: {MY_EQ_DN}")
        return
    if not PROF_EQ.exists():
        print(f"❌ Falta archivo profe: {PROF_EQ}")
        return

    with xr.open_dataset(MY_EQ_DN) as ds_my, xr.open_dataset(PROF_EQ) as ds_prof:
        v_my = ds_my[VAR_NAME].values
        v_prof = ds_prof[VAR_NAME].values

        diff = np.nanmax(np.abs(v_my - v_prof))
        ok = np.allclose(v_my, v_prof, rtol=1e-5, atol=1e-8, equal_nan=True)

        print(f"Diferencia máxima: {diff:.2e}")
        if ok:
            print(
                "✅ RESULTADO: Tus outputs ecualizados coinciden con los de la profesora."
            )
        else:
            print("❌ RESULTADO: Hay diferencias superiores al umbral.")


cross_validate()


# ==========================================
# 3. GRÁFICA COMPARATIVA PERFECTA (FIGURE 8-3)
# ==========================================
print("\n--- 2. GENERANDO GRÁFICA COMPARATIVA ---")

if MY_EQ_RAD.exists() and MY_NO_EQ_RAD.exists() and TRUTH_PATH.exists():
    with (
        xr.open_dataset(TRUTH_PATH) as ds_t,
        xr.open_dataset(MY_EQ_RAD) as ds_e,
        xr.open_dataset(MY_NO_EQ_RAD) as ds_ne,
    ):

        v_truth = ds_t[VAR_NAME].values
        v_eq = ds_e[VAR_NAME].values
        v_no_eq = ds_ne[VAR_NAME].values

        # Promediamos sobre el eje espacial/temporal (ALT) para obtener el perfil 1D
        while v_truth.ndim > 1:
            v_truth = np.nanmean(v_truth, axis=0)
        while v_eq.ndim > 1:
            v_eq = np.nanmean(v_eq, axis=0)
        while v_no_eq.ndim > 1:
            v_no_eq = np.nanmean(v_no_eq, axis=0)

        act_pixels = np.arange(len(v_truth))

        plt.figure(figsize=(10, 5))

        # Dibujo de curvas idéntico al PDF de la profesora
        plt.plot(
            act_pixels,
            v_truth,
            "b-",
            label="TOA after the ISRF",
            linewidth=1.8,
            zorder=1,
        )
        plt.plot(
            act_pixels,
            v_no_eq,
            "r-",
            label="TOA L1B no eq",
            linewidth=1.2,
            zorder=2,
        )
        plt.plot(
            act_pixels,
            v_eq,
            "k-",
            label="TOA L1B with eq",
            linewidth=1.5,
            zorder=3,
        )

        plt.title("Effect of the Equalization for VNIR-0")
        plt.xlabel("ACT pixel [-]")
        plt.ylabel("TOA [mW/m2/sr]")
        plt.legend(loc="upper left", fontsize=9)
        plt.grid(True, linestyle="-", alpha=0.5)
        plt.tight_layout()

        plt.show()
else:
    print("❌ No se encontraron todos los archivos necesarios.")