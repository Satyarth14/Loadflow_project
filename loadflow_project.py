# pv_battery_loadflow.py
# Interactive time-series load flow with PV + optimized battery dispatch
# Requires: pvlib, pandas, numpy, matplotlib, cvxpy, pandapower, scs (and/or ecos)

import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cvxpy as cp
import pandapower as pp

# -----------------------------
# Small helpers for safe inputs
# -----------------------------
def get_float(prompt, default=None, min_val=None, max_val=None):
    while True:
        try:
            s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
            if s == "" and default is not None:
                v = float(default)
            else:
                v = float(s)
            if min_val is not None and v < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("Enter a numeric value.")

def get_str(prompt, default=None):
    s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    return s if s else default

def yes_no(prompt, default="y"):
    s = input(f"{prompt} [{default}/{'n' if default.lower()=='y' else 'y'}]: ").strip().lower()
    if s == "":
        s = default.lower()
    return s in ("y", "yes")

# -----------------------------
# 1) Interactive inputs
# -----------------------------
print("\n=== PV + Battery Optimization + Time-Series Load Flow ===\n")

lat = get_float("Latitude (deg)", default=22.57)
lon = get_float("Longitude (deg)", default=88.36)
tz = get_str("Timezone (tz database string)", default="Asia/Kolkata")
start_date = get_str("Start date (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
end_date = get_str("End date (YYYY-MM-DD)", default=start_date)
freq = get_str("Time resolution (H for hourly, 30T for 30-min, etc.)", default="H")

system_kw = get_float("PV system capacity (kW STC)", default=1.0, min_val=0.01)
tilt = get_float("Tilt angle (deg, blank = use latitude)", default=None)
if tilt is None:
    tilt = round(lat, 1)
azimuth = get_float("Azimuth (deg, 180=south in N hemisphere)", default=180)
derate = get_float("System performance ratio (0-1)", default=0.82, min_val=0.5, max_val=1.0)

# Battery
cap_kwh = get_float("Battery capacity (kWh)", default=8.0, min_val=0.1)
p_charge_max = get_float("Max charge power (kW)", default=3.0, min_val=0.1)
p_discharge_max = get_float("Max discharge power (kW)", default=3.0, min_val=0.1)
eta_charge = get_float("Charge efficiency (0-1)", default=0.97, min_val=0.5, max_val=1.0)
eta_discharge = get_float("Discharge efficiency (0-1)", default=0.97, min_val=0.5, max_val=1.0)
soc_init_frac = get_float("Initial SoC as fraction of capacity (0-1)", default=0.5, min_val=0.0, max_val=1.0)
soc_min_frac = get_float("Minimum SoC fraction (0-1)", default=0.05, min_val=0.0, max_val=1.0)
soc_max_frac = 1.0

# Tariff
print("\nTariff: default is flat $0.12/kWh with 5–9 pm peak +$0.18. You can edit in-code if needed.\n")

# -----------------------------
# 2) Build time index
# -----------------------------
start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
times = pd.date_range(start=start, end=end, freq=freq, tz=tz)
if len(times) < 2:
    raise ValueError("Time index is too short. Increase range or adjust frequency.")

# dt (hours) from index spacing
dt_hours = (times[1] - times[0]).total_seconds() / 3600.0

# -----------------------------
# 3) PV simulation (pvlib clearsky + PVWatts-like)
# -----------------------------
site = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)
solpos = site.get_solarposition(times)
cs = site.get_clearsky(times, model="ineichen")  # ghi, dni, dhi

poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth'],
    dni=cs['dni'], ghi=cs['ghi'], dhi=cs['dhi']
)
poa_global = poa['poa_global'].clip(lower=0)

p_dc_kw = system_kw * (poa_global / 1000.0)
p_ac_kw = (p_dc_kw * derate).clip(lower=0)

df = pd.DataFrame({
    "ghi_wm2": cs['ghi'],
    "dni_wm2": cs['dni'],
    "dhi_wm2": cs['dhi'],
    "poa_wm2": poa_global,
    "pv_kW": p_ac_kw
}, index=times)

pv_energy_kwh = (df["pv_kW"] * dt_hours).sum()
print(f"PV simulation complete. Energy over period: {pv_energy_kwh:.2f} kWh")

# -----------------------------
# 4) Synthetic load profile (kW) – replace with CSV if you have real data
# -----------------------------
hours_of_day = df.index.hour
load_kw = (1.5
           + 0.5 * (np.sin((hours_of_day - 7) / 12 * np.pi) + 1)
           + 1.5 * (np.exp(-0.5*((hours_of_day-8)/1.5)**2) + np.exp(-0.5*((hours_of_day-19)/1.5)**2)))
load_kw = np.maximum(load_kw, 0.5)  # ensure min load
df["load_kW"] = load_kw

# -----------------------------
# 5) Tariff array ($/kWh)
# -----------------------------
tariff = 0.12 + 0.18 * ((df.index.hour >= 17) & (df.index.hour <= 21)).astype(float)
df["tariff"] = tariff

# -----------------------------
# 6) Battery optimization (cvxpy)
# -----------------------------
print("\nRunning battery optimization...")
T = len(df)
pv = df["pv_kW"].values
load = df["load_kW"].values

soc_init = soc_init_frac * cap_kwh
soc_min = soc_min_frac * cap_kwh
soc_max = soc_max_frac * cap_kwh

charge = cp.Variable(T, nonneg=True)      # kW
discharge = cp.Variable(T, nonneg=True)   # kW
soc = cp.Variable(T+1)                    # kWh
grid_import = cp.Variable(T, nonneg=True) # kW

constraints = [soc[0] == soc_init]
for t in range(T):
    constraints += [
        soc[t+1] == soc[t] + (charge[t] * eta_charge - discharge[t] / eta_discharge) * dt_hours,
        soc[t+1] >= soc_min,
        soc[t+1] <= soc_max,
        charge[t] <= p_charge_max,
        discharge[t] <= p_discharge_max,
        grid_import[t] >= load[t] - pv[t] - discharge[t] + charge[t]
    ]

objective = cp.Minimize(cp.sum(cp.multiply(grid_import * dt_hours, tariff)))  # $ = kWh * $/kWh
prob = cp.Problem(objective, constraints)

# Try ECOS, fallback to SCS, then default
solver_used = None
for solver in (cp.ECOS, cp.SCS, None):
    try:
        if solver is None:
            prob.solve()
            solver_used = "AUTO"
        else:
            prob.solve(solver=solver)
            solver_used = {cp.ECOS:"ECOS", cp.SCS:"SCS"}[solver]
        if prob.status in ("optimal", "optimal_inaccurate"):
            break
    except Exception:
        continue

if prob.status not in ("optimal", "optimal_inaccurate"):
    raise RuntimeError(f"Optimization failed: status={prob.status}")

print(f"Optimization successful (solver: {solver_used}).")

df["charge_kW"] = np.array(charge.value).flatten()
df["discharge_kW"] = np.array(discharge.value).flatten()
df["SoC_kWh"] = np.array(soc.value[1:]).flatten()
df["grid_import_kW"] = np.array(grid_import.value).flatten()

# Baseline vs optimized costs
baseline_import_kW = np.maximum(load - pv, 0)
baseline_cost = float(np.sum(baseline_import_kW * dt_hours * tariff))
optimized_cost = float(np.sum(df["grid_import_kW"].values * dt_hours * df["tariff"].values))
savings = baseline_cost - optimized_cost

peak_baseline_kW = float(np.max(baseline_import_kW))
peak_optimized_kW = float(np.max(df["grid_import_kW"].values))
peak_reduction_kW = peak_baseline_kW - peak_optimized_kW

print(f"\n=== Cost Summary ===")
print(f"Baseline cost:   ${baseline_cost:.2f}")
print(f"Optimized cost:  ${optimized_cost:.2f}")
print(f"Cost savings:    ${savings:.2f} ({(savings/baseline_cost*100 if baseline_cost>0 else 0):.1f}%)")
print(f"Peak import: baseline {peak_baseline_kW:.2f} kW → optimized {peak_optimized_kW:.2f} kW (Δ {peak_reduction_kW:.2f} kW)")

# -----------------------------
# 7) Build your 4-bus pandapower network
# -----------------------------
net = pp.create_empty_network()

# Buses (kept from your original)
b0 = pp.create_bus(net, vn_kv=110, name="Bus 0")
b1 = pp.create_bus(net, vn_kv=110, name="Bus 1")
b2 = pp.create_bus(net, vn_kv=110, name="Bus 2")
b3 = pp.create_bus(net, vn_kv=110, name="Bus 3")

# Slack/ext grid: keep nominal 110 kV base; use vm_pu to reflect entered kV (relative to 110)
slack_voltage_kv = 110.0  # fixed nominal; change to input if you’d like
pp.create_ext_grid(net, bus=b0, vm_pu=slack_voltage_kv / 110.0, name="Grid Connection")

# Lines (from your original parameters)
pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1, length_km=10,
    r_ohm_per_km=0.5, x_ohm_per_km=0.8, c_nf_per_km=0, max_i_ka=1)
pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=15,
    r_ohm_per_km=0.6, x_ohm_per_km=0.9, c_nf_per_km=0, max_i_ka=1)
pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=20,
    r_ohm_per_km=0.7, x_ohm_per_km=1.0, c_nf_per_km=0, max_i_ka=1)

# One load at Bus 2 (we’ll update p_mw each timestep)
load_idx = pp.create_load(net, bus=b2, p_mw=0.0, q_mvar=0.0, name="Main Load")

# PV as sgen at Bus 2
pv_idx = pp.create_sgen(net, bus=b2, p_mw=0.0, name="PV Gen")

# Battery modeled as sgen (discharge) + extra load (charge)
bat_dis_idx = pp.create_sgen(net, bus=b2, p_mw=0.0, name="Battery Discharge")
bat_chg_idx = pp.create_load(net, bus=b2, p_mw=0.0, q_mvar=0.0, name="Battery Charge")

# -----------------------------
# 8) Time-series load flow loop
# -----------------------------
records = []
for t, ts in enumerate(df.index):
    # Convert kW -> MW
    load_mw = (df.at[ts, "load_kW"] + df.at[ts, "charge_kW"]) / 1000.0
    pv_mw = df.at[ts, "pv_kW"] / 1000.0
    bat_dis_mw = df.at[ts, "discharge_kW"] / 1000.0

    net.load.at[load_idx, "p_mw"] = load_mw
    net.sgen.at[pv_idx, "p_mw"] = pv_mw
    net.sgen.at[bat_dis_idx, "p_mw"] = bat_dis_mw
    net.load.at[bat_chg_idx, "p_mw"] = df.at[ts, "charge_kW"] / 1000.0

    try:
        pp.runpp(net)
    except Exception:
        try:
            pp.runpp(net, algorithm="bfsw")
        except Exception:
            print(f"Power flow failed at {ts}. Skipping timestep.")
            continue

    # Metrics
    grid_mw = float(net.res_ext_grid.p_mw.sum())  # +import / -export (usually +)
    max_line_loading = float(net.res_line.loading_percent.max())
    min_bus_v = float(net.res_bus.vm_pu.min())

    records.append({
        "time": ts,
        "grid_import_mw": grid_mw,
        "min_bus_vm_pu": min_bus_v,
        "max_line_loading_%": max_line_loading,
        "pv_mw": pv_mw,
        "load_mw": load_mw,
        "bat_discharge_mw": bat_dis_mw,
        "bat_charge_mw": df.at[ts, "charge_kW"] / 1000.0,
        "soc_kwh": df.at[ts, "SoC_kWh"],
        "tariff": df.at[ts, "tariff"],
        "grid_import_opt_kW": df.at[ts, "grid_import_kW"],
        "grid_import_base_kW": max(df.at[ts, "load_kW"] - df.at[ts, "pv_kW"], 0.0)
    })

results = pd.DataFrame.from_records(records).set_index("time")
results.to_csv("timeseries_results.csv")
print("\nSaved: timeseries_results.csv")

# -----------------------------
# 9) Plots
# -----------------------------
# Min voltage vs time
plt.figure(figsize=(10,4))
plt.plot(results.index, results["min_bus_vm_pu"])
plt.title("Minimum Bus Voltage Over Time")
plt.ylabel("V (p.u.)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("voltage_profile.png", dpi=200)
plt.show()
print("Saved: voltage_profile.png")

# Grid import baseline vs optimized
plt.figure(figsize=(10,4))
plt.plot(results.index, results["grid_import_base_kW"], label="Baseline grid import (no battery)")
plt.plot(results.index, results["grid_import_opt_kW"], label="Optimized grid import")
plt.title("Grid Import: Baseline vs Optimized")
plt.ylabel("Power (kW)")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grid_import_comparison.png", dpi=200)
plt.show()
print("Saved: grid_import_comparison.png")

# Battery SoC + tariff
fig, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(df.index, df["SoC_kWh"], label="SoC (kWh)")
ax1.set_xlabel("Time")
ax1.set_ylabel("SoC (kWh)")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.bar(df.index, df["tariff"], alpha=0.3, width=0.03 if freq.upper()=="H" else 0.02, label="Tariff ($/kWh)")
ax2.set_ylabel("Tariff ($/kWh)")	

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

plt.title("Battery SoC and Tariff")
plt.tight_layout()
plt.savefig("battery_soc_tariff.png", dpi=200)
plt.show()
print("Saved: battery_soc_tariff.png")

# -----------------------------
# 10) Summary printout
# -----------------------------
print("\n=== Summary (for README/CV) ===")
print(f"Period: {start_date} to {end_date}, resolution: {freq}, dt={dt_hours:.3f} h, timezone: {tz}")
print(f"PV size: {system_kw:.2f} kW | Battery: {cap_kwh:.1f} kWh | ηc={eta_charge:.2f}, ηd={eta_discharge:.2f}")
print(f"Baseline cost: ${baseline_cost:.2f} | Optimized cost: ${optimized_cost:.2f} | Savings: ${savings:.2f}")
print(f"Peak grid import: baseline {peak_baseline_kW:.2f} kW → optimized {peak_optimized_kW:.2f} kW (Δ {peak_reduction_kW:.2f} kW)")
print("Files: timeseries_results.csv, voltage_profile.png, grid_import_comparison.png, battery_soc_tariff.png")
