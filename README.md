# ⚡ Integrated Load Flow, Solar PV Simulation & Battery Optimization

An **interactive power system analysis tool** that combines:
- **4-bus load flow simulation** using [pandapower](https://www.pandapower.org/),
- **Solar PV energy simulation** using [pvlib](https://pvlib-python.readthedocs.io/),
- **Battery storage optimization** using [CVXPY](https://www.cvxpy.org/) to minimize electricity costs based on time-of-use tariffs.

This project allows users to:
- Input custom power system parameters,
- Simulate PV generation for a given location,
- Run battery charge/discharge optimization to reduce grid import costs,
- Save results and plots for analysis.

---

## 📦 Features

- **Interactive user input** for bus parameters, PV system size, tilt, and azimuth.
- **Load flow calculation** using Pandapower.
- **Solar irradiance modeling** and PV output estimation with PVLib.
- **Battery storage optimization** to minimize cost under variable tariffs.
- **Automatic CSV export** of results.
- **Graphical outputs** for:
  - Voltage profile (bus results)
  - PV generation profile
  - Battery state of charge, charge/discharge, and tariff overlay

---

## 📂 Project Structure

loadflow_project/
│
├── loadflow_project.py # Main integrated script
├── requirements.txt # Dependencies
├── README.md # This file
├── results/ # Output folder
│ ├── bus_results.csv
│ ├── line_results.csv
│ ├── pv_results_YYYY-MM-DD_to_YYYY-MM-DD.csv
│ ├── pv_battery_results.csv
│ ├── voltage_profile.png
│ ├── pv_power_profile.png
│ └── pv_battery_optimization.png


---

## 🚀 How to Run

### 1. Install Python
Requires Python 3.9+ (Anaconda recommended for easier setup).

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Run the project

python loadflow_project.py

### 4. Follow interactive prompts:
Load Flow Section:

Slack Bus Voltage (kV)

Load (MW)

Reactive Load (MVAr)

PV Simulation Section:

Location (latitude, longitude, timezone)

PV system capacity, tilt, azimuth

Battery Optimization Section:

Battery capacity, charge/discharge limits, efficiency


📊 Outputs
CSV Files

bus_results.csv — Pandapower bus voltage and load data

line_results.csv — Pandapower line flows

pv_results_*.csv — Hourly PV generation data

pv_battery_results.csv — Combined PV + battery results

Graphs

voltage_profile.png — Bus voltage profile

pv_power_profile.png — PV generation over time

pv_battery_optimization.png — Battery SoC, tariff, and power flow


🛠 Tech Stack
Python

Pandapower — Load flow analysis

PVLib — PV performance modeling

Pandas — Data handling

Matplotlib — Plotting

CVXPY — Optimization



👨‍💻 Author
Satyarth — Electrical Engineering Student | Power Systems Enthusiast

