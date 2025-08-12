import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt

# === Helper function for validated float input ===
def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}. Try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}. Try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# === 1. Create an empty network ===
net = pp.create_empty_network()

# === 2. Create buses ===
b0 = pp.create_bus(net, vn_kv=110, name="Bus 0")
b1 = pp.create_bus(net, vn_kv=110, name="Bus 1")
b2 = pp.create_bus(net, vn_kv=110, name="Bus 2")
b3 = pp.create_bus(net, vn_kv=110, name="Bus 3")

# === 3. External grid ===
slack_voltage_kv = get_float_input("Enter slack bus voltage in kV (example: 110): ", min_val=50)
pp.create_ext_grid(net, bus=b0, vm_pu=slack_voltage_kv / 110, name="Grid Connection")

# === 4. Lines ===
pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1, length_km=10,
    r_ohm_per_km=0.5, x_ohm_per_km=0.8, c_nf_per_km=0, max_i_ka=1)
pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=15,
    r_ohm_per_km=0.6, x_ohm_per_km=0.9, c_nf_per_km=0, max_i_ka=1)
pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=20,
    r_ohm_per_km=0.7, x_ohm_per_km=1.0, c_nf_per_km=0, max_i_ka=1)

# === 5. Load ===
load_mw = get_float_input("Enter load in MW (example: 0.3): ", min_val=0)
load_mvar = get_float_input("Enter reactive load in MVar (example: 0.7): ", min_val=0)
pp.create_load(net, bus=b2, p_mw=load_mw, q_mvar=load_mvar, name="Main Load")

# Debug: Print setup
print("\n=== LOAD TABLE ===")
print(net.load)
print("\n=== BUS TABLE ===")
print(net.bus)

# === 6. Run load flow ===
try:
    pp.runpp(net)
except Exception:
    print("Standard solver failed — retrying with BFSW...")
    try:
        pp.runpp(net, algorithm='bfsw')
    except Exception:
        print("Load flow failed completely.")
        exit()

# === 7. Show results ===
print("\n=== BUS RESULTS ===")
print(net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]])

print("\n=== LINE RESULTS ===")
print(net.res_line)

# === 8. Save results to CSV with readable headers ===
bus_results = net.res_bus.copy()
line_results = net.res_line.copy()

bus_results.columns = [c.replace("_", " ").title() for c in bus_results.columns]
line_results.columns = [c.replace("_", " ").title() for c in line_results.columns]

bus_results.to_csv("bus_results.csv", index=False)
line_results.to_csv("line_results.csv", index=False)

print("\nResults saved to 'bus_results.csv' and 'line_results.csv'.")

# === 9. Plot voltage profile ===
if net.res_bus.empty:
    print("Load flow did not converge — try smaller load values.")
else:
    plt.figure(figsize=(6, 4))
    plt.plot(net.bus.index, net.res_bus["vm_pu"], marker="o", color="b")
    plt.xticks(net.bus.index, net.bus["name"], rotation=45)
    plt.xlabel("Bus")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage Profile")
    plt.ylim(0.9, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("voltage_profile.png", dpi=300)
    plt.show()
    print("Voltage profile saved as 'voltage_profile.png'")
