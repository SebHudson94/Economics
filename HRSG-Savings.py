
###PRELOADING DATA###

import timeit
import math
import decimal
import numpy as np
import cProfile
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import bisect
import pandas as pd

def get_excel_column_name(n):
    name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name

start = timeit.default_timer()

load_file = ["Price Data.xlsx"]
sheet = ["2021"]
save_file = ["Price Data Hours.xlsx"]
newVar = ["Default"]
original_file_name = sheet[0]
text_file_name = original_file_name + '.txt'  # Add the '.txt' extension to the base file name
text_save_file = [text_file_name]  # Keep it as a list with a single element, to match the original format

#Link the config variables (above) with the associated 'config21pre.txt' info
config_settings = {"LOAD FILE=":load_file,"SAVE FILE=":save_file,"SHEET NAME=":sheet}

#Try load the config settings. Checks each line for a key in the config_settings dict, and loads the data if there is a match
try:
	config = open("config21pre.txt", "r")
	for line in config:
		for setting in config_settings:
			if setting in line:
				value = line[len(setting):].strip()
				if value != "":
					config_settings[setting][0] = value
				break
	config.close()
except:
	print("'config21pre.txt' not found or has errors")

df = pd.read_excel(load_file[0], sheet_name=sheet[0], header=None)

excel_column_names = [get_excel_column_name(i) for i in range(1, df.shape[1] + 1)]

df.columns = excel_column_names

start_time = 0
operating_hours = 24

charge_price = [float('inf')] + df['G'].iloc[:-3].astype(float).tolist()
charge_points = [None] * len(charge_price)

df['H'] = df.apply(lambda row: row['G'] if (row['D'] > (start_time / 24) * 48) and (row['D'] <= (start_time / 24) + (operating_hours / 24) * 48) else -100, axis=1)

df.to_excel(save_file[0], index=False)

###LOADING DATA###

constant_gas_value = 0
constant_tax_value = 0

charge_price = [float('inf')] + df['G'].iloc[1:-3].astype(float).tolist()
discharge_price = [float('-inf')] + df['G'].iloc[1:-3].astype(float).tolist()
gas_price = [float('-inf')] + df['I'].iloc[1:-3].astype(float).tolist()
carbon_tax_price = [float('-inf')] + df['L'].iloc[1:-3].astype(float).tolist()
gas_price_ave = df['I'].iloc[1:-3].astype(float).tolist()
carbon_tax_price_ave = df['L'].iloc[1:-3].astype(float).tolist()

gas_prices = [None] * (len(gas_price))
carbon_tax_prices = [None] * len(carbon_tax_price)
gas_prices_ave = [None] * (len(gas_price_ave))
carbon_tax_prices_ave = [None] * len(carbon_tax_price_ave)

for i in range(len(gas_price)):
    gas_prices[i] = gas_price[i]

for i in range(len(carbon_tax_price)):
    carbon_tax_prices[i] = carbon_tax_price[i]

for i in range(len(gas_price_ave)):
    gas_prices_ave[i] = gas_price_ave[i]

for i in range(len(carbon_tax_price_ave)):
    carbon_tax_prices_ave[i] = carbon_tax_price_ave[i]

for g in gas_prices_ave:
    if g is not None:
        constant_gas_value += g 

for t in carbon_tax_prices_ave:
    if t is not None:
        constant_tax_value += t 
            
SPs = df['E'].count()

efficiency = 0.45
plant_output = 250 #MW
combustion_efficiency = 0.995
efficacy = 0.9
fuel_co2 = 0.18416
cpf = 18
carbon_price = [carbon_priced for carbon_priced in carbon_tax_price]
P_tax = (constant_tax_value / SPs) + cpf
eur_gbp = 0.8581
P_gas = (((constant_gas_value / SPs) / 29.30711) * 10) # 201.09
fuel_cv = 50.144 #MJ/kg
fuel_price_per_kWh_in = [price / 29.30711 for price in gas_price]  # £/kWh_in
fuel_price_per_MWh_in = fuel_price_per_kWh_in * 10 #£/MWh_in
O_and_M = 3 #£/MWh
sp_length = 30 #min
time_step = 60 * sp_length #s
time_step_hours = time_step / 3600 #hr
daily_time_steps = 24 * time_step_hours

charge_price = np.array(charge_price)
discharge_price = np.array(discharge_price)
gas_price = np.array(gas_price)
carbon_tax_price = np.array(carbon_tax_price)

plant_earnings = [(price * plant_output * time_step_hours) for price in charge_price[1:]]
gas_price_1 = [(((price / 29.30711) * 10) / efficiency) for price in gas_price[1:]]
gas_price_full = [((((price / 29.30711) * 10) / efficiency) * plant_output * time_step_hours) for price in gas_price[1:]]
carbon_tax_price_1 = [(((price + cpf) * fuel_co2) / efficiency) for price in carbon_tax_price[1:]]
carbon_tax_price_full = [((((price + cpf) * fuel_co2) / efficiency) * plant_output * time_step_hours) for price in carbon_tax_price[1:]]
charge_limit = [gas + tax for gas, tax in zip(gas_price_1, carbon_tax_price_1)]

min_discharge = max(charge_limit)
max_charge = min(charge_limit)
sum_charge = sum(gas_price_1)

min_charge = np.min(charge_price)
max_discharge = np.max(discharge_price)

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)

charge_triggers = np.arange(int(min_charge), int(max_charge))
discharge_triggers = np.arange(int(P_gas + P_tax),int(((P_gas + P_tax) + 100)))

best_charge_trigger = 0
best_discharge_trigger = 0
best_UHTS_profit = -1
best_UHTS_spent = -1
best_UHTS_earned = -1
best_cycles_1 = -1
best_pure_gas_energy_in = -1
best_pure_gas_cost = -1
best_pure_gas_income = -1
best_pure_gas_profit = -1
best_MWh_boost = -1
best_boost_percentage = -1
best_total_carbon_displaced = -1
best_total_carbon_savings = -1
best_system_value = -1
best_MWh_from_UHTS = -1
best_discharge_value = -1
best_total_store_discharge_income = -1
best_total_store_charge_cost = -1
best_gas_supplement = -1
best_replacement = -1
best_total_gas_supplement_cost = -1
best_store_air_percentage = 0
best_price_difference = 1000000000000000000000
best_charging_cost = []
best_discharging_income = []
best_constant_discharge_value = []
best_discharge_sum = []
best_total_carbon_displaced = []
best_total_carbon_savings = []
initial_store_air_percentage = 1  # For example, 50% of air is initially assumed to pass through the thermal store
full_charge_initial_store_air_percentage = 1
t_max = 1800 #K
t_min = 933 #K
heat_loss = 0.5 #%
core_mass = 24800000 #kg
stored_energy_max = ((48 + (1.127 * t_max) * core_mass / 1000) / 3600) #kJ
fuel_requirement = plant_output / fuel_cv #kg
boost = 0.05
boost_energy = plant_output * boost
max_exhaust_flow = 765 #kg/s
max_air_in_proportion = 0.9994 #%/100
max_air_in = max_air_in_proportion * max_exhaust_flow #kg/s
air_temp_in = 760 #K
ave_cp_air_in = (((1061.332 - (0.432819 * t_max) + (1.02344E-3 * t_max ** 2) - (6.47474E-7 * t_max ** 3) + (1.3864E-10 * t_max ** 4))+(1061.332 - (0.432819 * air_temp_in) + (1.02344E-3 * air_temp_in ** 2) - (6.47474E-7 * air_temp_in ** 3) + (1.3864E-10 * air_temp_in ** 4)))/2) #kJ/kg.K
q_air_in = max_air_in * ave_cp_air_in * air_temp_in
cp_al_liq = ((((31.75104))+(0.00000003935826*(((t_max/1000)))+(0.00000001786515*(((t_max/1000)**2)))+(0.000000002694171*(((t_max/1000)**3)))-(0.000000005480037/(t_max/1000)**4))))/26.9815386

###INITIAL VARIABLE VALUES###

if ((((plant_output) / efficiency) / ((max_air_in) * (ave_cp_air_in / 1000000))) + air_temp_in) < t_max:
    target_t = (((plant_output) / efficiency) / ((max_air_in) * (ave_cp_air_in / 1000000))) + air_temp_in
else:
    target_t = t_max

# Objective function
def objective(full_charge_store_air_percentage):
    full_charge_store_air = full_charge_store_air_percentage * max_air_in
    full_charge_other_air = (1 - full_charge_store_air_percentage) * max_air_in
    full_charge_outlet_temp = (t_max - air_temp_in) * np.exp(-time_step / (core_mass * cp_al_liq / (efficacy * full_charge_store_air * ave_cp_air_in / 1000))) + air_temp_in
    full_charge_overall_outlet_temp = (((full_charge_store_air * time_step) * (ave_cp_air_in / 1000) * full_charge_outlet_temp) + ((full_charge_other_air * time_step) * (ave_cp_air_in / 1000) * air_temp_in)) / (((full_charge_store_air * time_step) * (ave_cp_air_in / 1000)) + ((full_charge_other_air * time_step) * (ave_cp_air_in / 1000)))
    initial_outlet_temp = full_charge_overall_outlet_temp  
    return abs(full_charge_overall_outlet_temp - target_t), initial_outlet_temp, full_charge_outlet_temp

# Minimize the objective function
full_charge_result = minimize(lambda x: objective(x)[0], [full_charge_initial_store_air_percentage], method='L-BFGS-B', bounds=Bounds([1e-6], [1]))

# Print the optimal store air percentage and initial outlet temperature
full_charge_best_store_air_percentage = full_charge_result.x[0]
_, initial_mixed_outlet_temp, full_charge_outlet_temp = objective(full_charge_best_store_air_percentage)
initial_energy_discharged = ((((full_charge_best_store_air_percentage * max_air_in) * time_step) * (ave_cp_air_in / 1000) * (full_charge_outlet_temp - air_temp_in)) / (3600 * 1000))

# Objective function to find minimum store temp
def objective_t_min(store_temp):
    full_charge_outlet_temp = (store_temp - air_temp_in) * np.exp(-time_step / (core_mass * cp_al_liq / (efficacy * full_charge_initial_store_air_percentage * max_air_in * ave_cp_air_in / 1000))) + air_temp_in
    full_charge_overall_outlet_temp = (((full_charge_initial_store_air_percentage * max_air_in * time_step) * (ave_cp_air_in / 1000) * full_charge_outlet_temp) + ((1 - full_charge_initial_store_air_percentage) * max_air_in * time_step * (ave_cp_air_in / 1000) * air_temp_in)) / (((full_charge_initial_store_air_percentage * max_air_in * time_step) * (ave_cp_air_in / 1000)) + ((1 - full_charge_initial_store_air_percentage) * max_air_in * time_step * (ave_cp_air_in / 1000)))
    return abs(full_charge_overall_outlet_temp - target_t)

# Initial guess for store_temp
initial_store_temp_guess = 933  # Example initial guess, adjust as needed

# Minimize the objective function to find the store_temp
t_min_result = minimize(objective_t_min, initial_store_temp_guess, method='L-BFGS-B', bounds=Bounds([air_temp_in + 1], [t_max]))  # Bounds set from air_temp_in to an upper reasonable limit for store_temp

# Optimal store temperature found by the minimization
t_min_for_outlet_temp = t_min_result.x[0]

t_design = t_min_for_outlet_temp

stored_energy_max = ((48 + (1.127 * t_max) * core_mass / 1000) / 3600) #kJ
stored_energy_min = ((48 + (1.127 * t_min) * core_mass / 1000) / 3600) #kJ
stored_energy = stored_energy_max - stored_energy_min
charge_input = stored_energy_max * 2 #MW

enthalpy_min_to_tm = ((28.0892 * t_min) + (-5.414849 * ((t_min ** 2) / 2)) + (8.560423 * ((t_min ** 3) / 3)) + (3.42737 * ((t_min ** 4) / 4)) - (-0.277375 / t_min) + (-9.147187) - (0)) #kJ/kg
enthalpy_tm_to_max = ((31.75104 * t_max) + (3.935826E-8 * ((t_max ** 2) / 2)) + (1.786515E-8 * (( t_max ** 3) / 3)) + (2.694171E-9 * (( t_max ** 4) / 4)) - (5.480037E-9 / t_max) + (-0.945684) - (10.56201)) #kJ/kg

###UPDATED###

usable_stored_energy = stored_energy_max - stored_energy_min
store_temp = t_max #((((usable_stored_energy + stored_energy_min) * 3600 * 1000) / core_mass) - 48) / 1.127
outlet_temp = full_charge_outlet_temp #(store_temp - air_temp_in) * np.exp(((- time_step)) / (((core_mass * cp_al_liq) / (efficacy * max_air_in * (ave_cp_air_in / 1000))) + air_temp_in))
energy_discharged = initial_energy_discharged #((max_air_in * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)
recharging_required = stored_energy
#carbon_displaced = energy_discharged * fuel_co2

def calculate_temp(usable_stored_energy, store_temp, outlet_temp, energy_discharged):
    # Define the objective function
    def objective(store_air_percentage):
        store_air = store_air_percentage * max_air_in
        other_air = (1 - store_air_percentage) * max_air_in

        # Initial guess for outlet temperature
        overall_outlet_temp = ((((store_air * time_step) * (ave_cp_air_in / 1000) * ((store_temp - air_temp_in) * np.exp(((- time_step)) / (((core_mass * cp_al_liq) / (1e-6 + (efficacy * (initial_store_air_percentage * store_air) * (ave_cp_air_in / 1000))) + air_temp_in)))) + ((other_air * time_step) * (ave_cp_air_in / 1000) * air_temp_in)) / (((store_air * time_step) * (ave_cp_air_in / 1000)) + ((other_air * time_step) * (ave_cp_air_in / 1000)))))

        # Calculate new parameters using the current outlet temperature guess
        if usable_stored_energy > energy_discharged:
            usable_stored_energy_new = usable_stored_energy - (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)) - ((usable_stored_energy + stored_energy_min) * ((heat_loss / 100) / (24 / (time_step / 3600))))
        elif usable_stored_energy <= energy_discharged and usable_stored_energy > 0:
            usable_stored_energy_new = 0
        else:
            usable_stored_energy_new = usable_stored_energy - ((usable_stored_energy + stored_energy_min) * ((heat_loss / 100) / (24 / (time_step / 3600))))
        store_temp_new = ((((usable_stored_energy_new + stored_energy_min) * 3600) - 48) * 1000) / (core_mass * 1.127)
        outlet_temp_new = ((store_temp_new - air_temp_in) * (np.exp((- time_step) / ((core_mass * cp_al_liq) / (efficacy * store_air * (ave_cp_air_in / 1000))))) + air_temp_in)           
        if usable_stored_energy >= (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)):
            energy_discharged_new = (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000))
        elif usable_stored_energy > 0 and usable_stored_energy < (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)):
            energy_discharged_new = usable_stored_energy
        else:
            energy_discharged_new = 0
        overall_outlet_temp_new = ((((store_air * time_step) * (ave_cp_air_in / 1000) * outlet_temp_new) + ((other_air * time_step) * (ave_cp_air_in / 1000) * air_temp_in)) / (((store_air * time_step) * (ave_cp_air_in / 1000)) + ((other_air * time_step) * (ave_cp_air_in / 1000))))

        # We want to minimize the absolute value of the difference between outlet_temp_new and target_t
        return abs(overall_outlet_temp_new - target_t)

    # Initial guess for store_air_percentage
    initial_guess = [initial_store_air_percentage]
    bounds = Bounds([1e-6], [1])
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    best_store_air_percentage = result.x[0]

    # Recalculate the parameters with the optimal store_air_percentage
    store_air = best_store_air_percentage * max_air_in
    other_air = (1 - best_store_air_percentage) * max_air_in
    if usable_stored_energy > energy_discharged:
        usable_stored_energy_new = usable_stored_energy - (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)) - ((usable_stored_energy + stored_energy_min) * ((heat_loss / 100) / (24 / (time_step / 3600))))
    elif usable_stored_energy <= energy_discharged and usable_stored_energy > 0:
        usable_stored_energy_new = 0
    else:
        usable_stored_energy_new = usable_stored_energy - ((usable_stored_energy + stored_energy_min) * ((heat_loss / 100) / (24 / (time_step / 3600))))
    store_temp_new = ((((usable_stored_energy_new + stored_energy_min) * 3600) - 48) * 1000) / (core_mass * 1.127)
    outlet_temp_new = ((store_temp_new - air_temp_in) * (np.exp((- time_step) / ((core_mass * cp_al_liq) / (efficacy * store_air * (ave_cp_air_in / 1000))))) + air_temp_in)           
    if usable_stored_energy >= (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)):
        energy_discharged_new = (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000))
    elif usable_stored_energy > 0 and usable_stored_energy < (((store_air * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)):
        energy_discharged_new = usable_stored_energy
    else:
        energy_discharged_new = 0
    overall_outlet_temp = ((((store_air * time_step) * (ave_cp_air_in / 1000) * outlet_temp_new) + ((other_air * time_step) * (ave_cp_air_in / 1000) * air_temp_in)) / (((store_air * time_step) * (ave_cp_air_in / 1000)) + ((other_air * time_step) * (ave_cp_air_in / 1000))))
    additional_energy = (((max_air_in * time_step) * (ave_cp_air_in / 1000) * (overall_outlet_temp - air_temp_in)) / (3600 * 1000)) - energy_discharged_new

    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new, best_store_air_percentage 
 
def reset_temp_without_discharge(usable_stored_energy, store_temp, outlet_temp, energy_discharged):
    usable_stored_energy_new = stored_energy_max - stored_energy_min
    store_temp_new = t_max
    outlet_temp_new = full_charge_outlet_temp #K
    energy_discharged_new =  initial_energy_discharged #* discharge_values[i]

    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new #TUPLE
    
def calculate_recharge(usable_stored_energy, store_temp, outlet_temp, energy_discharged, recharging_required):
    usable_stored_energy_new = usable_stored_energy 
    store_temp_new = store_temp
    outlet_temp_new = outlet_temp
    energy_discharged_new = 0
    recharging_required_new = stored_energy - usable_stored_energy
    
    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new, recharging_required_new

def skip(usable_stored_energy, store_temp, outlet_temp, energy_discharged):
    usable_stored_energy_new = usable_stored_energy - ((usable_stored_energy + stored_energy_min) * ((heat_loss / 100) / (24 / (time_step / 3600))))
    store_temp_new = ((((usable_stored_energy_new + stored_energy_min) * 3600) - 48) * 1000) / (core_mass * 1.127)
    outlet_temp_new = outlet_temp          
    energy_discharged_new = 0
    
    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new

for charge_trigger in charge_triggers:
    for discharge_trigger in discharge_triggers:
        if charge_trigger >= discharge_trigger:
            continue
    
        charge_points = [None] * len(charge_price)
        discharge_points = [None] * len(discharge_price)
        charging_cost = [None] * len(charge_price)
        discharging_income = [None] * len(discharge_price)		
        prices_when_charging = [None] * len(charge_price)
        prices_when_discharging = [None] * len(discharge_price)
        energy_into_store_per_timestep = [None] * len(charge_price)
        energy_from_store_per_timestep = [None] * len(discharge_price)
        usable_stored_energy_per_timestep= [None] * len(discharge_price)
        store_temp_values_per_timestep = [None] * len(discharge_price)
        temp_values_per_timestep = [None] * len(discharge_price)
        store_air_percentage_per_timestep = [None] * len(discharge_price)
        overall_outlet_temp_per_timestep = [None] * len(discharge_price)
        price_when_charging_array = [None] * len(charge_price)		
        charge_summ = [None] * len(charge_price)
        price_when_discharging_array = [None] * len(discharge_price)		
        discharge_summ = [None] * len(discharge_price)
        charge_2 = [None] * len(charge_price)
        discharge_2 = [None] * len(discharge_price)
        discharge_volumes_test = [None] * len(discharge_price)
        dynamic_discharge_value = [None] * len(discharge_price)
        variable_discharge_values = [None] * len(discharge_price)
        energy_charged = [None] * len(discharge_price)
        test_volumes = [None] * len(discharge_price)
        gas_supplement_per_timestep = [None] * len(discharge_price)
        gas_supplement_cost_per_timestep = [None] * len(discharge_price)
        carbon_displaced_per_timestep = [None] * len(discharge_price)
        carbon_savings_per_timestep = [None] * len(discharge_price)

        for i in range(len(charge_price)):
            if charge_price[i] <= charge_trigger:
                price_when_charging_array[i] = charge_price[i]
                charge_summ[i] = charge_price[i]

        for i in range(len(discharge_price)):
            if discharge_price[i] >= charge_trigger:
                price_when_discharging_array[i] = discharge_price[i]
                discharge_summ[i] = discharge_price[i]
                                
        for i in range(len(charge_price)):
            if charge_price[i] <= charge_trigger:
                charge_points[i] = charge_price[i]
                
        for i in range(len(discharge_price)):
            if discharge_price[i] >= discharge_trigger:
                discharge_points[i] = discharge_price[i]

        charge_total_sum = list(charge_summ)

        for i in range(1, len(charge_total_sum)):		
            charge_1 = charge_total_sum[i]
            
            if charge_1 != None:
                charge_2[i] = charge_1

        discharge_total_sum = list(discharge_summ)

        for i in range(1, len(discharge_total_sum)):		
            discharge_1 = discharge_total_sum[i]
            
            if discharge_1 != None:
                discharge_2[i] = discharge_1

        combined = list(zip(charge_points, discharge_points))
                                           
        usable_stored_energy = stored_energy_max - stored_energy_min
        store_temp = t_max #((((usable_stored_energy + stored_energy_min) * 3600 * 1000) / core_mass) - 48) / 1.127
        outlet_temp = full_charge_outlet_temp #(store_temp - air_temp_in) * np.exp(((- time_step)) / (((core_mass * cp_al_liq) / (efficacy * max_air_in * (ave_cp_air_in / 1000))) + air_temp_in))
        energy_discharged = 0 #((max_air_in * time_step) * (ave_cp_air_in / 1000) * (outlet_temp - air_temp_in)) / (3600 * 1000)
        recharging_required = 0
        
        half_hours = -1
        for i in range(1, len(combined)):		
            values = combined[i]

            if values[0] == None and values[1] == None:

                usable_stored_energy, store_temp, outlet_temp, energy_discharged = skip(usable_stored_energy, store_temp, outlet_temp, energy_discharged)
                energy_from_store_per_timestep[i] = 0
                energy_into_store_per_timestep[i] = 0
                usable_stored_energy_per_timestep[i] = usable_stored_energy
                store_temp_values_per_timestep[i] = store_temp
                temp_values_per_timestep[i] = outlet_temp
                gas_supplement_per_timestep[i] = (((max_air_in * time_step) * (ave_cp_air_in / 1000) * (target_t - air_temp_in)) / (3600 * 1000)) - energy_from_store_per_timestep[i]
                gas_supplement_cost_per_timestep[i] = gas_supplement_per_timestep[i] * ((gas_prices[i] / 29.30711) * 10)
                carbon_displaced_per_timestep[i] = energy_from_store_per_timestep[i] * fuel_co2
                carbon_savings_per_timestep[i] = carbon_displaced_per_timestep[i] * carbon_tax_prices[i]
                store_air_percentage_per_timestep[i] = 0
                overall_outlet_temp_per_timestep[i] = 0
                                            
            if values[0] != None and values[1] == None:
            
                usable_stored_energy, store_temp, outlet_temp, energy_discharged, recharging_required = calculate_recharge(usable_stored_energy, store_temp, outlet_temp, energy_discharged, recharging_required)
                if recharging_required > initial_energy_discharged:
                    energy_into_store_per_timestep[i] = recharging_required
                else:
                    energy_into_store_per_timestep[i] = (stored_energy - usable_stored_energy) #+ initial_energy_discharged

                usable_stored_energy, store_temp, outlet_temp, energy_discharged = reset_temp_without_discharge(usable_stored_energy, t_max, outlet_temp, energy_discharged)
                prices_when_charging[i] = values[0]
                energy_from_store_per_timestep[i] = 0
                usable_stored_energy_per_timestep[i] = stored_energy
                store_temp_values_per_timestep[i] = store_temp
                temp_values_per_timestep[i] = outlet_temp
                gas_supplement_per_timestep[i] = (((max_air_in * time_step) * (ave_cp_air_in / 1000) * (target_t - air_temp_in)) / (3600 * 1000)) - energy_from_store_per_timestep[i]
                gas_supplement_cost_per_timestep[i] = gas_supplement_per_timestep[i] * ((gas_prices[i] / 29.30711) * 10)
                carbon_displaced_per_timestep[i] = energy_from_store_per_timestep[i] * fuel_co2
                carbon_savings_per_timestep[i] = carbon_displaced_per_timestep[i] * carbon_tax_prices[i]
                store_air_percentage_per_timestep[i] = 0
                overall_outlet_temp_per_timestep[i] = 0
                test_volumes[i] = usable_stored_energy

            if values[1] != None and values[0] == None:

                    usable_stored_energy, store_temp, outlet_temp, energy_discharged, best_store_air_percentage = calculate_temp(usable_stored_energy, store_temp, outlet_temp, energy_discharged) #, plant_output, efficiency, efficacy, stored_energy_min, core_mass, ave_cp_air_in, initial_store_air_percentage, target_t)
                    prices_when_discharging[i] = values[1]
                    energy_from_store_per_timestep[i] = energy_discharged
                    usable_stored_energy_per_timestep[i] = usable_stored_energy
                    store_temp_values_per_timestep[i] = store_temp
                    temp_values_per_timestep[i] = outlet_temp
                    store_air_percentage_per_timestep[i] = best_store_air_percentage
                    overall_outlet_temp_per_timestep[i] = (((((store_air_percentage_per_timestep[i] * max_air_in) * time_step) * (ave_cp_air_in / 1000) * outlet_temp) + ((((1-store_air_percentage_per_timestep[i]) * max_air_in) * time_step) * (ave_cp_air_in / 1000) * air_temp_in)) / ((((store_air_percentage_per_timestep[i] * max_air_in) * time_step) * (ave_cp_air_in / 1000)) + ((((1 - store_air_percentage_per_timestep[i]) * max_air_in) * time_step) * (ave_cp_air_in / 1000))))

                    if ((((max_air_in * time_step) * (ave_cp_air_in / 1000) * (target_t - air_temp_in)) / (3600 * 1000)) - energy_from_store_per_timestep[i]) < 1e-3:
                        gas_supplement_per_timestep[i] = 0
                    else:
                        gas_supplement_per_timestep[i] = (((max_air_in * time_step) * (ave_cp_air_in / 1000) * (target_t - air_temp_in)) / (3600 * 1000)) - energy_from_store_per_timestep[i]
                    gas_supplement_cost_per_timestep[i] = gas_supplement_per_timestep[i] * ((gas_prices[i] / 29.30711) * 10)
                    carbon_displaced_per_timestep[i] = energy_from_store_per_timestep[i] * fuel_co2
                    carbon_savings_per_timestep[i] = carbon_displaced_per_timestep[i] * carbon_tax_prices[i]
                    energy_into_store_per_timestep[i] = 0
                    test_volumes[i] = usable_stored_energy

            if values[0] == None and values[1] == None:

                    usable_stored_energy, store_temp, outlet_temp, energy_discharged, best_store_air_percentage = calculate_temp(usable_stored_energy, store_temp, outlet_temp, energy_discharged) #, plant_output, efficiency, efficacy, stored_energy_min, core_mass, ave_cp_air_in, initial_store_air_percentage, target_t)
                    prices_when_discharging[i] = values[1]
                    energy_from_store_per_timestep[i] = energy_discharged
                    usable_stored_energy_per_timestep[i] = usable_stored_energy
                    store_temp_values_per_timestep[i] = store_temp
                    temp_values_per_timestep[i] = outlet_temp
                    store_air_percentage_per_timestep[i] = best_store_air_percentage
                    overall_outlet_temp_per_timestep[i] = (((((store_air_percentage_per_timestep[i] * max_air_in) * time_step) * (ave_cp_air_in / 1000) * outlet_temp) + ((((1-store_air_percentage_per_timestep[i]) * max_air_in) * time_step) * (ave_cp_air_in / 1000) * air_temp_in)) / ((((store_air_percentage_per_timestep[i] * max_air_in) * time_step) * (ave_cp_air_in / 1000)) + ((((1 - store_air_percentage_per_timestep[i]) * max_air_in) * time_step) * (ave_cp_air_in / 1000))))

                    if ((((max_air_in * time_step) * (ave_cp_air_in / 1000) * (target_t - air_temp_in)) / (3600 * 1000)) - energy_from_store_per_timestep[i]) < 1e-3:
                        gas_supplement_per_timestep[i] = 0
                    else:
                        gas_supplement_per_timestep[i] = (((max_air_in * time_step) * (ave_cp_air_in / 1000) * (target_t - air_temp_in)) / (3600 * 1000)) - energy_from_store_per_timestep[i]
                    gas_supplement_cost_per_timestep[i] = gas_supplement_per_timestep[i] * ((gas_prices[i] / 29.30711) * 10)
                    carbon_displaced_per_timestep[i] = energy_from_store_per_timestep[i] * fuel_co2
                    carbon_savings_per_timestep[i] = carbon_displaced_per_timestep[i] * carbon_tax_prices[i]
                    energy_into_store_per_timestep[i] = 0
                    test_volumes[i] = usable_stored_energy

            variable_discharge = range(0, len(energy_from_store_per_timestep))
            variable_charge = range(0, len(energy_from_store_per_timestep))

            combined_variable = list(zip(prices_when_charging, energy_into_store_per_timestep, prices_when_discharging, energy_from_store_per_timestep))
            for i in range(1, len(combined_variable)):		
                combined_values = combined_variable[i]

                if combined_values[2] != None:
                    discharging_income[i] =  combined_values[2] * combined_values[3]
                    charging_cost[i] =  0
                else:
                    if combined_values[2] != None:
                        discharging_income[i] = combined_values[2] * initial_energy_discharged
                    else:
                        discharging_income[i] = 0
                        
                if combined_values[0] != None:
                    discharging_income[i] =  combined_values[0] * initial_energy_discharged
                    charging_cost[i] =  combined_values[0] * combined_values[1]
                else:
                    pass

            half_hours += 1
      
        UHTS_spent = 0
        UHTS_earned = 0
        charge_value = 0
        discharge_value = 0
        energy_into_store_total = 0
        energy_from_store_total = 0
        charge_sum = 0
        discharge_sum = 0
        total_store_discharge_income = 0
        total_store_charge_cost = 0
        energy_from_store_total = 0
        total_gas_supplement = 0
        total_gas_supplement_cost = 0
        total_carbon_displaced = 0
        total_carbon_savings = 0

        for c in charging_cost:
            if c is not None:
                UHTS_spent += c
                
        for d in discharging_income:
            if d is not None:
                UHTS_earned += d 
                
        for c in prices_when_charging:
            if c is not None:
                charge_value += c
                
        for d in prices_when_discharging:
            if d is not None:
                discharge_value += d 
                
        for c in energy_into_store_per_timestep:
            if c is not None:
                energy_into_store_total += c 
                
        for d in energy_from_store_per_timestep:
            if d is not None:
                energy_from_store_total += d 
                
        for c in price_when_charging_array:
            if c is not None:
                charge_sum += c 
                    
        for d in price_when_discharging_array:
            if d is not None:
                discharge_sum += d 
                
        for d in discharging_income:
            if d is not None:
                total_store_discharge_income += d 
                
        for c in charging_cost:
            if c is not None:
                total_store_charge_cost += c

        for d in gas_supplement_per_timestep:
            if d is not None:
                total_gas_supplement += d

        for d in gas_supplement_cost_per_timestep:
            if d is not None:
                total_gas_supplement_cost += d

        for d in carbon_displaced_per_timestep:
            if d is not None:
                total_carbon_displaced += d
                        
        for d in carbon_savings_per_timestep:
            if d is not None:
                total_carbon_savings += d    
                    
        UHTS_profit = (UHTS_earned - UHTS_spent)
        MWh_from_UHTS = energy_from_store_total
        output_profit = ((UHTS_earned * (stored_energy_max / stored_energy)) - UHTS_spent)
        cycles_1 = (energy_into_store_total / stored_energy)
        pure_gas_energy_in = (plant_output / efficiency) * (SPs * (time_step / 3600))
        pure_gas_cost = sum(gas_price_full) 
        pure_gas_income = sum(plant_earnings)
        pure_gas_profit = sum(plant_earnings) - pure_gas_cost
        pure_gas_carbon_cost = sum(carbon_tax_price_full) 
        replacement = (MWh_from_UHTS / pure_gas_energy_in) * 100
        UHTS_operation_cost = total_gas_supplement_cost + UHTS_spent
        price_difference = pure_gas_cost - UHTS_operation_cost
        
        if price_difference > 0:
            if price_difference < best_price_difference:
                if replacement > best_replacement:
                    best_UHTS_profit = UHTS_profit
                    best_output_profit = output_profit
                    best_UHTS_spent = UHTS_spent
                    best_UHTS_earned = UHTS_earned
                    best_charge_trigger = charge_trigger
                    best_discharge_trigger = discharge_trigger
                    best_charging_cost = charging_cost
                    best_discharging_income = discharging_income
                    best_charge_value = charge_value
                    best_discharge_value = discharge_value
                    best_MWh_from_UHTS = MWh_from_UHTS
                    best_total_store_discharge_income = total_store_discharge_income
                    best_total_store_charge_cost = total_store_charge_cost
                    best_cycles_1 = round(cycles_1)
                    best_pure_gas_energy_in = pure_gas_energy_in
                    best_pure_gas_cost = pure_gas_cost
                    best_pure_gas_income = pure_gas_income
                    best_total_gas_supplement = total_gas_supplement
                    best_total_gas_supplement_cost = total_gas_supplement_cost
                    best_replacement = replacement
                    best_total_carbon_displaced = total_carbon_displaced
                    best_total_carbon_savings = total_carbon_savings
                    best_price_difference = price_difference

###PRINTING TO PYTHON###

float = P_gas
calc_gas_cost = "{:.2f}".format(float)
print("Ave Fuel Price: £", calc_gas_cost)

float = max_charge
calc_gas_and_tax_cost = "{:.2f}".format(float)
print("Fuel Price w/ Tax: £", calc_gas_and_tax_cost)

print("Best Charge Trigger: ", best_charge_trigger)
print("Best Discharge Trigger: ", best_discharge_trigger)

float = best_pure_gas_energy_in
format_mwh_in = "{:.2f}".format(float)
print("Required MWh: ", format_mwh_in)

float = best_MWh_from_UHTS
format_uhts_out = "{:.2f}".format(float)
print("Displaced MWh: ", format_uhts_out)

float = best_replacement
format_replacement = "{:.2f}".format(float)
print("Fuel Replacement:", format_replacement,"%")

float = best_UHTS_spent
format_charge_cost = "{:.2f}".format(float)
print("Store Charge Cost: £", format_charge_cost)

float = best_total_gas_supplement_cost
format_uhts_gas = "{:.2f}".format(float)
print("Supplement Gas Cost: £", format_uhts_gas)

float = best_pure_gas_cost
format_standard_gas = "{:.2f}".format(float)
print("Pure Gas Cost: £", format_standard_gas)

float = (best_UHTS_spent + best_total_gas_supplement_cost)
format_uhts_total = "{:.2f}".format(float)
print("Gas/Store Operation Cost: £", format_uhts_total)

float = pure_gas_carbon_cost
format_standard_carbon = "{:.2f}".format(float)
print("Gas Carbon Tax: £", format_standard_carbon)

float = best_total_carbon_savings
format_uhts_carbon = "{:.2f}".format(float)
print("UHTS Carbon Tax: £", format_uhts_carbon)

float = (best_total_carbon_displaced)
format_carbon_displaced = "{:.2f}".format(float)
print("Carbon Displaced: ", format_carbon_displaced)

df.to_excel(save_file[0], index=False)

stop = timeit.default_timer()

float = stop - start
format_time = "{:.2f}".format(float)
print("Time: ", format_time)
print("",)

# Assuming 'save_file_name' holds the name of the file you want to save to
with open(text_save_file[0], "w") as file:
    # Writing each line to the file
    file.write(f"Ave Fuel Price: £ {calc_gas_cost}\n")
    file.write(f"Fuel Price w/ Tax: £ {calc_gas_and_tax_cost}\n")
    file.write(f"Best Charge Trigger: {best_charge_trigger}\n")
    file.write(f"Best Discharge Trigger: {best_discharge_trigger}\n")
    file.write(f"Required MWh: {format_mwh_in}\n")
    file.write(f"Displaced MWh: {format_uhts_out}\n")
    file.write(f"Fuel Replacement: {format_replacement}%\n")
    file.write(f"Store Charge Cost: £ {format_charge_cost}\n")
    file.write(f"Supplement Gas Cost: £ {format_uhts_gas}\n")
    file.write(f"Pure Gas Cost: £ {format_standard_gas}\n")
    file.write(f"Gas/Store Operation Cost: £ {format_uhts_total}\n")
    file.write(f"Gas Carbon Tax: £ {format_standard_carbon}\n")
    file.write(f"UHTS Carbon Tax: £ {format_uhts_carbon}\n")
    file.write(f"Carbon Displaced: £ {format_carbon_displaced}\n")
    file.write(f"Time: {format_time}\n")

