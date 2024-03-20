
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
original_file_name = load_file[0]
base_name, _ = original_file_name.rsplit('.', 1)  # This removes the extension and keeps the base file name
text_file_name = base_name + '.txt'  # Add the '.txt' extension to the base file name
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
discharge_triggers = np.arange(int(P_gas + P_tax),int(min(max_discharge, 250)))

best_charge_trigger = 0
best_discharge_trigger = 0
best_UHTS_profit = -1
best_UHTS_spent = -1
best_UHTS_earned = -1
best_tax_savings = -1
best_fuel_savings = -1
best_savings = -1
best_cycles_1 = -1
best_pure_gas_energy_in = -1
best_pure_gas_cost = -1
best_pure_gas_income = -1
best_pure_gas_profit = -1
best_MWh_boost = -1
best_boost_percentage = -1
best_carbon_displaced = -1
best_carbon_savings = -1
best_system_value = -1
best_store_air_percentage = 0
best_charge_totals = []
best_discharge_totals = []
best_constant_discharge_value = []
best_discharge_sum = []
#best_carbon_displaced = []
best_carbon_savings = []
initial_store_air_percentage = 0.5  # Adjust this value as needed
t_max = 1800 #K
t_min = 933 #K
heat_loss = 0.5 #%
core_mass = 24800000 #kg
t_store_max = 1800
stored_energy_max = ((48 + (1.127 * t_max) * core_mass / 1000) / 3600) #kJ
fuel_requirement = plant_output / fuel_cv #kg
boost = 0.05
flow_increase = 0.1
boost_energy = plant_output * boost
max_exhaust_flow = 765 #kg/s
#max_air_in_proportion = 0.9994 #%/100
max_air_in = flow_increase * max_exhaust_flow #kg/s
air_temp_in = 760 #K
ave_cp_air_in = (((1061.332 - (0.432819 * t_store_max) + (1.02344E-3 * t_store_max ** 2) - (6.47474E-7 * t_store_max ** 3) + (1.3864E-10 * t_store_max ** 4))+(1061.332 - (0.432819 * air_temp_in) + (1.02344E-3 * air_temp_in ** 2) - (6.47474E-7 * air_temp_in ** 3) + (1.3864E-10 * air_temp_in ** 4)))/2) #kJ/kg.K
q_air_in = max_air_in * ave_cp_air_in * air_temp_in
cp_al_liq = ((((31.75104))+(0.00000003935826*(((t_store_max/1000)))+(0.00000001786515*(((t_store_max/1000)**2)))+(0.000000002694171*(((t_store_max/1000)**3)))-(0.000000005480037/(t_store_max/1000)**4))))/26.9815386
min_outlet_temp = (((boost_energy / efficiency) * time_step_hours) / (max_air_in * (ave_cp_air_in / 1000000))) + air_temp_in
t_min = (min_outlet_temp - air_temp_in) / (math.exp((- time_step_hours) / ((core_mass * cp_al_liq) / (efficacy * (max_air_in) * (ave_cp_air_in / 1000000))))) + air_temp_in
stored_energy_max = ((48 + (1.127 * t_max) * core_mass / 1000) / 3600) #kJ
stored_energy_min = ((48 + (1.127 * t_min) * core_mass / 1000) / 3600) #kJ
stored_energy = stored_energy_max - stored_energy_min
charge_input = stored_energy_max * 2 #MW

enthalpy_min_to_tm = ((28.0892 * t_min) + (-5.414849 * ((t_min ** 2) / 2)) + (8.560423 * ((t_min ** 3) / 3)) + (3.42737 * ((t_min ** 4) / 4)) - (-0.277375 / t_min) + (-9.147187) - (0)) #kJ/kg
enthalpy_tm_to_max = ((31.75104 * t_max) + (3.935826E-8 * ((t_max ** 2) / 2)) + (1.786515E-8 * (( t_max ** 3) / 3)) + (2.694171E-9 * (( t_max ** 4) / 4)) - (5.480037E-9 / t_max) + (-0.945684) - (10.56201)) #kJ/kg

###UPDATED###

usable_stored_energy = stored_energy_max - stored_energy_min
initial_outlet_temp =  ((t_store_max - air_temp_in) * (math.exp((- time_step) / ((core_mass * cp_al_liq) / (efficacy * max_air_in * (ave_cp_air_in / 1000))))) + air_temp_in) #K
initial_energy_discharged = (plant_output * boost) * (time_step_hours)
store_temp = ((((usable_stored_energy + stored_energy_min) * 3600 * 1000) / core_mass) - 48) / 1.127
outlet_temp = (store_temp - air_temp_in) * math.exp(((- time_step)) / (((core_mass * cp_al_liq) / (efficacy * max_air_in * (ave_cp_air_in / 1000))) + air_temp_in))
energy_discharged = (boost_energy / efficiency) * (time_step_hours)
carbon_displaced = energy_discharged * fuel_co2
recharging_required = stored_energy - usable_stored_energy
thermal_loss = ((usable_stored_energy + stored_energy_min) * ((heat_loss / 100) / (daily_time_steps / (time_step_hours))))

def calculate_temp(usable_stored_energy, store_temp, air_temp_in, time_step_hours, plant_output, boost, efficiency, efficacy, thermal_loss, stored_energy_min, core_mass, ave_cp_air_in, boost_energy, initial_store_air_percentage):
    # Define the objective function
    def objective(store_air_percentage):
        store_air = store_air_percentage * max_air_in
        other_air = (1 - store_air_percentage) * max_air_in

        # Initial guess for outlet temperature
        outlet_temp_new = ((store_temp - air_temp_in) * (np.exp((- time_step_hours) / ((core_mass * cp_al_liq) / (efficacy * store_air * (ave_cp_air_in / 1000000) + 1e-6)))) + air_temp_in)

        # Calculate new parameters using the current outlet temperature guess
        usable_stored_energy_new = usable_stored_energy - (((plant_output * boost) / (efficiency)) * (time_step_hours)) - thermal_loss
        store_temp_new = ((((usable_stored_energy_new + stored_energy_min) * 3600 * 1000) / core_mass) - 48) / 1.127
        outlet_temp_new = ((store_temp_new - air_temp_in) * (np.exp((- time_step_hours) / ((core_mass * cp_al_liq) / (efficacy * (max_air_in * store_air_percentage) * (ave_cp_air_in / 1000000) + 1e-6)))) + air_temp_in)           
        energy_discharged_new = (boost_energy / efficiency) * (time_step_hours)
        carbon_displaced = energy_discharged_new * fuel_co2

        # Calculate the energy difference
        current_energy_increase = store_air * (ave_cp_air_in / 1000000) * (outlet_temp_new - air_temp_in)
        energy_difference = current_energy_increase - (boost_energy / efficiency)

        # We want to minimize the absolute value of energy_difference
        return abs(energy_difference)

    # Initial guess for store_air_percentage
    initial_guess = [initial_store_air_percentage]

    # Call the minimize function
    result = minimize(objective, initial_guess, method='Nelder-Mead')

    # The optimal store_air_percentage is stored in result.x
    best_store_air_percentage = result.x[0]

    # Recalculate the parameters with the optimal store_air_percentage
    store_air = best_store_air_percentage * max_air_in
    other_air = (1 - best_store_air_percentage) * max_air_in
    usable_stored_energy_new = usable_stored_energy - (((plant_output * boost) / (efficiency)) * (time_step_hours)) - thermal_loss
    store_temp_new = ((((usable_stored_energy_new + stored_energy_min) * 3600 * 1000) / core_mass) - 48) / 1.127
    outlet_temp_new = ((store_temp_new - air_temp_in) * (math.exp((- time_step_hours) / ((core_mass * cp_al_liq) / (efficacy * (max_air_in * best_store_air_percentage) * (ave_cp_air_in / 1000000) + 1e-6)))) + air_temp_in)           
    energy_discharged_new = (boost_energy / efficiency) * (time_step_hours)
    current_energy_increase = store_air * (ave_cp_air_in / 1000000) * (outlet_temp_new - air_temp_in)
    best_energy_difference = current_energy_increase - (boost_energy / efficiency)
    carbon_displaced = energy_discharged_new * fuel_co2

    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new, best_store_air_percentage, best_energy_difference, carbon_displaced

def calculate_loss(usable_stored_energy, store_temp, outlet_temp, energy_discharged):
    usable_stored_energy_new = usable_stored_energy - thermal_loss
    store_temp_new = ((((usable_stored_energy_new + stored_energy_min) * 3600 * 1000) / core_mass) - 48) / 1.127
    outlet_temp_new = ((store_temp_new - air_temp_in) * (math.exp((- time_step_hours) / ((core_mass * cp_al_liq) / (efficacy * (max_air_in * best_store_air_percentage) * (ave_cp_air_in / 1000000) + 1e-6)))) + air_temp_in)           
    energy_discharged_new = 0

    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new #TUPLE

def reset_temp(usable_stored_energy, store_temp, outlet_temp, energy_discharged):
    usable_stored_energy_new = stored_energy_max - stored_energy_min
    store_temp_new = t_store_max
    outlet_temp_new = ((store_temp_new - air_temp_in) * (math.exp((- time_step_hours) / ((core_mass * cp_al_liq) / (efficacy * (max_air_in * best_store_air_percentage) * (ave_cp_air_in / 1000000) + 1e-6)))) + air_temp_in)           
    energy_discharged_new = 0

    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new #TUPLE
    
def calculate_recharge(usable_stored_energy, store_temp, outlet_temp, energy_discharged, recharging_required):
    usable_stored_energy_new = usable_stored_energy 
    store_temp_new = store_temp
    outlet_temp_new = outlet_temp
    energy_discharged_new = 0
    recharging_required_new = stored_energy - usable_stored_energy + thermal_loss
    
    return usable_stored_energy_new, store_temp_new, outlet_temp_new, energy_discharged_new, recharging_required_new

for charge_trigger in charge_triggers:
    for discharge_trigger in discharge_triggers:
        if charge_trigger >= discharge_trigger:
            continue
    
        charge_points = [None] * len(charge_price)
        discharge_points = [None] * len(discharge_price)
        charge_totals = [None] * len(charge_price)
        discharge_totals = [None] * len(discharge_price)		
        prices_when_charging = [None] * len(charge_price)
        price_when_discharging = [None] * len(discharge_price)
        energy_into_store_per_timestep = [None] * len(charge_price)
        energy_from_store_per_timestep = [None] * len(discharge_price)
        price_when_charging_array = np.where(charge_price <= charge_trigger, charge_price, None)
        charge_summ = np.where(charge_price <= charge_trigger, charge_price, None)
        price_when_discharging_array = np.where(discharge_price >= charge_trigger, discharge_price, None)
        discharge_summ = np.where(discharge_price >= charge_trigger, discharge_price, None)
        charge_2 = [None] * len(charge_price)
        discharge_2 = [None] * len(discharge_price)
        discharge_volumes_test = [None] * len(discharge_price)
        dynamic_discharge_value = [None] * len(discharge_price)
        charging_cost = [None] * len(charge_price)
        discharging_income = [None] * len(discharge_price)
        variable_discharge_values = [None] * len(discharge_price)
        energy_charged = [None] * len(discharge_price)
        constant_discharge_income = [None] * len(discharge_price)		
        test_volumes = [None] * len(discharge_price)
        carbon_displaced_per_timestep = [None] * len(discharge_price)
        carbon_savings = [None] * len(discharge_price)

        price_when_charging_array = np.where(charge_price <= charge_trigger, charge_price, None)
        charge_summ = np.where(charge_price <= charge_trigger, charge_price, None)
                                
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

        combined = list(zip(charge_points, discharge_points, discharge_price))
                                           
        usable_stored_energy = stored_energy
        store_temp = t_store_max
        outlet_temp = outlet_temp
        energy_discharged = 0
        recharging_required = 0
        
        half_hours = -1

        for i in range(1, len(combined)):		
            values = combined[i]
            
            constant_discharge_income[i] = values[2] * plant_output * (time_step_hours)
 
            if values[0] == None and values[1] == None:
                usable_stored_energy, store_temp, outlet_temp, energy_discharged = calculate_loss(usable_stored_energy, store_temp, outlet_temp, energy_discharged)
                test_volumes[i] = usable_stored_energy
                carbon_displaced_per_timestep[i] = 0
                carbon_savings[i] = 0

            if values[0] != None:
                usable_stored_energy, store_temp, outlet_temp, energy_discharged, recharging_required = calculate_recharge(usable_stored_energy, store_temp, outlet_temp, energy_discharged, recharging_required)
                energy_into_store_per_timestep[i] = recharging_required
                prices_when_charging[i] = values[0]

                usable_stored_energy, store_temp, outlet_temp, energy_discharged = reset_temp(usable_stored_energy, t_store_max, outlet_temp, energy_discharged)
                test_volumes[i] = usable_stored_energy
                carbon_displaced_per_timestep[i] = 0
                carbon_savings[i] = 0

            if values[1] != None:
                usable_stored_energy, store_temp, outlet_temp, energy_discharged, best_store_air_percentage, best_energy_difference, extra_value = calculate_temp(usable_stored_energy, store_temp, air_temp_in, time_step_hours, plant_output, boost, efficiency, efficacy, thermal_loss, stored_energy_min, core_mass, ave_cp_air_in, boost_energy, initial_store_air_percentage)                
                price_when_discharging[i] = values[1]
                if usable_stored_energy > 0:
                    energy_from_store_per_timestep[i] = energy_discharged
                    test_volumes[i] = usable_stored_energy
                    carbon_displaced_per_timestep[i] = energy_from_store_per_timestep[i] * fuel_co2
                    carbon_savings[i] = carbon_displaced * carbon_tax_prices[i]
                else:
                    energy_from_store_per_timestep[i] = 0
                    carbon_displaced_per_timestep[i] = 0
                    carbon_savings[i] = 0

            variable_discharge = range(0, len(energy_from_store_per_timestep))
            variable_charge = range(0, len(energy_from_store_per_timestep))
            
            combined_variable = list(zip(prices_when_charging, energy_into_store_per_timestep, price_when_discharging, energy_from_store_per_timestep))
            for i in range(1, len(combined_variable)):		
                combined_values = combined_variable[i]

                if combined_values[2] != None:
                    discharging_income[i] =  combined_values[2] * combined_values[3]
                        
                if combined_values[0] != None:
                    charging_cost[i] =  combined_values[0] * combined_values[1]

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
        constant_discharge_value = 0
        energy_from_store_total = 0
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
                
        for d in price_when_discharging:
            if d is not None:
                discharge_value += d 
                
        for d in energy_from_store_per_timestep:
            if d is not None:
                energy_from_store_total += d 
            
        for c in energy_into_store_per_timestep:
            if c is not None:
                energy_into_store_total += c 
                
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
                
        for d in carbon_displaced_per_timestep:
            if d is not None:
                total_carbon_displaced += d
                        
        for d in carbon_savings:
            if d is not None:
                total_carbon_savings += d    
                
        UHTS_profit = (total_store_discharge_income - total_store_charge_cost)
        output_profit = ((UHTS_earned * (stored_energy_max / stored_energy)) - UHTS_spent)
        pure_gas_energy_in = (plant_output / efficiency) * (SPs * (time_step_hours))
        pure_gas_cost = sum(gas_price_full) 
        pure_gas_income = sum(plant_earnings)
        pure_gas_profit = sum(plant_earnings) - pure_gas_cost
        MWh_boost = energy_from_store_total
        boost_percentage = ((energy_from_store_total) / (((plant_output / efficiency)) * (SPs * (time_step_hours)))) * 100
        cycles_2 = MWh_boost / (((plant_output * time_step_hours) / efficiency) * boost)

        #if MWh_sold > best_MWh_sold and pure_gas_cost > uhts_cost:
        if UHTS_profit > best_UHTS_profit:
            best_UHTS_profit = UHTS_profit
            best_UHTS_spent = UHTS_spent
            best_UHTS_earned = UHTS_earned
            best_charge_trigger = charge_trigger
            best_discharge_trigger = discharge_trigger
            best_charge_totals = charge_totals
            best_discharge_totals = discharge_totals
            best_constant_discharge_value = constant_discharge_value
            best_cycles_1 = round(cycles_2)
            best_pure_gas_energy_in = pure_gas_energy_in
            best_pure_gas_cost = pure_gas_cost
            best_pure_gas_income = pure_gas_income
            best_pure_gas_profit = pure_gas_profit
            best_MWh_boost = MWh_boost
            best_boost_percentage = boost_percentage
            best_carbon_displaced = total_carbon_displaced
            best_carbon_savings = total_carbon_savings

###PRINTING TO PYTHON###

float = P_gas
calc_gas_cost = "{:.2f}".format(float)
print("Ave Fuel Price: £", calc_gas_cost)

float = max_charge
calc_gas_and_tax_cost = "{:.2f}".format(float)
print("Fuel Price w/ Tax: £", calc_gas_and_tax_cost)

print("Best Charge Trigger: £", best_charge_trigger)
print("Best Disharge Trigger: £", best_discharge_trigger)

float = best_pure_gas_energy_in
format_mwh_in = "{:.2f}".format(float)
print("Standard MWh Out: ", format_mwh_in)

float = best_pure_gas_cost
format_gas_cost = "{:.2f}".format(float)
print("Pure Gas Cost: £", format_gas_cost)

float = best_pure_gas_income
format_gas_income = "{:.2f}".format(float)
print("Pure Gas Income: £", format_gas_income)

float = best_pure_gas_profit
format_gas_profit = "{:.2f}".format(float)
print("Pure Gas Profit: £", format_gas_profit)

print("Cycles: ", (best_cycles_1))

float = best_MWh_boost
format_uhts_boost = "{:.2f}".format(float)
print("Boosted MWh: ", format_uhts_boost)

float = best_boost_percentage
format_boost_perc = "{:.2f}".format(float)
print("Boosted MWh %: ", format_boost_perc,"%")

float = (best_UHTS_spent)
format_BS_combined1 = "{:.2f}".format(float)
print("UHTS Charge Cost: £", format_BS_combined1)

float = (best_UHTS_spent + best_pure_gas_cost)
format_BS_combined = "{:.2f}".format(float)
print("UHTS/Gas Charge Cost: £", format_BS_combined)

float = (best_UHTS_earned)
format_BE_combined1 = "{:.2f}".format(float)
print("UHTS Discharge Income: £", format_BE_combined1)

float = (best_UHTS_earned + best_pure_gas_income)
format_BE_combined = "{:.2f}".format(float)
print("UHTS/Gas Discharge Income: £", format_BE_combined)

float = best_UHTS_profit
format_BP = "{:.2f}".format(float)
print("Pure UHTS Profit: £", format_BP)

float = (best_UHTS_profit + best_pure_gas_profit)
format_BP_combined = "{:.2f}".format(float)
print("UHTS/Gas Profit: £", format_BP_combined)

float = best_carbon_displaced
format_CD = "{:.2f}".format(float)
print("Carbon Displaced (tn):", format_CD)

float = best_carbon_savings
format_uhts_carbon = "{:.2f}".format(float)
print("UHTS Carbon Tax: £", format_uhts_carbon)

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
    file.write(f"Best Charge Trigger: £ {best_charge_trigger}\n")
    file.write(f"Best Disharge Trigger: £ {best_discharge_trigger}\n")
    file.write(f"Standard MWh Out:  {format_mwh_in}\n")
    file.write(f"Pure Gas Cost: £ {format_gas_cost}\n")
    file.write(f"Pure Gas Income: £ {format_gas_income}\n")
    file.write(f"Pure Gas Profit: £ {format_gas_profit}\n")
    file.write(f"Cycles:  {best_cycles_1}\n")
    file.write(f"Boosted MWh:  {format_uhts_boost}\n")
    file.write(f"Boosted MWh %:  {format_boost_perc}%\n")
    file.write(f"UHTS Charge Cost: £ {format_BS_combined1}\n")
    file.write(f"UHTS/Gas Charge Cost: £ {format_BS_combined}\n")
    file.write(f"UHTS Discharge Income: £ {format_BE_combined1}\n")
    file.write(f"UHTS/Gas Discharge Income: £ {format_BE_combined}\n")
    file.write(f"Pure UHTS Profit: £ {format_BP}\n")
    file.write(f"UHTS/Gas Profit: £ {format_BP_combined}\n")
    file.write(f"Carbon Displaced (tn): {format_CD}\n")
    file.write(f"UHTS Carbon Tax: £ {format_uhts_carbon}\n")
    file.write(f"Time: {format_time}\n")
