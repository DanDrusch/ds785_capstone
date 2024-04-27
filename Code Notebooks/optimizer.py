from pyomo.environ import *
import datetime
import pandas as pd
import os, sys
import copy
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def train_and_predict_for_month(df, method, params, month, target_var='2018 Actual Meteorological Year Generation (kWh)', drop_month=False):
    train = df[df.month != month]
    test = df[df.month == month]

    if drop_month:
        x_train = train.drop([target_var, 'month'], axis=1)
        x_test = test.drop([target_var, 'month'], axis=1)
    else:
        x_train = train.drop(target_var, axis=1)
        x_test = test.drop(target_var, axis=1)

    y_train = train[target_var]

    y_test = test[target_var]

    # Create the model with the passed method
    model = method(**params)
    # Train the model
    model.fit(x_train, y_train)

    # Generate the predictions
    y_pred = model.predict(x_test)

    # Set minimum value
    y_pred[y_pred < y_train.min()] = y_train.min()

    # Create a DF with both predictions and measure data
    month_results = pd.DataFrame({target_var: pd.Series(y_pred, index=y_test.index)})

    return month_results

def lookup_rate_category_for_time(timestamp, rate_schedule):
    """
    Provides the rate category for the given timestamp using the given rate_schedule

    Args: 
        timestamp (datetime.datetime): time used for lookup
        rate_schedule (dict): definition of when different rate categories are used

    Returns:
        str: rate category for the time
    """


    # Get weekend or weekday
    if timestamp.weekday() < 5:
        day = rate_schedule[str(timestamp.month)]['Weekday']
    else:
        day = rate_schedule[str(timestamp.month)]['Weekend']

    return day[timestamp.hour]

def generate_hourly_schedule(start_timestamp, duration, rate_schedule):
    """
    Returns a <duration>-length array of hourly rate categories according to the 
    <rate_schedule≥ starting from <start_timestamp>.

    Args:
        start_timestamp (datetim.datetime): first hour of schedule
        duration (int): number of hours to output
        rate_schedule (dict): definition of when different rate categories are used

    Returns:
        array: the hourly rate categories for the given time period 
    """
    output = []

    for i in range(duration):
        output.append(lookup_rate_category_for_time(start_timestamp + datetime.timedelta(hours=i), rate_schedule))

    return output
    
def basic_heuristic(use_load, generation, rate_plan, system_state, storage_system):
    # If there's excess generation, charge battery until full, then export
    # IF there's excess load, discharge battery until empty, then import
    rate_schedule = rate_plan['rate_schedule']
    rate_prices = rate_plan['rate_prices']

    duration = len(use_load)

    df = use_load.merge(generation, left_index=True, right_index=True)
    df["net_energy"] = df['2018 Actual Meteorological Year Generation (kWh)'] - df['out.electricity.total.energy_consumption']

    # Generate which rates at what time
    rate_list = generate_hourly_schedule(use_load.index[0], duration, rate_schedule)
    rate_groups = pd.Series(rate_list + [key for key, value in system_state.items() if key not in ['soc']]).unique() 

    df['rate_group'] = pd.Series(rate_list, index=df.index)

    # Set up tracking variables
    rate_group_import_export = {}
    for group in rate_groups:
        rate_group_import_export[group] = {
            "to_grid": 0 + system_state.get(group, {}).get("to_grid",0),
            "from_grid": 0 + system_state.get(group, {}).get("from_grid",0)
        }
    soc = system_state['soc']

    max_soc = storage_system['battery_capacity_kwh']
    efficiency = storage_system['efficiency']
    max_discharge = storage_system['max_discharge_power_kw']
    max_charge = storage_system['max_charge_power_kw']

    for idx, row in df.iterrows():
        # Check if we're net producing or net consuming
        net_energy = row['net_energy']
        df.loc[idx, 'starting_soc'] = soc

        if net_energy > 0:
            # Net producing
            # Check if we have room for charging
            if soc >= max_soc:
                # export it all
                export_energy = net_energy
                df.loc[idx, 'export_energy'] = export_energy
                rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
            else:
                # store at least part of it
                export_energy = 0
                if net_energy > max_charge:
                    export_energy = net_energy - max_charge
                    net_energy = max_charge
                
                room_left = max_soc - soc
                if room_left < net_energy:
                    charge_energy = room_left
                    export_energy += (net_energy - room_left)
                else:
                    charge_energy = net_energy # Store it all
                
                soc += charge_energy
                df.loc[idx, 'charge_energy'] = charge_energy
                df.loc[idx, 'export_energy'] = export_energy
                rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
        else:
            # Net consuming
            # Check if we have battery to discharge
            if soc <= 0:
                # import it all
                import_energy =  (-net_energy)
                df.loc[idx, 'import_energy'] = import_energy
                rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
            else:
                # Discharge battery as much as possible
                import_energy = 0
                if -net_energy > max_discharge:
                    import_energy = -net_energy - max_discharge
                    net_energy = net_energy + max_charge

                charge_left = soc * efficiency
                if charge_left < -net_energy:
                    # We need some from the grid
                    discharge_energy = charge_left
                    import_energy = (-net_energy) - charge_left
                else:
                    discharge_energy = (-net_energy)  # power entirely from battery
            
                soc -= discharge_energy / efficiency
                df.loc[idx, 'discharge_energy'] = discharge_energy
                df.loc[idx, 'import_energy'] = import_energy
                rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
        
        df.loc[idx, 'ending_soc'] = soc

    # Go through each group and calculate each of the three categories
    cost = 0
    for group in rate_groups:
        cost += rate_group_import_export[group]['from_grid'] * rate_prices[group]['from_grid']
        if rate_group_import_export[group]['to_grid'] > rate_group_import_export[group]['from_grid']:
            surplus = rate_group_import_export[group]['to_grid'] - rate_group_import_export[group]['from_grid']
            offset = rate_group_import_export[group]['from_grid']
        else:
            surplus = 0
            offset = rate_group_import_export[group]['to_grid']
        cost += offset * rate_prices[group]['to_grid'][0]
        cost += surplus * rate_prices[group]['to_grid'][1]

    return df, cost

def no_generation(use_load, rate_plan):
    rate_schedule = rate_plan['rate_schedule']
    rate_prices = rate_plan['rate_prices']

    duration = len(use_load)

    # Generate which rates at what time
    rate_list = generate_hourly_schedule(use_load.index[0], duration, rate_schedule)
    use_load['import price'] = pd.Series([rate_prices[i]['from_grid'] for i in rate_list], index=use_load.index)

    return (use_load['import price'] * use_load['out.electricity.total.energy_consumption']).sum()

def optimize_period(use_load, generation, rate_plan, system_state, storage_system, force_directionality=False):
    # 
    rate_schedule = rate_plan['rate_schedule']
    rate_prices = rate_plan['rate_prices']
    source_limit = rate_plan['source_limit']

    duration = len(use_load)

    # Generate which rates at what time
    rate_list = generate_hourly_schedule(use_load.index[0], duration, rate_schedule)
    rate_groups = pd.Series(rate_list + [key for key, value in system_state.items() if key not in ['soc']]).unique() 

    # Declaration
    model = ConcreteModel("Day Optimization")

    # Time Period
    model.T = Set(initialize=RangeSet(duration), ordered=True)

    # Fixed Parameters
    model.initial_soc = Param(initialize=system_state["soc"])
    model.efficiency = Param(initialize=storage_system['efficiency'])
    model.battery_max_energy = Param(initialize=storage_system['battery_capacity_kwh'])
    model.max_charging_power = Param(initialize=storage_system['max_charge_power_kw'])
    model.max_discharge_power = Param(initialize=storage_system["max_discharge_power_kw"])

    # Make columns for rate groups
    rate_group_active = pd.DataFrame([[1 if rate_list[time-1] == group else 0 for group in rate_groups] for time in model.T],
            index=model.T,
            columns=rate_groups)

    # Time-based Parameters
    model.use_load = Param(model.T, initialize=dict(enumerate(use_load["out.electricity.total.energy_consumption"],1)))
    model.generation = Param(model.T, initialize=dict(enumerate(generation["2018 Actual Meteorological Year Generation (kWh)"],1)))
    model.rate_group = Param(model.T, initialize=dict(enumerate(rate_list, 1)))

    # Decision Variables
    model.soc = Var(model.T, domain=NonNegativeReals)
    model.battery_energy_charge = Var(model.T, domain=NonNegativeReals)
    model.battery_energy_discharge = Var(model.T, domain=NonNegativeReals)
    model.to_grid = Var(model.T, domain=NonNegativeReals)
    model.from_grid = Var(model.T, domain=NonNegativeReals)
    model.to_grid_bin = Var(model.T, domain=Binary)
    model.from_grid_bin = Var(model.T, domain=Binary)

    model.from_grid_grouped= Var(rate_groups, domain=NonNegativeReals)
    model.to_grid_total_grouped = Var(rate_groups, domain=NonNegativeReals)
    model.to_grid_offset_grouped = Var(rate_groups, domain=NonNegativeReals)
    model.to_grid_surplus_grouped = Var(rate_groups, domain=NonNegativeReals)

    # Objective
    # Fix for time series
    model.total_cost = Objective(expr = sum(rate_prices[group]['from_grid']  * model.from_grid_grouped[group]       + # Cost to buy from grid
                                            rate_prices[group]['to_grid'][0] * model.to_grid_offset_grouped[group]  + # Selling to the grid at offset prices (negative)
                                            rate_prices[group]['to_grid'][1] * model.to_grid_surplus_grouped[group]   # Selling to the grid at surplus prices (negative)
                                            for group in rate_groups
                                            ),
                                sense=minimize)

    # CONSTRAINTS

    # Update requirements
    def battery_update_constraint(model, t):
        if t == model.T.first():
            # This is the first time period, make sure the battery state of charge matches the passed in setting
            return model.soc[t] == model.initial_soc
        else:
            # Take the efficiency hit of the system on discharge
            return model.soc[t] == model.soc[t-1] + model.battery_energy_charge[t - 1] - (model.battery_energy_discharge[t - 1] / model.efficiency)
                                                
    # Battery charge has to be between 0 and battery_max_energy
    def battery_soc_max_constraint(model, t):
        return (model.soc[t] + model.battery_energy_charge[t]) <= model.battery_max_energy
    def battery_soc_min_constraint(model, t):
        return 0 <= (model.soc[t] - model.battery_energy_discharge[t] / model.efficiency)

    # Battery energy flow must be between max charge and discharge rates
    def battery_power_charge_constraint(model, t):
        return model.battery_energy_charge[t] <= model.max_charging_power
    def battery_power_discharge_constraint(model, t):
        return model.battery_energy_discharge[t] <= model.max_discharge_power

    # Flow from grid, to grid, and in/out of battery must equal generation + load
    def energy_conservation_constraint(model, t):
        return model.battery_energy_charge[t] - model.battery_energy_discharge[t] + model.to_grid[t] - model.from_grid[t] == model.generation[t] - model.use_load[t]

    # Only charge from the solar
    def charge_source_constraint(model, t):
        return model.battery_energy_charge[t] <= model.generation[t]
    def charge_source_constraint_2(model, t):
        return model.battery_energy_charge[t] + model.to_grid[t] <= model.generation[t]

    # Only discharge to power the house
    def discharge_destination_constraint(model, t):
        return model.battery_energy_discharge[t] <= model.use_load[t]
    def discharge_destination_constraint_2(model, t):
        return model.battery_energy_discharge[t] + model.from_grid[t] <= model.use_load[t]

    # Only from_grid or to_grid. Not both at the same time
    def single_direction_to(model, t):
        return model.to_grid[t] <= model.battery_energy_discharge[t] + model.generation[t]
    def single_direction_from(model, t):
        return model.from_grid[t] <= model.battery_energy_charge[t] + model.use_load[t]
    def direction_bin1(model, t):
        return model.to_grid_bin[t] + model.from_grid_bin[t] <=1 # only multiplier of one
    def direction_bin3(model, t):
        return model.to_grid[t] <= model.to_grid_bin[t] * 100 # Only value of 1 or 0 for bin works
    def direction_bin4(model, t):
        return model.from_grid[t] <= model.from_grid_bin[t] * 100 # Only value of 1 or 0 for bin works

    model.batt_update_con = Constraint(model.T, rule=battery_update_constraint)
    model.batt_soc_max_con = Constraint(model.T, rule=battery_soc_max_constraint)
    model.batt_soc_min_con = Constraint(model.T, rule=battery_soc_min_constraint)
    model.batt_charge_power_con = Constraint(model.T, rule=battery_power_charge_constraint)
    model.batt_discharge_power_con = Constraint(model.T, rule=battery_power_discharge_constraint)
    model.energy_con = Constraint(model.T, rule=energy_conservation_constraint)

    model.to_grid_con  = Constraint(model.T, rule=single_direction_to)
    model.from_grid_con  = Constraint(model.T, rule=single_direction_from)

    # Enforces binary directionality
    # This adds integer constraints and explodes compute time
    if force_directionality:
        model.dir1_con  = Constraint(model.T, rule=direction_bin1)
        # model.dir2_con  = Constraint(model.T, rule=direction_bin2)
        model.dir3_con = Constraint(model.T, rule=direction_bin3)
        model.dir4_con = Constraint(model.T, rule=direction_bin4)

    # Limits battery use to only from solar/to load (no battery to/from grid)
    if source_limit:
        model.source_con  = Constraint(model.T, rule=charge_source_constraint)
        model.source_con2 = Constraint(model.T, rule=charge_source_constraint_2)
        model.dest_con  = Constraint(model.T, rule=discharge_destination_constraint)
        model.dest_con2 = Constraint(model.T, rule=discharge_destination_constraint_2)

    # Rate Groups Restrictions
    model.rate_group_from_con = Constraint(Any)
    for group in rate_groups:
        # Make sure everything adds up for each rate group
        model.rate_group_from_con[group] = (
            sum([model.from_grid[t] * rate_group_active.loc[t, group] for t in model.T]) + system_state[group]['from_grid'] == model.from_grid_grouped[group]
        )

    model.rate_group_to_total_con = Constraint(Any)
    for group in rate_groups:
        # Make sure everything adds up for each rate group
        model.rate_group_to_total_con[group] = (
            sum([model.to_grid[t] * rate_group_active.loc[t, group] for t in model.T]) + system_state[group]['to_grid'] == model.to_grid_total_grouped[group]
        )

    # Make sure for each group that we balance split the offset and surplus to the grid
    model.rate_group_to_split_con = Constraint(Any)
    for group in rate_groups:
        model.rate_group_to_split_con[group, 'total'] = (
            model.to_grid_total_grouped[group] == model.to_grid_offset_grouped[group] + model.to_grid_surplus_grouped[group]
        )
        model.rate_group_to_split_con[group, 'offset'] = (
            model.to_grid_offset_grouped[group] <= model.from_grid_grouped[group]
        )

    ### SOLUTION ###
    solver = SolverFactory('glpk')
    solver.solve(model)

    # Get last time period index
    max_key = max(model.soc.extract_values().keys())

    # Update the system state
    new_system_state = copy.deepcopy(system_state)
    new_system_state["soc"] = model.soc[max_key].value + model.battery_energy_charge[max_key].value - model.battery_energy_discharge[max_key].value
    for group in rate_groups:
        new_system_state[group] = {
            "from_grid": model.from_grid_grouped[group].value,
            "to_grid": model.from_grid_grouped[group].value
        }

    # Provide recommendation based on first hour's behavior
    # If we charged at all, recommend charge
    # If we discharged at all, recommend discharge
    # If we didn't charge or discharge, recommend charge (arbitrarily)
    recommend_charge = False # Default to discharging
    recommend_export = False # Don't default to selling
    recommend_import = False # Don't default to importing
    if model.battery_energy_charge[1].value > 0:
        recommend_charge = True
        if model.from_grid[1].value > model.use_load[1]:
            recommend_import = model.from_grid[1].value
    elif model.battery_energy_discharge[1].value > 0:
        recommend_charge = False
        if model.to_grid[1].value > 0 and (model.to_grid[1].value - model.from_grid[1].value) > 0:
            recommend_export = model.to_grid[1].value - model.from_grid[1].value

    return recommend_charge, recommend_export, recommend_import, new_system_state, model

def model_summary(model):
    # get all the variables (assuming the fuller model will have constraints, params, etc.)
    model_vars = model.component_map(ctype=Var)

    serieses = []   # collection to hold the converted "serieses"
    for k in model_vars.keys():   # this is a map of {name:pyo.Var}
        v = model_vars[k]

        # make a pd.Series from each    
        s = pd.Series(v.extract_values(), index=v.extract_values().keys())

        s = pd.DataFrame(s)         # force transition from Series -> df
        # print(s)

        # multi-index the columns
        s.columns = [k]
        
        serieses.append(s)

    model_vars = model.component_map(ctype=Param)

    # serieses = []   # collection to hold the converted "serieses"
    for k in model_vars.keys():   # this is a map of {name:pyo.Var}
        v = model_vars[k]

        # make a pd.Series from each    
        s = pd.Series(v.extract_values(), index=v.extract_values().keys())

        s = pd.DataFrame(s)         # force transition from Series -> df
        # print(s)

        # multi-index the columns
        s.columns = [k]
        
        serieses.append(s)

    df = pd.concat(serieses, axis=1)
    return df.drop([None]).dropna(axis=1, how='all')

def predict_load_for_month(use_load, month):
    # Optimal method and hyperparameters from load_forecaster.ipynb
    return train_and_predict_for_month(use_load, SVR, {'C': 1, 'gamma': 0.0001, 'kernel': 'linear'}, month, target_var='out.electricity.total.energy_consumption')

def predict_gen_for_month(generation, month):
    # Optimal method from generation_forecaster.ipynb
    return train_and_predict_for_month(generation, LinearRegression, {"n_jobs":-1}, month, target_var='2018 Actual Meteorological Year Generation (kWh)', drop_month=True)

def predict_load_and_gen(use_load, generation):
    pred_load_file = "../Project_data/Predicted Values/predicted_load.csv"
    pred_gen_file = "../Project_data/Predicted Values/predicted_generation.csv"

    # Use Load
    if os.path.exists(pred_load_file):
        # Use the data if we've already generated it before
        predicted_load = pd.read_csv(pred_load_file, index_col=0, parse_dates=True)
    else:
        load_sections = []
        for month in range(1,13):
            load_sections.append(predict_load_for_month(use_load, month))
        # Combine into single structure
        predicted_load = pd.concat(load_sections)
        # save to files
        predicted_load.to_csv(pred_load_file)

    # Solar Generation
    if os.path.exists(pred_gen_file):
        # Use the data if we've already generated it before
        predicted_gen = pd.read_csv(pred_gen_file, index_col=0, parse_dates=True)
    else:
        gen_sections = []
        for month in range(1,13):
            gen_sections.append(predict_gen_for_month(generation, month))
        # Combine into single structure
        predicted_gen = pd.concat(gen_sections)
        # save to files
        predicted_gen.to_csv(pred_gen_file)

    return predicted_load, predicted_gen

def monthly_predict_and_optimize(use_load, generation, rate_plan, system_state, storage_system, month, predicted_load=None, predicted_generation=None):
    # Train and predict for the month
    if predicted_load is None:
        predicted_load = use_load[use_load['month'] == month].copy()
    if predicted_generation is None:
        predicted_generation = generation[generation['month'] == month].copy()

    rate_schedule = rate_plan['rate_schedule']
    rate_prices = rate_plan['rate_prices']

    use_load = use_load[use_load['month'] == month].copy()
    generation = generation[generation['month'] == month].copy()

    # Actual use/generation
    df = use_load[['out.electricity.total.energy_consumption']].merge(generation[['2018 Actual Meteorological Year Generation (kWh)']], left_index=True, right_index=True)
    df["net_energy"] = df['2018 Actual Meteorological Year Generation (kWh)'] - df['out.electricity.total.energy_consumption']

    # Generate which rates at what time
    rate_list = generate_hourly_schedule(use_load.index[0], len(use_load), rate_schedule)
    rate_groups = pd.Series(rate_list + [key for key, value in system_state.items() if key not in ['soc']]).unique() 

    df['rate_group'] = pd.Series(rate_list, index=df.index)

    # Set up tracking variables
    system_state = copy.deepcopy(system_state)
    rate_group_import_export = {}
    for group in rate_groups:
        rate_group_import_export[group] = {
            "to_grid": 0 + system_state.get(group, {}).get("to_grid",0),
            "from_grid": 0 + system_state.get(group, {}).get("from_grid",0)
        }
    soc = system_state['soc']

    max_soc = storage_system['battery_capacity_kwh']
    efficiency = storage_system['efficiency']
    max_discharge = storage_system['max_discharge_power_kw']
    max_charge = storage_system['max_charge_power_kw']

    # Iterate through each hour and do the following:
    # - Optimize the next 24 hours
    # - Take recommendation of charge/discharge
    # - Update state based on decision
    for idx, row in df.iterrows():
        # print(idx)
        # Optimize for the next 24 hours based on predicted values
        recommend_charge, recommend_export, recommend_import, _, model = optimize_period(predicted_load.loc[idx:idx+pd.Timedelta('23h')], 
                                                                                                    predicted_generation.loc[idx:idx+pd.Timedelta('23h')], 
                                                                                                    rate_plan=rate_plan, 
                                                                                                    system_state=system_state, 
                                                                                                    storage_system=storage_system)

        # Check if we're net producing or net consuming
        net_energy = row['net_energy']
        df.loc[idx, 'starting_soc'] = soc
        df.loc[idx, 'recommend_charge'] = recommend_charge
        df.loc[idx, 'recommend_export'] = recommend_export
        df.loc[idx, 'recommend_import'] = recommend_import

        # recommend_export = False
        # recommend_import = False

        if recommend_export:
            # recommend_export = max_discharge
            if net_energy > 0:
                # If we're net producing, send it all out
                export_energy = net_energy
                df.loc[idx, 'export_energy'] = export_energy
                rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
            else:
                # If we're consuming, send what we can after powering the system
                if soc <= 0:
                    # import it all
                    import_energy = (-net_energy)
                    df.loc[idx, 'import_energy'] = import_energy
                    rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
                elif soc * efficiency < -net_energy:
                    # Don't have quite enough to self supply
                    import_energy = 0
                    charge_left = soc * efficiency
                    # We need some from the grid
                    discharge_energy = charge_left
                    import_energy = (-net_energy) - charge_left
                
                    soc -= discharge_energy / efficiency
                    df.loc[idx, 'discharge_energy'] = discharge_energy
                    df.loc[idx, 'import_energy'] = import_energy
                    rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
                elif (recommend_export + (-net_energy)) <= (soc * efficiency):
                    # Discharge as much as we can!
                    discharge_energy = recommend_export + (-net_energy)
                    if discharge_energy > max_discharge:
                        discharge_energy = max_discharge
                        export_energy = discharge_energy - (-net_energy)
                    else:
                        export_energy = recommend_export
                    df.loc[idx, 'export_energy'] = export_energy
                    rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
                    df.loc[idx, 'discharge_energy'] = discharge_energy
                    soc -= discharge_energy / efficiency
                else:
                    # Discharge whatever is left
                    charge_left = soc * efficiency
                    export_energy = charge_left - (-net_energy)
                    df.loc[idx, 'export_energy'] = export_energy
                    rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
                    df.loc[idx, 'discharge_energy'] = charge_left
                    soc -= charge_left / efficiency
        elif recommend_import:
            # recommend_import = max_charge
            if net_energy < 0:
                # If we're net consuming, import all of it
                import_energy = (-net_energy)
                df.loc[idx, 'import_energy'] = import_energy
                rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
            else:
                # If we're producing, store what we can after powering the system
                if soc >= max_soc:
                    # export it all
                    export_energy = net_energy
                    df.loc[idx, 'export_energy'] = export_energy
                    rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
                elif (max_soc - soc) < net_energy:
                    # Don't have quite enough storage
                    export_energy = 0

                    room_left = max_soc - soc
                    # Send some back to the grid
                    charge_energy = room_left
                    export_energy += (net_energy - room_left)

                    soc += charge_energy
                    df.loc[idx, 'charge_energy'] = charge_energy
                    df.loc[idx, 'export_energy'] = export_energy
                    rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
                elif (max_soc - soc) >= recommend_import and recommend_import >= net_energy:
                    # Charge as much as we can!
                    charge_energy = recommend_import - net_energy
                    df.loc[idx, 'import_energy'] = recommend_import
                    rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + recommend_import
                    df.loc[idx, 'charge_energy'] = charge_energy
                    soc += charge_energy
                else:
                    # Charge whatever is left
                    room_left = max_soc - soc
                    import_energy = room_left - net_energy
                    df.loc[idx, 'import_energy'] = import_energy
                    rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
                    df.loc[idx, 'charge_energy'] = room_left
                    soc += room_left
        elif net_energy > 0:
            # Net producing
            if recommend_charge:
                if soc >= max_soc:
                    # export it all
                    export_energy = net_energy
                    df.loc[idx, 'export_energy'] = export_energy
                    rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
                else:
                    # store at least part of it
                    export_energy = 0
                    if net_energy > max_charge:
                        export_energy = net_energy - max_charge
                        net_energy = max_charge
                    
                    room_left = max_soc - soc
                    if room_left < net_energy:
                        charge_energy = room_left
                        export_energy += (net_energy - room_left)
                    else:
                        charge_energy = net_energy # Store it all
                    
                    soc += charge_energy
                    df.loc[idx, 'charge_energy'] = charge_energy
                    df.loc[idx, 'export_energy'] = export_energy
                    rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
            else:
                # export it all
                export_energy = net_energy
                df.loc[idx, 'export_energy'] = export_energy
                rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy
        else:
            # Net consuming
            if not recommend_charge:
                if soc <= 0:
                    # import it all
                    import_energy =  (-net_energy)
                    df.loc[idx, 'import_energy'] = import_energy
                    rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
                else:
                    # Discharge battery as much as possible
                    import_energy = 0
                    if -net_energy > max_discharge:
                        import_energy = -net_energy - max_discharge
                        net_energy = net_energy + max_charge

                    charge_left = soc * efficiency
                    if charge_left < -net_energy:
                        # We need some from the grid
                        discharge_energy = charge_left
                        import_energy = (-net_energy) - charge_left
                    else:
                        discharge_energy = (-net_energy)  # power entirely from battery
                
                    soc -= discharge_energy / efficiency
                    df.loc[idx, 'discharge_energy'] = discharge_energy
                    df.loc[idx, 'import_energy'] = import_energy
                    rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
            else:
                # import it all
                import_energy =  (-net_energy)
                df.loc[idx, 'import_energy'] = import_energy
                rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
        
        system_state['soc'] = soc
        system_state[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid']
        system_state[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid']
        df.loc[idx, 'ending_soc'] = soc

    # Go through each group and calculate each of the three categories
    cost = 0
    for group in rate_groups:
        cost += rate_group_import_export[group]['from_grid'] * rate_prices[group]['from_grid']
        if rate_group_import_export[group]['to_grid'] > rate_group_import_export[group]['from_grid']:
            surplus = rate_group_import_export[group]['to_grid'] - rate_group_import_export[group]['from_grid']
            offset = rate_group_import_export[group]['from_grid']
        else:
            surplus = 0
            offset = rate_group_import_export[group]['to_grid']
        cost += offset * rate_prices[group]['to_grid'][0]
        cost += surplus * rate_prices[group]['to_grid'][1]

    return df, cost

def monthly_set_and_optimize(use_load, generation, rate_plan, system_state, storage_system, month):
    # Train and predict for the month
    # predicted_load = train_and_predict_for_month(use_load, SVR, {'C': 1, 'gamma': 0.0001, 'kernel': 'linear'}, month, target_var='out.electricity.total.energy_consumption')
    # predicted_generation = train_and_predict_for_month(generation, LinearRegression, {"n_jobs":-1}, month, target_var='2018 Actual Meteorological Year Generation (kWh)', drop_month=True)

    predicted_load = use_load[use_load['month'] == month].copy()
    predicted_generation = generation[generation['month'] == month].copy()

    rate_schedule = rate_plan['rate_schedule']
    rate_prices = rate_plan['rate_prices']

    use_load = use_load[use_load['month'] == month].copy()
    generation = generation[generation['month'] == month].copy()

    # Actual use/generation
    df = use_load[['out.electricity.total.energy_consumption']].merge(generation[['2018 Actual Meteorological Year Generation (kWh)']], left_index=True, right_index=True)
    df["net_energy"] = df['2018 Actual Meteorological Year Generation (kWh)'] - df['out.electricity.total.energy_consumption']

    # Generate which rates at what time
    rate_list = generate_hourly_schedule(use_load.index[0], len(use_load), rate_schedule)
    rate_groups = pd.Series(rate_list + [key for key, value in system_state.items() if key not in ['soc']]).unique() 

    df['rate_group'] = pd.Series(rate_list, index=df.index)

    # Set up tracking variables
    system_state = copy.deepcopy(system_state)
    rate_group_import_export = {}
    for group in rate_groups:
        rate_group_import_export[group] = {
            "to_grid": 0 + system_state.get(group, {}).get("to_grid",0),
            "from_grid": 0 + system_state.get(group, {}).get("from_grid",0)
        }
    soc = system_state['soc']
    efficiency = storage_system['efficiency']

    # Iterate through each hour and do the following:
    # - Optimize the next 24 hours
    # - Take recommendation of charge/discharge
    # - Update state based on decision
    for idx, row in df.iterrows():
        # print(idx)
        # Optimize for the next 24 hours based on predicted values
        recommend_charge, recommend_export, recommend_import, _, model = optimize_period(predicted_load.loc[idx:idx+pd.Timedelta('23h')], 
                                                                                                    predicted_generation.loc[idx:idx+pd.Timedelta('23h')], 
                                                                                                    rate_plan=rate_plan, 
                                                                                                    system_state=system_state, 
                                                                                                    storage_system=storage_system)

        # Check if we're net producing or net consuming
        df.loc[idx, 'starting_soc'] = soc

        charge_energy = model.battery_energy_charge[1].value
        discharge_energy = model.battery_energy_discharge[1].value
        import_energy = model.from_grid[1].value
        export_energy = model.to_grid[1].value

        soc = soc + charge_energy - discharge_energy / efficiency

        df.loc[idx, 'discharge_energy'] = discharge_energy
        df.loc[idx, 'charge_energy'] = charge_energy
        if import_energy > 0 and export_energy > 0:
            if import_energy > export_energy:
                import_energy = import_energy - export_energy
                export_energy = 0
            else:
                export_energy = export_energy - import_energy
                import_energy = 0
        df.loc[idx, 'import_energy'] = import_energy
        df.loc[idx, 'export_energy'] = export_energy
        rate_group_import_export[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid'] + import_energy
        rate_group_import_export[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid'] + export_energy

        system_state['soc'] = soc
        system_state[row['rate_group']]['from_grid'] = rate_group_import_export[row['rate_group']]['from_grid']
        system_state[row['rate_group']]['to_grid'] = rate_group_import_export[row['rate_group']]['to_grid']

        df.loc[idx, 'ending_soc'] = soc

    # Go through each group and calculate each of the three categories
    cost = 0
    for group in rate_groups:
        cost += rate_group_import_export[group]['from_grid'] * rate_prices[group]['from_grid']
        if rate_group_import_export[group]['to_grid'] > rate_group_import_export[group]['from_grid']:
            surplus = rate_group_import_export[group]['to_grid'] - rate_group_import_export[group]['from_grid']
            offset = rate_group_import_export[group]['from_grid']
        else:
            surplus = 0
            offset = rate_group_import_export[group]['to_grid']
        cost += offset * rate_prices[group]['to_grid'][0]
        cost += surplus * rate_prices[group]['to_grid'][1]

    return df, cost
