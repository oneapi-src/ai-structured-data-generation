# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions for generating synthetic data specific to an industry or custom config
'''
def generate_UID(num_rows, id_length):
    symbols = {}
    letters = string.ascii_uppercase + string.digits
    for j in range(num_rows):  # pylint: disable=W0612
        sym = ''.join(secrets.choice(letters) for n in range(id_length))
        while sym in symbols:
            sym = ''.join(secrets.choice(letters) for n in range(id_length))
        symbols[sym] = 1
    return symbols

def custom_dataset(conf, dataset_length):  # pylint: disable=W0102
    np_time = 0
    sp_time = 0
    custom_df = pd.DataFrame()
    num_rows = dataset_length  # aka dataset length
    ids = generate_UID(num_rows, 6)
    add_categorical_column(custom_df, num_rows, "UID", ids.keys())

    numeric_cols = conf["Numeric_columns"]
    numeric_dists = conf["Numeric_distributions"]
    numeric_vals = conf["Numeric_init_vals"]
    for i in range(1, numeric_cols + 1):
        sp_time += add_numeric_column(custom_df, num_rows, "Num_col_" + str(i), numeric_dists[str(i)].lower(), numeric_vals[str(i)])

    cat_cols = conf["Categorical_columns"]
    cat_types = conf["Cat_col_types"]
    cat_vals = conf["Cat_col_vals"]
    for i in range(1, cat_cols + 1):
        if cat_types[str(i)] == "list":
            if len(cat_vals[str(i)]["probabilities"]) == 0:
                add_categorical_column(custom_df, num_rows, "Cat_col_" + str(i), [], cat_vals[str(i)]["values"], None)                
            elif sum(cat_vals[str(i)]["probabilities"]) != 1:
                logger.info("Probabilities of categorical column must add up to 1 for categorical column #%d", i)
            else:
                add_categorical_column(custom_df, num_rows, "Cat_col_" + str(i), [], cat_vals[str(i)]["values"], cat_vals[str(i)]["probabilities"])
        elif cat_types[str(i)] == "UID":
            symbols = generate_UID(num_rows, cat_vals[str(i)]["length"])
            add_categorical_column(custom_df, num_rows, "Cat_col_" + str(i), symbols.keys())

    time_cols = conf["Timeseries_columns"]
    time_dur = conf["Time_duration"]
    time_types = conf["Time_col_types"]
    time_params = conf["Time_col_params"]
    for i in range(1, time_cols + 1):
        np_time += add_timeseries_column(custom_df, num_rows, time_dur, time_types[str(i)].upper(), time_params[str(i)], "Timeseries_col_" + str(i))
    
    target_cols = conf["Target_columns"]
    target_types = conf["Target_col_types"]
    target_class = conf["Target_col_classes"]
    for i in range(1, target_cols + 1):
        if target_types[str(i)] == "binary":
            vals = target_class[str(i)]["values"] + target_class[str(i)]["weights"]
            sp_time += add_numeric_column(custom_df, num_rows, "Target_" + str(i), "binomial", vals)
        elif target_types[str(i)] == "multi-class":
            add_categorical_column(custom_df, num_rows, "Target_" + str(i), [], target_class[str(i)]["values"], target_class[str(i)]["weights"])
        else:
            sp_time += add_numeric_column(custom_df, num_rows, "Target_" + str(i), target_class[str(i)]["dist"], target_class[str(i)]["values"])

    for col in custom_df.columns:
        logger.info("Created column %s of size %d", col, len(custom_df[col]))
    filename = ".//data//custom_data_" + str(num_rows) + ".csv"
    custom_df.to_csv(filename, index=False)
    return sp_time, np_time

def fin_CAR_dataset(dataset_length, time_length, conf):
    sp_time = 0
    np_time = 0
    pd_time = 0
    fin_df = pd.DataFrame()
    add_numeric_column(fin_df, dataset_length)

    economy = conf["Economy_health"]
    start_range = conf["Init_values"]
    sectors = conf["Sectors"]
    data_context = conf["Sub-context"]
    dist = conf["Distribution"]

    params = {}
    if economy == "normal":
        growths = [-2, -1.5, -1, 1, 1.5, 2, 2.5, 3]
        params["sigma"] = 15
    elif economy == "recession":
        growths = [-2, -1.5, -1]
        params["sigma"] = 25
    elif economy == "boom":
        growths = [1.5, 2, 2.5, 3]
        params["sigma"] = 7
    params["sector_options"] = {}
    for sec in sectors:
        params["sector_options"][sec] = secrets.choice(growths)
    params["sectors"] = [secrets.choice(list(params["sector_options"].keys())) for i in range(dataset_length)]
    add_categorical_column(fin_df, dataset_length, "Sectors", params["sectors"])

    symbols = generate_UID(dataset_length, 4)
    add_categorical_column(fin_df, dataset_length, "Symbols", symbols.keys())

    start, end = start_range
    if dist == "normal":
        start = time.time()
        params["starts"] = stats.norm.rvs((start+end)//2, abs(end-start)//6, dataset_length)
        sp_time += time.time() - start
    else:
        params["starts"] = []
        for i in range(dataset_length):
            params["starts"].append(secrets.choice(range(start, end+1)))
    add_categorical_column(fin_df, dataset_length, "Start_prices", params["starts"])
    np_time += add_timeseries_column(fin_df, dataset_length, time_length, "CAR", params)

    filename = ".//data//financial_data_" + data_context + '_' + str(dataset_length) + ".csv"
    fin_df.to_csv(filename, index=False)
    return sp_time, np_time, pd_time

def health_MG_dataset(dataset_length, time_length, conf):
    sp_time = 0
    np_time = 0
    pd_time = 0
    health_df = pd.DataFrame()
    add_numeric_column(health_df, dataset_length)

    equil = conf["Equilibrium_value"]
    data_context = conf["Sub-context"]
    dist = conf["Distribution"]
    params = {}
    params["equil"] = equil
    params["init_exp"] = True
    if dist == "normal":
        start = time.time()
        params["tau"] = stats.norm.rvs(16, 2, dataset_length)
        sp_time += time.time() - start
        params["tau"] = params["tau"].astype(int)
        start = time.time()
        params["tau"] = np.where(params["tau"] > 10, params["tau"], 10)
        params["tau"] = np.where(params["tau"] < 22, params["tau"], 22)
        np_time += time.time() - start
    else:
        params["tau"] = []
        for i in range(dataset_length):
            params["tau"].append(secrets.choice(range(10, 23)))
    add_categorical_column(health_df, dataset_length, "Tau_values", params["tau"])

    target = []

    for i in range(dataset_length):
        if params["tau"][i] > 17:
            target.append("unstable")
        else:
            target.append("stable")
    add_categorical_column(health_df, dataset_length, "Target", target)
    np_time += add_timeseries_column(health_df, dataset_length, time_length, "MG", params)

    def my_noise(x):
        return add_noise(time_length, x, white=True, wparams={"std": 0.1})
    start = time.time()
    health_df["Timeseries"] = health_df["Timeseries"].apply(func=my_noise)
    pd_time += time.time() - start

    filename = ".//data//healthcare_data_" + data_context + '_' + str(dataset_length) + ".csv"
    health_df.to_csv(filename, index=False)

    return sp_time, np_time, pd_time

def electrical_SIN_dataset(dataset_length, time_length, conf):
    global sp_time  # pylint: disable=W0603
    sp_time = 0
    np_time = 0
    pd_time = 0
    electric_df = pd.DataFrame()
    add_numeric_column(electric_df, dataset_length)

    data_context = conf["Sub-context"]
    dist = conf["Distribution"]
    params = {}
    target = []
    params["amplitude"] = []
    params["frequency"] = []
    params["ftype"] = []

    def neg_sin(x):
        return np.sin(x) + (np.sin(4*x))/5 + (np.sin(2*x))/5 + (np.sin(5*x))/4

    def pos_sin(x):
        return np.sin(x) + np.sin(2*np.pi*x)/8

    start = time.time()
    n_func = np.frompyfunc(neg_sin, 1, 1)
    p_func = np.frompyfunc(pos_sin, 1, 1)
    np_time += time.time() - start
    if dist == "skewed":
        negative_count = (dataset_length//16)*15
        positive_count = dataset_length - negative_count
    else:
        negative_count = dataset_length//2
        positive_count = dataset_length - negative_count

    n_amp = 20
    p_amp = 15

    for i in range(negative_count):  # pylint: disable=W0612
        params["amplitude"].append(n_amp)
        params["frequency"].append(50)
        params["ftype"].append(n_func)
        target.append(0)
    for i in range(positive_count):  # pylint: disable=W0612
        params["amplitude"].append(p_amp)
        params["frequency"].append(50)
        params["ftype"].append(p_func)
        target.append(1)

    add_categorical_column(electric_df, dataset_length, "Target", target)
    np_time += add_timeseries_column(electric_df, dataset_length, time_length, "SIN", params)

    def my_noise(x):
        return add_noise(time_length, x, white=True, wparams={"std": 2})
    start = time.time()
    electric_df["Timeseries"] = electric_df["Timeseries"].apply(func=my_noise)
    pd_time += time.time() - start

    def my_anomalies(x):
        global sp_time  # pylint: disable=W0603
        if x["Target"] == 0:
            p_small, n_small, spt = add_anomalies(time_length, time_length//200, n_amp)
            sp_time += spt
        else:
            p_small, n_small, spt = add_anomalies(time_length, time_length//120, p_amp*3)
            sp_time += spt
        return x["Timeseries"] + p_small + n_small
    electric_df["Timeseries"] = electric_df.apply(my_anomalies, axis=1)
    
    filename = ".//data//utilities_data_" + data_context + '_' + str(dataset_length) + ".csv"
    electric_df.to_csv(filename, index=False)
    
    return sp_time, np_time, pd_time

def ecommerce_dataset(dataset_length, time_length, conf):
    sp_time = 0
    np_time = 0
    pd_time = 0
    ecom_df = pd.DataFrame()
    add_numeric_column(ecom_df, dataset_length)

    cond = conf["Financial_condition"]
    start_range = conf["Init_values"]
    sector = conf["Sector"]
    categories = conf["Categories"]
    dist = conf["Distribution"]

    if cond == "stable":
        growth_options = [-2, -1.5, -1, 1, 1.25, 1.5, 1.75, 2]
        sigma = 15
    elif cond == "crashing":
        growth_options = [-2, -1.5, -1]
        sigma = 25
    elif cond == "growing":
        growth_options = [1.5, 2, 2.5, 3]
        sigma = 7
    params = {}
    params["growths"] = [secrets.choice(growth_options) for i in range(dataset_length)]
    params["sigma"] = sigma
    start, end = start_range
    if dist == "normal":
        start = time.time()
        params["starts"] = stats.norm.rvs((start+end)//2, abs(end-start)//6, dataset_length)
        sp_time += time.time() - start
    else:
        params["starts"] = [secrets.choice(range(start, end+1)) for i in range(dataset_length)]

    stores = {}
    letters = string.ascii_uppercase + string.digits
    for i in range(dataset_length):
        s_id = ''.join(secrets.choice(letters) for i in range(5))
        while s_id in stores:
            s_id = ''.join(secrets.choice(letters) for i in range(5))
        stores[s_id] = 1
    add_categorical_column(ecom_df, dataset_length, "Store_id", stores.keys())

    if "Sales" in categories:
        np_time += add_timeseries_column(ecom_df, dataset_length, time_length, "CAR", params, "Sales_Timeseries")

    if "Revenue" in categories:
        for i in range(dataset_length):
            if params["growths"][i] > 0:
                params["starts"][i] *= params["growths"][i]

        np_time += add_timeseries_column(ecom_df, dataset_length, time_length, "CAR", params, "Revenue_Timeseries")

    if "Subscribers" in categories:
        params["starts"] = [secrets.choice(range(10, 500)) for i in range(dataset_length)]
        np_time += add_timeseries_column(ecom_df, dataset_length, time_length, "CAR", params, "Subscribers_Timeseries")

        def my_season(x):
            return add_seasonality(x, time_length, 1/365, 10).astype(int)
        start = time.time()
        ecom_df["Subscribers_Timeseries"] = ecom_df["Subscribers_Timeseries"].apply(func=my_season)
        pd_time += time.time() - start

    filename = ".//data//ecomm_data_" + sector + '_' + str(dataset_length) + ".csv"
    ecom_df.to_csv(filename, index=False)

    return sp_time, np_time, pd_time

def env_PSUEDO_dataset(dataset_length, time_length, conf):
    sp_time = 0
    np_time = 0
    pd_time = 0
    env_df = pd.DataFrame()
    add_numeric_column(env_df, dataset_length)

    start, end = conf["Init_values"]
    data_context = conf["Sub-context"]
    dist = conf["Distribution"]
    trend = conf["Trend"]
    region = conf["Region"]
    params = {}
    if dist == "normal":
        start = time.time()
        params["starts"] = stats.norm.rvs((start+end)//2, abs(end-start)//6, dataset_length)
        sp_time += time.time() - start
    else:
        params["starts"] = [secrets.choice(range(start, end+1)) for i in range(dataset_length)]

    if trend == "negative":
        if dataset_length <= 100:
            growth_options = [0.9, 0.95, 0.97, 1]
        elif dataset_length <= 500:
            growth_options = [0.85, 0.87, 0.9, 1]
        else:
            growth_options = [0.8, 0.83, 0.85, 1]
    else:
        if dataset_length <= 100:
            growth_options = [1.03, 1.05, 1.07, 1]
        elif dataset_length <= 500:
            growth_options = [1.07, 1.1, 1.13, 1]
        else:
            growth_options = [1.13, 1.15, 1.17, 1]
    params["growths"] = [secrets.choice(growth_options) for i in range(dataset_length)]

    def env_pp(x):
        return np.sin(2*np.pi*x/12 + np.pi/2)

    start = time.time()
    func = np.frompyfunc(env_pp, 1, 1)
    np_time += time.time() - start
    params["ftype"] = func
    np_time += add_timeseries_column(env_df, dataset_length, time_length, "PSEUDOPERIODIC", params)

    def my_noise(x):
        return add_noise(time_length, x, red=True, wparams={"std": 0.5, "tau": 0.8})
    start = time.time()
    env_df["Timeseries"] = env_df["Timeseries"].apply(func=my_noise)
    pd_time += time.time() - start

    filename = ".//data//env_data_" + region + '_' + data_context + '_' + str(dataset_length) + ".csv"
    env_df.to_csv(filename, index=False)

    return sp_time, np_time, pd_time

if __name__ == "__main__":
    import argparse
    import logging
    import pathlib
    import warnings
    import json
    import secrets
    import time
    import string
    import numpy as np
    from scipy import stats  # pylint: disable=E0401
    from generator import add_numeric_column, add_categorical_column, add_timeseries_column
    from generator import add_seasonality, add_noise, add_anomalies

    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore")

    parser.add_argument('--industry',
                        type=str,
                        default="",
                        help="pick a preset indsutry for which to generate synthetic data: \
                        finance, healthcare, utilities, e-commerce, environmental, custom")
    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    parser.add_argument('-n',
                        '--dataset_len',
                        default=100,
                        help="number of points in dataset")
    parser.add_argument('-t',
                        '--time_duration',
                        default=1000,
                        help="integer value for length of unit-less time")
    parser.add_argument('--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies")
    
    FLAGS = parser.parse_args()
    INTEL_FLAG = FLAGS.intel
    data_len = int(FLAGS.dataset_len)
    time_len = int(FLAGS.time_duration)
    industry = FLAGS.industry.lower()
    sp_time = 0
    np_time = 0
    pd_time = 0

    if INTEL_FLAG:
        import modin.config as cfg  # pylint: disable=E0401
        cfg.Engine.put('ray')
        import modin.pandas as pd  # pylint: disable=E0401
        import ray  # pylint: disable=E0401
        ray.init()
        RAY_DISABLE_MEMORY_MONITOR = 1
    else:
        import pandas as pd

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logging.getLogger('modin').setLevel(logging.WARNING)

    with open("src/myconfig.json", "r") as jsonfile:  # pylint: disable=W1514
        data = json.load(jsonfile)

    print("Data generation can take a while depending on the size expected and level of customization. Please wait until the data generation is complete...")

    if industry == "":
        logger.info("no specifications provided, please run script with a configuration or preset industry")
    
    elif industry == "custom":
        sp_time, np_time = custom_dataset(data["Custom"], data_len)

    elif industry == "finance":
        sp_time, np_time, pd_time = fin_CAR_dataset(data_len, time_len, data["Finance"])

    elif industry == "healthcare":
        sp_time, np_time, pd_time = health_MG_dataset(data_len, time_len, data["Healthcare"])

    elif industry == "utilities":
        sp_time, np_time, pd_time = electrical_SIN_dataset(data_len, time_len, data["Utilities"])

    elif industry == "ecommerce" or industry == "e-commerce":  # pylint: disable=R1714
        sp_time, np_time, pd_time = ecommerce_dataset(data_len, time_len, data["E-commerce"])

    elif industry == "environmental":
        sp_time, np_time, pd_time = env_PSUEDO_dataset(data_len, time_len, data["Environmental"])
    else:
        logger.info("industry not found in implementation, ")
    if sp_time > 0:
        logger.info("Scipy time to generate data for %s industry: %f secs", industry, sp_time)
    if np_time > 0:
        logger.info("Numpy time to generate data for %s industry: %f secs", industry, np_time)
    if pd_time > 0 and INTEL_FLAG:
        logger.info("Modin time to generate data for %s industry: %f secs", industry, pd_time)
    elif pd_time > 0:
        logger.info("Pandas time to generate data for %s industry: %f secs", industry, pd_time)
