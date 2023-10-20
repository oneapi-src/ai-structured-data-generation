# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions for generating synthetic data including timeseries, categorical, and numeric data.
'''
import random
import time
import secrets
import numpy as np
from scipy import signal  # pylint: disable=E0401
from scipy import stats  # pylint: disable=E0401
import timesynth as ts  # pylint: disable=E0401
from tqdm import tqdm

def add_numeric_column(df, length, label="Sig_id", dist=None, vals=None):
    print("adding numeric column")
    spt = 0
    if dist is None or dist == "range":
        df[label] = [num for num in tqdm(range(length))]  # pylint: disable=R1721

    elif dist == "normal":
        start = time.time()
        df[label] = stats.norm.rvs((vals[0] + vals[1])//2, abs(vals[0] - vals[1])//6, length)
        spt += time.time() - start

    elif dist == "random":
        arr = []
        for i in tqdm(range(length)):
            arr.append(secrets.choice(range(min(vals), max(vals))))
        df[label] = arr

    elif dist == "binomial":
        arr = []
        arr += [vals[0]] * int(vals[2]*length)
        arr += [vals[1]] * int(vals[3]*length)
        if len(arr) < length:
            arr += [vals[1]] * int(length - len(arr))
        elif len(arr) > length:
            arr = arr[:length]
        random.shuffle(arr)
        df[label] = arr

    elif dist == "uniform":
        diff = abs(vals[0]-vals[1]) + 1
        arr = []
        for i in range(min(vals), max(vals) + 1):
            arr += [i] * (length//diff)
        if len(arr) < length:
            arr += [vals[1]] * (length - len(arr))
        elif len(arr) > length:
            arr = arr[:length]
        random.shuffle(arr)
        df[label] = arr
    print("adding numeric column - Done\n")
    return spt


def add_categorical_column(df, length, label, values, options=None, weights=None):
    print("adding categorical column")
    if options is None:
        df[label] = values
    elif weights is None or len(weights) == 0:
        df[label] = [secrets.choice(options) for i in tqdm(range(length))]
    else:
        arr = []
        for i in range(len(options)):
            arr += [options[i]] * int(length * weights[i])
        if len(arr) < length:
            arr += [options[-1]] * int(length - len(arr))
        elif len(arr) > length:
            arr = arr[:length]
        random.shuffle(arr)
        df[label] = arr
    print("adding categorical column - Done\n")


def add_timeseries_column(df, dataset_length, time_length, sig_type, params, label="Timeseries"):
    print("adding time-series column")
    np_time = 0
    if sig_type == "CAR":
        sig_dict = {}
        mparams = {}
        for i in tqdm(range(dataset_length)):
            mparams["sigma"] = params["sigma"] if "sigma" in params else 0.5
            mparams["ar_param"] = params["ar_param"] if "ar_param" in params else 0.9
            if "starts" in params:
                mparams["start"] = params["starts"][i] if len(params["starts"]) == dataset_length else secrets.choice(params["starts"])
            else:
                mparams["start"] = 0.01
            if "sectors" in params:
                sector = params["sectors"][i]
                mparams["growth"] = params["sector_options"][sector]
            elif "growths" in params:
                mparams["growth"] = params["growths"][i] if len(params["growths"]) == dataset_length else secrets.choice(params["growths"])
            sig_dict[str(i)], npt = gen_car_signal(time_length, mparams)
            np_time += npt
        df[label] = list(sig_dict.values())

    elif sig_type == "AR":
        sig_dict = {}
        mparams = {}
        for i in tqdm(range(dataset_length)):
            mparams["sigma"] = params["sigma"] if "sigma" in params else 0.5
            mparams["ar_param"] = params["ar_param"] if "ar_param" in params else [1.5, -0.75]  # arbitrary numbers for AR(2) model
            mparams["start_value"] = params["start_value"] if "start_value" in params else [None]
            if "growths" in params:
                if len(params["growths"]) == dataset_length:
                    mparams["growth"] = params["growths"][i]
                    mparams["start"] = params["starts"][i]
                else:
                    mparams["growth"] = secrets.choice(params["growths"])
                    mparams["start"] = secrets.choice(params["starts"])
            sig_dict[str(i)], npt = gen_ar_signal(time_length, mparams)
            np_time += npt
        df[label] = list(sig_dict.values())

    elif sig_type == "NARMA":
        sig_dict = {}
        mparams = {}
        for i in tqdm(range(dataset_length)):
            mparams["order"] = params["order"] if "order" in params else 10
            mparams["coefficients"] = params["coefficients"] if "coefficients" in params else [0.3, 0.05, 1.5, 0.1]
            mparams["initial_condition"] = params["initial_condition"] if "initial_condition" in params else None
            mparams["error_initial_condition"] = params["error_initial_condition"] if "error_initial_condition" in params else None
            mparams["seed"] = params["seed"] if "seed" in params else 42
            if "growths" in params:
                if len(params["growths"]) == dataset_length:
                    mparams["growth"] = params["growths"][i]
                    mparams["start"] = params["starts"][i]
                else:
                    mparams["growth"] = secrets.choice(params["growths"])
                    mparams["start"] = secrets.choice(params["starts"])
            sig_dict[str(i)], npt = gen_narma_signal(time_length, mparams)
            np_time += npt
        df[label] = list(sig_dict.values())

    elif sig_type == "MG":
        sig_dict = {}
        mparams = {}
        for i in tqdm(range(dataset_length)) :
            if "tau" in params:
                mparams["tau"] = params["tau"][i] if len(params["tau"]) == dataset_length else secrets.choice(params["tau"])
            else:
                mparams["tau"] = 16
            mparams["equil"] = params["equil"] if "equil" in params else 1
            mparams["n"] = params["n"] if "n" in params else 8
            mparams["beta"] = params["beta"] if "beta" in params else 0.2
            mparams["gamma"] = params["gamma"] if "gamma" in params else 0.1
            mparams["initial_condition"] = params["initial_condition"] if "initial_condition" in params else None
            mparams["burn_in"] = params["burn_in"] if "burn_in" in params else 500
            if "init_exp" in params:
                mparams["init_exp"] = True
            sig_dict[str(i)], npt = gen_mg_signal(time_length, mparams)
            np_time += npt
        df[label] = list(sig_dict.values())

    elif sig_type == "SIN":
        sig_dict = {}
        mparams = {}
        for i in tqdm(range(dataset_length)):
            if "frequency" in params:
                mparams["frequency"] = params["frequency"][i] if len(params["frequency"]) == dataset_length else secrets.choice(params["frequency"])
            else:
                mparams["frequency"] = 1.0
            if "amplitude" in params:
                mparams["amplitude"] = params["amplitude"][i] if len(params["amplitude"]) == dataset_length else secrets.choice(params["amplitude"])
            else:
                mparams["amplitude"] = 1.0
            if "ftype" in params:
                mparams["ftype"] = params["ftype"][i] if len(params["ftype"]) == dataset_length else secrets.choice(params["ftype"])
            else:
                mparams["ftype"] = np.sin
            sig_dict[str(i)] = gen_sin_signal(time_length, mparams)
        df[label] = list(sig_dict.values())
        npt = 0

    elif sig_type == "PSEUDOPERIODIC":
        sig_dict = {}
        mparams = {}
        for i in tqdm(range(dataset_length)):
            mparams["frequency"] = params["frequency"] if "frequency" in params else 1.0
            mparams["amplitude"] = params["amplitude"] if "amplitude" in params else 1.0
            mparams["ampSD"] = params["ampSD"] if "ampSD" in params else 0.1
            mparams["freqSD"] = params["freqSD"] if "freqSD" in params else 0.1
            mparams["ftype"] = params["ftype"] if "ftype" in params else np.sin
            if "growths" in params:
                if len(params["growths"]) == dataset_length:
                    mparams["growth"] = params["growths"][i]
                    mparams["start"] = params["starts"][i]
                else:
                    mparams["growth"] = secrets.choice(params["growths"])
                    mparams["start"] = secrets.choice(params["starts"])
            sig_dict[str(i)], npt = gen_pseudoperiodic_signal(time_length, mparams)
            np_time += npt
        df[label] = list(sig_dict.values())
    print("adding time-series column - Done\n")
    return np_time

def add_seasonality(og_sig, length, freq, mean):
    time_sampler = ts.TimeSampler(stop_time=length)
    regular_time_samples = time_sampler.sample_regular_time(num_points=length)

    period = ts.signals.Sinusoidal(amplitude=mean, frequency=freq)
    white_noise = ts.noise.GaussianNoise(std=2)
    series = ts.TimeSeries(period, noise_generator=white_noise)
    sig = series.sample(regular_time_samples)[0]

    return og_sig + sig

def add_noise(length, og_sig, red=False, white=False, rparams={}, wparams={}):  # pylint: disable=W0102
    white_noise = None
    red_noise = None
    errors = np.zeros(length)
    samples = og_sig.copy()
    time_sampler = ts.TimeSampler(stop_time=length)
    time_vector = time_sampler.sample_regular_time(num_points=length)
    if white:
        wstd = wparams["std"] if "std" in wparams else 1
        wmean = wparams["mean"] if "mean" in wparams else 0
        white_noise = ts.noise.GaussianNoise(std=wstd, mean=wmean)
        
        errors = white_noise.sample_vectorized(time_vector)
        samples += errors

    elif red:
        rstd = rparams["std"] if "std" in rparams else 1
        rmean = rparams["mean"] if "mean" in rparams else 0
        rtau = rparams["mean"] if "tau" in rparams else 0.2
        rstart_value = rparams["start_value"] if "start_value" in rparams else 0
        red_noise = ts.noise.RedNoise(std=rstd, mean=rmean, tau=rtau, start_value=rstart_value)
        for i in tqdm(range(length)):
            t = time_vector[i]
            errors[i] = red_noise.sample_next(t, samples[:i - 1], errors[:i - 1])
            samples[i] += errors[i]

    return samples

def add_anomalies(length, num, amp):
    start = time.time()
    pos_impulses = signal.unit_impulse(length, random.sample(range(length), num))*random.sample(range(0, amp), 1)
    neg_impulses = signal.unit_impulse(length, random.sample(range(length), num))*random.sample(range(-1*amp, 0), 1)
    return pos_impulses, neg_impulses, time.time()-start

def add_linear_trend(initial, growth_factor, length):
    return np.linspace(initial, initial*growth_factor, length)

def gen_ar_signal(length, params):
    npt = 0
    time_sampler = ts.TimeSampler(stop_time=length)
    regular_time_samples = time_sampler.sample_regular_time(num_points=length)
    ar_p = ts.signals.AutoRegressive(ar_param=params["ar_param"], sigma=params["sigma"], start_value=params["start_value"])
    ar_p_series = ts.TimeSeries(signal_generator=ar_p)
    ar_samples = ar_p_series.sample(regular_time_samples)

    if "growth" in params:
        start = time.time()
        trend = add_linear_trend(params["start"], params["growth"], length)
        data = np.add(ar_samples, trend)
        data = np.where(data > 0, data, data * -1)
        npt += time.time() - start
        return data, npt

    return ar_samples, npt

def gen_car_signal(length, params):
    npt = 0
    time_sampler = ts.TimeSampler(stop_time=length)
    regular_time_samples = time_sampler.sample_regular_time(num_points=length)
    car = ts.signals.CAR(ar_param=params["ar_param"], sigma=params["sigma"])
    car_series = ts.TimeSeries(signal_generator=car)
    car_samples = car_series.sample(regular_time_samples)[0]

    if "growth" in params:
        start = time.time()
        trend = add_linear_trend(params["start"], params["growth"], length)
        data = np.add(car_samples, trend)
        data = np.where(data > 0, data, data * -1)
        npt += time.time() - start
        return data, npt

    return car_samples, npt

def gen_narma_signal(length, params):
    npt = 0
    time_sampler = ts.TimeSampler(stop_time=length)
    regular_time_samples = time_sampler.sample_regular_time(num_points=length)

    narma = ts.signals.NARMA(
        order=params["order"], coefficients=params["coefficients"], initial_condition=params["initial_condition"],
        error_initial_condition=params["error_initial_condition"], seed=params["seed"])
    narma_series = ts.TimeSeries(narma)
    narma_samples = narma_series.sample(regular_time_samples)[0]

    if "growth" in params:
        start = time.time()
        trend = add_linear_trend(params["start"][0], params["growth"], length)
        data = np.add(narma_samples, trend)
        data = np.where(data > 0, data, data * -1)
        npt += time.time() - start
        return data, npt

    return narma_samples, npt

def gen_mg_signal(length, params):
    npt = 0
    time_sampler = ts.TimeSampler(stop_time=length)
    irregular_time_samples_mg = time_sampler.sample_irregular_time(num_points=length*2, keep_percentage=50)

    mg = ts.signals.MackeyGlass(
        tau=params["tau"], n=params["n"], beta=params["beta"], gamma=params["gamma"],
        initial_condition=params["initial_condition"], burn_in=params["burn_in"])
    mg_series = ts.TimeSeries(signal_generator=mg)
    mg_samples = mg_series.sample(irregular_time_samples_mg)[0]

    if params["equil"] != 1:
        start = time.time()
        mg_samples = mg_samples + np.linspace(params["equil"] - 1, params["equil"] - 1, length)
        npt += time.time() - start

    if "init_exp" in params:
        ylim = params["equil"] - 1
        start = time.time()
        xlim = (np.where(mg_samples[50:] < ylim + 0.01) and np.where(mg_samples[50:] > ylim - 0.01))[0][0] + 50
        root = np.sqrt(ylim)
        x = np.linspace(0, xlim, xlim)
        y = np.exp(((root*x)/(xlim*1.2))**2) - 1
        npt += time.time() - start
        mg_samples[:xlim] = y

    return mg_samples, npt

def gen_sin_signal(length, params):
    time_sampler = ts.TimeSampler(stop_time=1/params["frequency"])
    regular_time_samples = time_sampler.sample_regular_time(num_points=length)

    sinus = ts.signals.Sinusoidal(amplitude=params["amplitude"], frequency=params["frequency"], ftype=params["ftype"])
    sin_series = ts.TimeSeries(sinus)
    sin_samples = sin_series.sample(regular_time_samples)[0]

    return sin_samples

def gen_pseudoperiodic_signal(length, params):
    npt = 0
    time_sampler = ts.TimeSampler(stop_time=length)
    irregular_time_samples_pp = time_sampler.sample_irregular_time(num_points=length*2, keep_percentage=50)

    pseudo_periodic = ts.signals.PseudoPeriodic(
        amplitude=params["amplitude"], frequency=params["frequency"], freqSD=params["freqSD"], ampSD=params["ampSD"], ftype=params["ftype"])
    pp_timeseries = ts.TimeSeries(pseudo_periodic)
    pp_samples = pp_timeseries.sample(irregular_time_samples_pp)[0]

    if "growth" in params:
        start = time.time()
        trend = add_linear_trend(params["start"], params["growth"], length)
        data = np.add(pp_samples, trend)
        data = np.where(data > 0, data, data * -1)
        npt += time.time() - start
        return data, npt

    return pp_samples, npt
