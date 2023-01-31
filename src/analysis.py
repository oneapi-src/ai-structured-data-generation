# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions for basic data analysis given a generated dataset.
'''
def find_skewness(data):
    return stats.skew(data), stats.kurtosis(data, fisher=True)

def f(x, a, b, c):
    return a*x**2 + b*x + c

def residual(p, x, y):
    return y - f(x, *p)

def find_trend(time_samples, signal):
    p0 = [1., 1., 1.]
    start_time = time.time()
    popt = optimize.leastsq(residual, p0, args=(time_samples, signal))[0]
    
    return popt, time.time() - start_time

def plot_mean(x, data_type, dataset_size):
    plt.figure(figsize=(10, 5))
    plt.hist(x)
    plt.xlabel("Means of Individual Signals")
    plt.ylabel("Count")
    plt.savefig(data_type + "_Signal_Mean.png")

def plot_var(x, data_type, dataset_size):
    plt.figure(figsize=(10, 5))
    plt.hist(x)
    plt.xlabel("Variances of Individual Signals")
    plt.ylabel("Count")
    plt.savefig(data_type + "_Signal_Var.png")

def exploratory_stats(sig_arr):
    results = {}
    results["Means"] = np.mean(sig_arr, axis=1)
    results["Variances"] = np.var(sig_arr, axis=1)
    results["Left_Vectors"], results["Singular_Values"], results["Right_Vectors"] = np.linalg.svd(sig_arr)
    sig_range = []

    for sig in sig_arr:
        sig_range.append([np.min(sig), np.max(sig)])
    results["Signal_Ranges"] = sig_range

    norm = np.linalg.norm(sig_arr)
    norm_sig = sig_arr/norm

    return results, norm_sig

if __name__ == "__main__":
    import argparse
    import logging
    import pathlib
    import warnings
    import time
    import sys
    import numpy as np
    from scipy import stats  # pylint: disable=E0401
    from scipy import optimize  # pylint: disable=E0401
    import matplotlib.pyplot as plt  # pylint: disable=E0401
    import matplotlib.backends.backend_pdf  # pylint: disable=E0401
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore")

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies")
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default="",
                        help="data csv file to analyze")

    FLAGS = parser.parse_args()
    INTEL_FLAG = FLAGS.intel
    data_file = FLAGS.data

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
    logging.getLogger('matplotlib.*').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('modin').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    if data_file == "":
        logger.info("no data provided, please run script again with the data's location")
        sys.exit()
    else:
        path = pathlib.Path(data_file)
        if not path.exists():
            logger.info("invalid path provided, please run script with a valid data location")
            sys.exit()

    start_time = time.time()
    data = pd.read_csv(data_file, index_col=False)
    pd_time += time.time() - start_time

    if INTEL_FLAG:
        logger.info("Modin time for read csv: %f", pd_time)
        start_time = time.time()
        import pandas as pd
        data = data._to_pandas()  # pylint: disable=W0212
        logger.info("converting modin to pandas: %f", time.time() - start_time)
    else:
        logger.info("Pandas time for read csv: %f", pd_time)

    ts_cols = []
    tab_cols = []
    for col in list(data.columns):
        if "timeseries" in col.lower():
            ts_cols.append(col)
        elif "id" not in col.lower() and str(data[col][0]).isdigit():
            tab_cols.append(col)
    
    print("Running Data Analysis for Time Series Columns")
    for col in tqdm(ts_cols):
        num_sigs = len(data[col])
        start_time = time.time()
        sigs = [np.fromstring(data[col][i][1:-1], sep=' ') for i in range(num_sigs)]
        explore, normal = exploratory_stats(sigs)
        npt = time.time() - start_time
        np_time += npt

        logger.info("Executed exploratory analysis for %s data that includes mean, range, variance, and SVD", col)
        logger.info("Executed normalization of %s data", col)
        logger.info("Exploratory analysis and normalization for %s data using numpy took %f secs", col, npt)

        plot_mean(explore["Means"], col, num_sigs)
        plot_var(explore["Variances"], col, num_sigs)

        pdf = matplotlib.backends.backend_pdf.PdfPages(col+"_trends.pdf")
        num_points = len(sigs[0])
        x = np.linspace(0, num_points, num_points)
        y_funs = {}
        for i in tqdm(range(num_sigs)):
            coeffs, spt = find_trend(x, sigs[i])
            sp_time += spt
            y_funs[i] = f(x, coeffs[0], coeffs[1], coeffs[2])
            fig = plt.figure(figsize=(10, 5))
            plt.plot(x, sigs[i], marker='o', markersize=4)
            plt.plot(x, y_funs[i])
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title(col + " signal trendline")
            pdf.savefig(fig)
        pdf.close()
    
    print("Running Data Analysis for Other Tabular Columns")
    for col in tqdm(tab_cols):
        start_time = time.time()
        skew, kur = find_skewness(data[col].to_numpy().astype(float))
        sp_time += time.time() - start_time

        logger.info("Skewness for %s: %f", col, skew)
        logger.info("Kurtosis for %s: %f", col, kur)

    logger.info("Total scipy time: %f secs", sp_time)
    logger.info("Total numpy time: %f secs", np_time)
