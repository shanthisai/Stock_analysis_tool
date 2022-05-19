import ray
import pandas as pd
import numpy as np
import os
import re
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn import metrics
import traceback
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def pre_process_data(df, null_threshold):
    """
    Drops Date and Unix Date columns from the data.
    Drops the columns which has null values more than specified null_threshold.
    Replaces infinite values with NAN.
    Drops the rows which has null values.

    Parameters
    ----------
    data : dataframe

    null_threshold : numeric
        numeric value describing the amount of null values that can be present.

    Returns
    -------
    data : dataframe
        an updated dataframe after performing all the opertaions.
    """

    df.drop(columns=['Date'], axis=1, inplace=True)
    total = df.shape[0]
    for col in df.columns:
        if null_threshold * total / 100 < df[col].isnull().sum():
            df.drop(columns=[col], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df.columns] = (df[df.columns].astype(str)).apply(pd.to_numeric, errors='coerce')
    df.dropna(axis=0, inplace=True)
    return df


def error_metrics(y_true, y_pred):
    rmse = metrics.mean_squared_error(y_true, y_pred) ** 0.5
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2_score = metrics.r2_score(y_true, y_pred)
    return {"root_mean_squared_error": rmse, "mean_absolute_error": mae, "mean_squared_error": mse,
            "r2_score": r2_score}


def split_dataset(X, Y, t):
    tr = int(len(X) * t)
    tt = len(X) - tr
    xtr = X[:tr]
    xtt = X[tr:tr + tt]
    ytr = Y[:tr]
    ytt = Y[tr:tr + tt]
    return (xtr, xtt, ytr, ytt)


def remove_next_columns(df, column):
    cols = [col for col in df.columns if "next" not in col.lower()]
    cols.append(column)
    df = df[cols]
    return (df, column)


def remove_cp_columns(df):
    cols = [col for col in df.columns if not col.lower().startswith("cp")]
    df = df[cols]
    return df


def remove_previous_columns(df, column):
    cols = [col for col in df.columns if not col.lower().startswith("previous")]
    cols.append(column)
    df = df[cols]
    return df


def remove_max_avg_min_columns(df):
    cols = [col for col in df.columns if not (col.lower().startswith(
        "max") or col.lower().startswith("avg") or col.lower().startswith("min"))]
    df = df[cols]
    return df


def run_linear(X_train, X_test, Y_train, Y_test, num, col, symbol):
    linear_pipeline = Pipeline([("feature_selection", SequentialFeatureSelector(LinearRegression(
    ), n_jobs=None, n_features_to_select=num)), ("linear_regression", LinearRegression())])
    linear_pipeline.fit(X_train, Y_train)
    # pickle.dump(linear_pipeline, open(os.path.join(
    #     modelpath, str(symbol) + "_" + col + ".sav", 'wb')))
    Y_pred = linear_pipeline.predict(X_test)
    result = error_metrics(Y_test, Y_pred)
    selected_features = X_train.columns[linear_pipeline["feature_selection"].get_support(
    )].tolist()
    result.update({"selected_features": selected_features})
    result.update({"numoffeatures": len(selected_features)})
    result.update({"predicted_column": col})
    result.update({"model": "linear"})
    result.update({"actual": Y_test.values.tolist()})
    result.update({"predicted": Y_pred.tolist()})
    return result


def run_models(df, col, symbol):
    ref = df.copy()
    days = int(re.findall(r"\d+", col)[0])
    # start = df['Date'].iloc[0] + datetime.timedelta(days=days)
    # end = df['Date'].iloc[-1] - datetime.timedelta(days=days)
    # df = df[df.Date.between(start, end)]
    df = pre_process_data(df, 60)
    df[df.columns] = (df[df.columns].astype(str)).apply(
        pd.to_numeric, errors='coerce')
    df, column = remove_next_columns(df, col)
    X = df.drop(columns=[column])
    Y = df[column]
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y, 0.70)
    num = 0.33
    linres = run_linear(X_train, X_test, Y_train, Y_test,
                        num, column, symbol)
    linres.update({"close": ref.loc[X_test.index]
    ['Close'].values.tolist()})
    linres.update({"date": ref.loc[X_test.index]['Date'].apply(
        lambda row: row.strftime('%Y-%m-%d')).values.tolist()})

    return linres


# @ray.remote
def run_companies_lb(symbol, col):
    try:
        symbol = str(symbol)
        df = pd.read_csv(os.path.join(path, "gr_" + symbol + ".csv"))
        df = df.iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[necessary_columns]
        result = run_models(df, col, symbol)
        result.update({"company": symbol})
        return result
    except:
        return None


# @ray.remote
def run_companies_ub(symbol, col):
    try:
        symbol = str(symbol)
        df = pd.read_csv(os.path.join(path, "gr_" + symbol + ".csv"))
        df = df.iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[necessary_columns]
        result = run_models(df, col, symbol)
        result.update({"company": symbol})
        return result
    except:
        return None


def intial_run():
    fullresult = []
    for symbol in symbols:
        # try:
        print(symbol)
        lbresult = run_companies_lb(
            symbol, columns_to_predict[0])
        ubresult = run_companies_ub(
            symbol, columns_to_predict[1])
        result = ([lbresult, ubresult])
        if result[0] != None and result[1] != None:
            fullresult.extend(result)
        # except:
        #     traceback.print_exc()
    resultdf = pd.DataFrame(fullresult)
    resultdf.to_csv(os.path.join(os.getcwd(), "NASDAQ_Data",
                                 filename), index=None)


def create_files(days):
    if not os.path.exists(filename):
        os.makedirs(filename)
    df = pd.read_csv(os.path.join(npath, filename))

    ndx100 = pd.read_csv(os.path.join(npath, "equity_data.csv"))
    sorted_ndx = ndx100.sort_values(by=["Market Cap"], ascending=False)
    sorted_ndx = sorted_ndx[:101]
    index_li = []
    for index,row in sorted_ndx.iterrows():
        if row["Symbol"] in no_data:
            index_li.append(index)
    sorted_ndx.drop(sorted_ndx.index[index_li])
    cols = ['predicted_column', 'actual',
            'predicted', 'close', 'date', 'company']
    df = df[cols]
    # df['company'] = sorted_ndx["Symbol"].loc[:10].values.tolist()
    for n,g in df.groupby(by=['company']):
        try:
            print(n)
            g = g.reset_index(drop=True)
            lower = g.iloc[0] if "LB" in g["predicted_column"].iloc[0] else g.iloc[1]
            upper = g.iloc[0] if "UB" in g["predicted_column"].iloc[0] else g.iloc[1]

            date = [d.strip()
                    for d in lower['date'][1:-1].replace("\'", "").split(",")]
            close = [float(c.strip()) for c in lower['close'][1:-1].split(",")]
            actual_lb = [float(a.strip())
                         for a in lower['actual'][1:-1].split(",")]
            predicted_lb = [float(p.strip())
                            for p in lower['predicted'][1:-1].split(",")]
            actual_ub = [float(a.strip())
                         for a in upper['actual'][1:-1].split(",")]
            predicted_ub = [float(p.strip())
                            for p in upper['predicted'][1:-1].split(",")]

            cols = ["date", "close", "actual lb",
                    "predicted lb", "actual ub", "predicted ub"]
            refdf = pd.DataFrame(zip(date, close, actual_lb,
                                     predicted_lb, actual_ub, predicted_ub), columns=cols)

            refdf["actual ub close diff"] = abs(
                refdf["close"] - refdf["close"] * refdf["actual ub"])
            refdf["predicted ub close diff"] = abs(
                refdf["close"] - refdf["close"] * refdf["predicted ub"])
            refdf["actual lb close diff"] = abs(
                refdf["close"] - refdf["close"] * refdf["actual lb"])
            refdf["predicted lb close diff"] = abs(
                refdf["close"] - refdf["close"] * refdf["predicted lb"])
            refdf["predicted lb ub diff"] = refdf["predicted ub close diff"] - \
                                            refdf["predicted lb close diff"]
            refdf["predicted lb %"] = 1 - refdf["predicted lb"]
            refdf["predicted ub %"] = refdf["predicted ub"] - 1
            refdf["invest"] = refdf.apply(lambda row: True if row["predicted lb %"] < 0.01 and (
                    row["predicted ub %"] - row["predicted lb %"]) > 0.1 else False, axis=1)
            refdf["exit"] = refdf.apply(lambda row: True if row["predicted ub %"] < 0.01 and (
                    row["predicted ub %"] + row["predicted lb %"]) > 0.05 else False, axis=1)
            if os.path.exists(os.path.join(simpath, n + "_" + str(days) + ".csv")):
                os.remove(os.path.join(simpath, n + "_" + str(days) + ".csv"))
                refdf.to_csv(os.path.join(simpath, n + "_" + str(days) + ".csv"), index=None)
            else:
                refdf.to_csv(os.path.join(simpath, n + "_" + str(days) + ".csv"), index=None)
        except Exception as e:
            print(e)


necessary_columns = ["Date", "Close", "Previous 360 days UB", "Min Inc % in 180 days", "Next 60 days LB", "Previous 720 days UB", "CP % LV 180 days", "Max Inc % in 180 days", "Next 1080 days LB", "CP % BA 180 days", "Next Day Low GR", "Max Dec % in 90 days", "Total Operating Expenses Gr", "CP % HV 90 days", "Min Dec % in 365 days", "Max Dec % in 365 days", "CP % HV 7 days", "CP % BA 7 days", "Avg Inc % in 365 days", "Min Inc % in 90 days", "Avg Inc % in 180 days", "Low GR", "Previous 1080 days UB", "CP % HV 180 days", "Next 180 days UB", "Previous 60 days UB", "CP % BA 90 days", "Avg Inc % in 90 days", "Sequential Increase %", "CP % BA 30 days", "Avg Dec % in 180 days", "Previous 720 days LB", "EPS Gr", "Next 360 days UB", "CP % HV 365 days", "Spread Close-Open GR", "Min Dec % in 180 days", "Next 30 days LB", "Sequential Increase", "Previous 360 days LB",
                     "Alpha GR", "CP % LV 365 days", "Dividend Gr", "Sequential Decrease", "Next 360 days LB", "Avg Dec % in 365 days", "Gross Profit Gr", "CP % LV 7 days", "CP % HV 30 days", "Min Inc % in 365 days", "Sequential Decrease %", "Beta GR", "Next 30 days UB", "High GR", "Spread High-Low GR", "Net Income Gr", "Max Dec % in 180 days", "Previous 30 days UB", "Next 90 days UB", "Next 90 days LB", "Next 1080 days UB", "Open GR", "Next 720 days LB", "Max Inc % in 365 days", "Previous 90 days LB", "Previous 90 days UB", "Next 60 days UB", "Avg Dec % in 90 days", "Previous 30 days LB", "Previous 1080 days LB", "Next Day Open GR", "Next Day High GR", "CP % BA 365 days", "Max Inc % in 90 days", "Total Revenue Gr", "CP % LV 30 days", "Min Dec % in 90 days", "Next 180 days LB", "Previous 180 days LB", "Close GR", "CP % LV 90 days", "Previous 60 days LB", "Previous 180 days UB", "Next 720 days UB", "Next Day Close GR"]

for d in [30,60,90,180,360,720,1080]:
    columns_to_predict = ['Next {} days LB'.format(d), 'Next {} days UB'.format(d)]
    days = d
    npath = os.path.join(os.getcwd(),"NASDAQ_Data")
    simpath = os.path.join(os.getcwd(), "NASDAQ_Data", "Simulation")
    if not os.path.exists(simpath):
        os.makedirs(simpath)
    path = os.path.join(os.getcwd(), "NASDAQ_Data", "GRStock")
    df = pd.read_csv(os.path.join(npath, "equity_data.csv"))
    # download_index_data()
    sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
    symbols = sorted_df["Symbol"].values.tolist()
    symbols = symbols[:101]
    no_data = ["HDB","RIO","SNY","UL","DEO","CVX","BHP","T"]
    for i in no_data:
        if i in symbols:
            symbols.remove(i)
    # ray.init(ignore_reinit_error=True)
    intial_run()
    filename = "next_{}_days.csv".format(days)
    create_files(days)