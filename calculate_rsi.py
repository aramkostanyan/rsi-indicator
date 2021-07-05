"""
Calculate RSI from OHLCV data with the following parameters given in the
rsi_paremeters.ini configuration file:
CANDLE PERIOD, RSI LENGTH, OHLCV_FILENAME, LOG FILE, LOG LEVEL, START/END TIME
EXPORT FILENAME

Considerations: When adjusting the candle period the final candle might not
have exactly the number of intended aggregated candles/data entries.

Author: Aram Kostanyan - aramkostanyan@gmail.com
"""
import configparser
import json
import logging as log
import time

import numpy as np
import pandas as pd

# get configuration parameters
config = configparser.ConfigParser()
config.read("rsi_parameters.ini")

# Setup logging
LOG_FILE = config.get("RSI", "LOG_FILE")
level = log.getLevelName(config.get("RSI", "LOG_LEVEL"))
ch = log.StreamHandler()
fh = log.FileHandler(LOG_FILE)
log.basicConfig(
    encoding="utf-8",
    format="%(asctime)s %(levelname)s CalculateRSI - %(message)s",
    level=level,
    handlers=[ch, fh],
)


class CalculateRSI:
    def __init__(self):
        log.info("Initializing Parameters")

        self.candle_period = config.get("RSI", "CANDLE_PERIOD")
        self.rsi_length = config.getint("RSI", "RSI_LENGTH")
        self.ohlcv_filename = config.get("RSI", "OHLCV_FILENAME")

        self.start_time = config.get("RSI", "START_TIME")
        self.end_time = config.get("RSI", "END_TIME")

        self.export_filename = config.get("RSI", "EXPORT_FILENAME")
        if self.export_filename:
            log.info(
                "RSI with original data will be exported to: {}".format(
                    self.export_filename
                )
            )

    def run(self):
        """Main function calculated RSI"""

        while True:  # used for certain type of service implementation

            df = self.load_file2df(self.ohlcv_filename)
            log.debug("File loaded to dataframe of shape: {}".format(df.shape))

            if df.empty:
                message = "Empty Datafile: {}".format(self.ohlcv_filename)
                log.error(message)
                raise Exception(message)

            if not self.is_epoch_ms(df["_id"]):
                message = "The time stamp is not in ms"
                log.error(message)
                raise Exception(message)

            if self.candle_period:
                df = self.adjustCandlePeriod(df, self.candle_period)
                log.debug("Candle period adjusted to {}".format(self.candle_period))

            df["RSI"] = self.computeRSI(df["close"], self.rsi_length)
            log.debug("RSI calculated")

            if self.start_time or self.end_time:
                df = self.filterByDate(df, self.start_time, self.end_time)

            if self.export_filename:
                self.save_file(df, self.export_filename)
                log.debug("Calculated RSI saved to: {}".format(self.export_filename))

            log.info("RSI calculation successfull.")
            print(df.shape)
            time.sleep(60)

    def load_file2df(self, filename):
        """
        Returns dataframe from a json file with
        each record as a row.
        """
        try:
            with open(filename, "r") as f:
                df = pd.read_json(f)
        except Exception as e:
            log.exception("File not accessible: " + filename + " - " + str(e))
            raise

        return df

    def is_epoch_ms(self, df):
        if not str(df[0]).isdigit():
            return False
        if not len(str(df[0])) == 13:
            return False
        return True

    def save_file(self, df, filename):
        """
        Saves the DF to a json file with
        each row as a record.
        """

        result = df.to_json(orient="records")
        parsed = json.loads(result)

        with open(filename, "w") as f:
            json.dump(parsed, f, indent=4)

    def filterByDate(self, df, start_time=None, end_time=None):
        """
        Filter the data frame by start and end datetime.
        If no startime is given, return the df from the beginning.
        If no endtime is given, return the df to the end of file.
        """
        if start_time:
            try:
                start_epoch = (
                    pd.Timestamp(start_time) - pd.Timestamp("1970-01-01")
                ) // pd.Timedelta("1ms")
                df = df[df["_id"] >= start_epoch]
            except Exception as e:
                log.exception(
                    "Error at parsing and filtering by start date: {} - {}".format(
                        start_time, e
                    )
                )
                raise
            log.debug(
                "Data filtered from starting date of: {}".format(
                    pd.Timestamp(start_time)
                )
            )

        if end_time:
            try:
                end_epoch = (
                    pd.Timestamp(end_time) - pd.Timestamp("1970-01-01")
                ) // pd.Timedelta("1ms")
                df = df[df["_id"] <= end_epoch]
            except Exception as e:
                log.exception(
                    "Error at parsing and filtering by end date: {} - {}".format(
                        end_time, e
                    )
                )
                raise
            log.debug("Data filtered to end date of: {}".format(pd.Timestamp(end_time)))

        return df

    def adjustCandlePeriod(self, df, window_str):
        """
        Adjust the candle period of the df by combining the row
        following the aggregations of candle_aggr.
        Window is the number of rows to combine.
        """
        # calculate the time interval between two entries
        # this should be in ms
        dt = df["_id"][1] - df["_id"][0]
        log.debug("Candle period of the data: {}".format(dt))

        # transform the candle period from str to ms
        window_ms = self.get_window_ms(window_str)
        log.debug("Requested candle period: {}".format(window_ms))

        if window_ms <= 0:
            message = "Candle period should be positive - {} given.".format(window_ms)
            log.error(message)
            raise Exception(message)

        if dt > window_ms:
            message = "Time interval {} is larger than the requested candle period {}.".format(
                dt, window_ms
            )
            log.error(message)
            raise Exception(message)

        if dt == window_ms:
            log.debug("Requested time frame equates that of the data.")
            return df

        # calculate the window in terms of number of rows
        window = window_ms // dt

        # setup the aggregations for each column
        candle_aggr = {
            "_id": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "last",
            "quote_asset_volume": "sum",
            "number_of_trades": "sum",
        }

        try:
            res = df.groupby(np.arange(len(df)) // window).agg(candle_aggr)
        except Exception as e:
            log.exception("Error during dataframe aggregation - {}".format(e))
            raise

        return res

    def get_window_ms(self, timeframe):
        """Return extracted from string time delta in ms"""
        try:
            res = pd.Timedelta(value=timeframe).delta // 1_000_000
        except Exception as e:
            log.exception(
                "Timeframe parsing unsuccessfull: {} - {}".format(timeframe, e)
            )
            raise
        return res

    def computeRSI(self, data, time_window):
        """
        Compute the RSI using the given
        data set (closing prices) and time window, and
        return the RSI.
        """
        diff = data.diff(1).dropna()  # diff in one field

        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[diff > 0]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[diff < 0]

        # RMA (Running Moving average) with timewindow corresponds to
        # exponential exponential moving average with alpha = 1/timewindow
        up_chg_avg = up_chg.ewm(alpha=1 / time_window, min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(
            alpha=1 / time_window, min_periods=time_window
        ).mean()

        rs = abs(up_chg_avg / down_chg_avg)
        rsi = 100 - 100 / (1 + rs)
        return rsi


if __name__ == "__main__":
    rsi = CalculateRSI()
    rsi.run()
