import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys

from simtools import log_message

# VWAP with factor affecting theoretical value

matplotlib.rcParams['figure.figsize'] = (14, 6)

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Record a trade in our trade array
def record_trade(trade_df, idx, trade_px, trade_qty, current_bar, order_side):
    trade_df.loc[idx] = [trade_px, trade_qty, current_bar, order_side]

    return


# MAIN ALGO LOOP
def algo_loop(trading_day, last_min_penalty, round_lot=100., tick_coef=0., model_coef=0., using_model=False):
    log_message('Beginning Trading run:')

    round_lot = round_lot
    avg_spread = (trading_day.ask_px - trading_day.bid_px).rolling(100, min_periods=1).mean().fillna(0)
    half_spreads = avg_spread / 2
    trading_day["half_spread"] = half_spreads
    # print("Average stock spread for sample: {:.4f}".format(avg_spread))

    # caclulate last_minutes_from_open in a specific trading day
    last_time_from_open = (trading_day.index[-1] - pd.Timedelta(hours=9, minutes=30))
    last_minutes_from_open = (last_time_from_open.hour * 60) + last_time_from_open.minute


    # init our price and volume variables
    [last_price, last_size, bid_price, bid_size, ask_price, ask_size, volume] = np.zeros(7)

    # init some time series objects for collection of telemetry
    fair_values = pd.Series(index=trading_day.index)
    midpoints = pd.Series(index=trading_day.index)
    tick_factors = pd.Series(index=trading_day.index)
    model_factors = pd.Series(index=trading_day.index)


    # order side
    live_order_side_s_price, live_order_side_b_price = 0., 0.
    live_order_side_s_quantity, live_order_side_b_quantity = 0., 0.


    # let's set up a container to hold trades. preinitialize with the index
    trades = pd.DataFrame(columns=['price', 'shares', 'bar', 'order_side'])


    # MAIN EVENT LOOP
    current_bar = 0


    # track state and values for a current working order
    current_holdings = 0

    # other order and market variables
    total_trade_count = 0
    total_buy_count = 0
    total_sell_count = 0

    # fair value pricing variables
    midpoint = 0.0
    fair_value = 0.0

    # define our accumulator for the tick EMA
    message_type = 0
    # tick_coef = 1.
    tick_window = 20
    tick_factor = 0
    tick_ema_alpha = 2 / (tick_window + 1)
    prev_tick = 0
    prev_price = 0

    last_index = trading_day.index[-1]

    log_message('starting main loop')
    for index, row in trading_day.iterrows():
        # get the time of this message
        time_from_open = (index - pd.Timedelta(hours=9, minutes=30))
        minutes_from_open = (time_from_open.hour * 60) + time_from_open.minute

        # MARKET DATA HANDLING
        if pd.isna(row.trade_px):  # it's a quote
            # skip if not NBBO
            if not ((row.qu_source == 'N') and (row.natbbo_ind == 4)):
                continue
            # set our local NBBO variables
            if (row.bid_px > 0 and row.bid_size > 0):
                bid_price = row.bid_px
                bid_size = row.bid_size * round_lot
            if (row.ask_px > 0 and row.ask_size > 0):
                ask_price = row.ask_px
                ask_size = row.ask_size * round_lot
            quote_count += 1
            message_type = 'q'
        else:  # it's a trade
            # store the last trade price
            prev_price = last_price
            # now get the new data
            last_price = row.trade_px
            last_size = row.trade_size
            trade_count += 1
            cumulative_volume += row.trade_size
            vwap_numerator += last_size * last_price
            message_type = 't'

            # CHECK OPEN ORDER(S) if we have a live order,
            # has it been filled by the trade that just happened?
            trading_quantity = live_order_side_b_quantity - live_order_side_s_quantity
            # trading quantity will base on current holdings
            # will decide later
            if trading_quantity > 0:
                if (last_price <= live_order_side_b_price):
                    fill_size = min(trading_quantity, last_size)
                    record_trade(trades, index, live_order, fill_size, current_bar, 'b')

                    # even if we only got partially filled, let's assume we're cancelling the entire quantity.
                    # If we're still behind we'll replace later in the loop
                    live_order_side_s_quantity, live_order_side_b_quantity = 0.0
                    live_order_side_s_price, live_order_side_b_price = 0., 0.
            elif trading_quantity < 0:
                if (last_price >= live_order_side_s_price):
                    fill_size = min(live_order_quantity, last_size)
                    record_trade(trades, index, live_order_price, fill_size, current_bar, order_side)
                    total_quantity_filled += fill_size
                    quantity_remaining = max(0, quantity_remaining - fill_size)
                    # print("{} passive. quantity_behind:{} new_order_quantity:{}".format(index, quantity_behind,
                    #                                                                     fill_size))
                    total_pass_count += 1

                    # even if we only got partially filled, let's assume we're cancelling the entire quantity.
                    # If we're still behind we'll replace later in the loop
                    live_order = False
                    live_order_price = 0.0
                    live_order_quantity = 0.0
                    quantity_behind = current_target_shares - total_quantity_filled


        # TICK FACTOR
        # only update if it's a trade
        if message_type == 't':
            # calc the tick
            this_tick = np.sign(last_price - prev_price)
            if this_tick == 0:
                this_tick = prev_tick

            # now calc the tick
            if tick_factor == 0:
                tick_factor = this_tick
            else:
                tick_factor = (tick_ema_alpha * this_tick) + (1 - tick_ema_alpha) * tick_factor

                # store the last tick
            prev_tick = this_tick

        # PRICING LOGIC
        new_midpoint = bid_price + (ask_price - bid_price) / 2
        if new_midpoint > 0:
            midpoint = new_midpoint

        if using_model:
            if not pd.isna(row.signal_discrete):
                model_factor = row.signal_discrete
            else:
                model_factor = 0
        else:
            model_coef = 0
            model_factor = 0

        # FAIR VALUE CALCULATION
        # check inputs, skip of the midpoint is zero, we've got bogus data (or we're at start of day)
        if midpoint == 0:
            # print("{} no midpoint. b:{} a:{}".format(index, bid_price, ask_price))
            continue

        half_spread = row.half_spread

        fair_value = midpoint +  (tick_coef * tick_factor * half_spread) + (model_coef * model_factor * half_spread)
        # collect our data
        fair_values[index] = fair_value
        midpoints[index] = midpoint
        tick_factors[index] = tick_factor
        model_factors[index] = model_factor

        # TRADING LOGIC
        if index == last_index:
            new_order_quantity = current_holdings
            if current_holdings > 0:
                new_trade_price = bid_price
                record_trade(trades, index, new_trade_price, new_order_quantity, current_bar, 'a')
            elif current_holdings < 0:
                new_trade_price = ask_price
                record_trade(trades, index, new_trade_price, new_order_quantity, current_bar, 'a')
            break

        if (fair_value >= ask_price or current_bar == last_minutes_from_open) and quantity_behind > round_lot:
            live_order_side = 'b'

            live_order_quantity = 1.
            live_order_price = ask_price
            live_order = True


        if (fair_value <= bid_price or current_bar == last_minutes_from_open) and quantity_behind > round_lot:
            live_order_side = 's'

            live_order_quantity = 1.
            live_order_price = ask_price
            live_order = True


        else:
            # we shouldn't have got here, something is wrong with our order type
            print('Got an unexpected order_side value: ' + str(order_side))

    # looping done
    log_message('end simulation loop')
    log_message('order analytics')

    # Now, let's look at some stats
    trades = trades.dropna()
    day_vwap = vwap_numerator / cumulative_volume

    # prep our text output
    avg_price = (trades['price'] * trades['shares']).sum() / trades['shares'].sum()

    log_message('VWAP run complete.')

    nforward = 10
    mysign = lambda x: 0 if abs(x) < 1e-3 else (1 if x > 0 else -1)
    labels = (trading_day["trade_px"].rolling(nforward).mean().shift(-nforward) - trading_day["trade_px"]).apply(mysign)
    labels.fillna(method="ffill", inplace=True)

    # assemble results and return

    return {'midpoints': midpoints,
            'fair_values': fair_values,
            'schedule_factors': schedule_factors,
            'tick_factors': tick_factors,
            'model_factors': model_factors,
            'trades': trades,
            'correct_movements': labels,
            'quote_count': quote_count,
            'day_vwap': day_vwap,
            'avg_price': avg_price,
            'avg_spread': avg_spread,
            'last_min_order': last_min_order,
            }
