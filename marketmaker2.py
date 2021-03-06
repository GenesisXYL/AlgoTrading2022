import datetime
import numpy as np
import pandas as pd
import time
import sys

from simtools import log_message

# Lee-Ready tick strategy simulator

# Record a trade in our trade array
def record_trade( trade_df, idx, trade_px, trade_qty, current_bar, trade_type, side ):
    #print( "Trade! {} {} {} {}".format( idx, trade_px, trade_qty, current_bar ) )
    trade_df.append([ trade_px, trade_qty, current_bar, trade_type ])

    return

    
# MAIN ALGO LOOP
# trading_day: our data
# tick_coef: strength of the tick signal
def algo_loop( trading_day, tick_coef=0.0, risk_coef=0.0, max_position_dollars=100000, peg_to_bbo=False, model_coef = True, using_model = True, round_lot = 1. ):
    log_message( 'Beginning Market-Making Strategy run' )

    round_lot = round_lot
    avg_spread = ( trading_day.ask_px - trading_day.bid_px ).mean()
    half_spread = avg_spread / 2
    print( "Average stock spread for sample: {:.4f}".format(avg_spread) )

    # this should be a higher hard constraint but for now let's just leave it at a fixed amount
    live_order_quantity = 200
    
    # init our price and volume variables
    [ last_price, last_size, bid_price, bid_size, ask_price, ask_size, volume ] = np.zeros(7)

    # init our counters
    [ trade_count, quote_count, cumulative_volume ] = [ 0, 0, 0 ]
    
    
    # let's set up a container to hold trades. preinitialize with the index
    buys = []
    sells = []
    # leaving this as a DataFrame since I may want to add other attributes
    
    current_net_pos = 0
    
    # MAIN EVENT LOOP
    current_bar = 0

    # other order and market variables
    total_quantity_bought = 0
    total_quantity_sold = 0
    vwap_numerator = 0.0

    total_trade_count = 0
    total_agg_count = 0
    total_pass_count = 0

    # fair value pricing variables
    midpoint = 0.0
    fair_value = 0.0
    
    # define our accumulator for the tick EMA
    message_type = 0   
    tick_window = 20
    tick_factor = 0
    tick_ema_alpha = 2 / ( tick_window + 1 )
    prev_tick = 0
    prev_price = 0
    
    # quoting related
    our_bid = 0.0
    our_ask = 0.0
    half_spread_coef = 0.5
    
    # risk factor for part 2
    risk_factor = 0.0
    
    #log_message( 'tick_coef: {:.2f} risk_coef: {:.2f} max_position_dollars: {}'.format(tick_coef, risk_coef, max_position_dollars))

    log_message( 'starting main loop' )
    for index, row in trading_day.iterrows():
        # get the time of this message
        time_from_open = (index - pd.Timedelta( hours = 9, minutes = 30 ))
        minutes_from_open = (time_from_open.hour * 60) + time_from_open.minute
        
        # MARKET DATA HANDLING
        if pd.isna( row.trade_px ): # it's a quote
            # skip if not NBBO
            if not ( ( row.qu_source == 'N' ) and ( row.natbbo_ind == 4 ) ):
                continue
            # set our local NBBO variables
            if ( row.bid_px > 0 and row.bid_size > 0 ):
                bid_price = row.bid_px
                bid_size = row.bid_size * round_lot
            if ( row.ask_px > 0 and row.ask_size > 0 ):
                ask_price = row.ask_px
                ask_size = row.ask_size * round_lot
            quote_count += 1
            message_type = 'q'
        else: # it's a trade
            # store the last trade price
            prev_price = last_price
            # now get the new data
            last_price = row.trade_px
            last_size = row.trade_size
            trade_count += 1
            cumulative_volume += row.trade_size
            vwap_numerator += last_size * last_price
            message_type = 't'

            # check the trade against our quotes
            # for now assume we always have a quote out there
            # has our quote been hit or lifted?
        
            # check bid
            if ( last_price <= our_bid ) & ( bid_price > 0 ): # our bid was hit
                fill_size = max( min( live_order_quantity, last_size ), round_lot )
                record_trade( buys, index, our_bid, fill_size, current_bar, 'p', 'b' )
                current_net_pos += fill_size
                #print("buy: NBO: {} trade size: {} net_pos: {}".format( our_bid, fill_size, current_net_pos ) )
            # check offer
            if ( last_price >= our_ask ) & ( ask_price > 0 ): # our ask was lifted
                fill_size = max( min( live_order_quantity, last_size ), round_lot )
                record_trade( sells, index, our_ask, fill_size, current_bar, 'p', 's' )
                current_net_pos -= fill_size
                #print("sell: NBO: {} trade size: {} net_pos: {}".format( our_ask, fill_size, current_net_pos ) )
        

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
                tick_factor = ( tick_ema_alpha * this_tick ) + ( 1 - tick_ema_alpha ) * tick_factor    
            
            # store the last tick
            prev_tick = this_tick
            
        # RISK FACTOR     
        # simple factor- reaching max position should an immediate spread crossing
        risk_factor = -1 * (current_net_pos * last_price) / max_position_dollars
        
        # MODEL FACTOR
        if using_model:
            if not pd.isna(row.signal_discrete):
                model_factor = row.signal_discrete
            else:
                model_factor = 0
        else:
            model_coef = 0
            model_factor = 0
        
        # long, lower TV
        # Short, raise TV
            
        # PRICING LOGIC
        new_midpoint = bid_price + ( ask_price - bid_price ) / 2
        if new_midpoint > 0:
            midpoint = new_midpoint
        
        # FAIR VALUE CALCULATION
        # check inputs, skip of the midpoint is zero, we've got bogus data (or we're at start of day)
        if midpoint == 0:
            #print( "{} no midpoint. b:{} a:{}".format( index, bid_price, ask_price ) )
            continue
        fair_value = midpoint + (half_spread * ((tick_coef * tick_factor) + (risk_coef * risk_factor) 
                                                + (model_coef*model_factor)))
        #print('rf: {} net_pos: {}'.format(risk_factor, current_net_pos))

        #print(risk_factors[index])
        
        # QUOTE PLACEMENT
        # for now we're going to assume no hedging or aggressive trading at all.
        
        # in this version just ignore fair value and always set our price to the BBO
        if peg_to_bbo:
            #print("Pegging to NBBO")
            our_bid = bid_price
            our_ask = ask_price
            
        # use our fair value
        else: 
            # in this version, bid and offer are placed around the fair value but constrained by nbbo
            our_bid = round( min( fair_value - ( half_spread_coef * half_spread ), bid_price ), 2 )
            our_ask = round( max( fair_value + ( half_spread_coef * half_spread ) , ask_price ), 2 )
            #print("bid: {} ask: {}".format(our_bid, our_ask))

    # looping done
    log_message( 'end simulation loop' )
    log_message( 'order analytics' )
    
    
    # last order
    last_order = 0
    
    # add a final trade for net position at m2m to do simple P&L closeout
    if current_net_pos > 0: # we need to sell
        record_trade( sells, index, bid_price, abs(current_net_pos), current_bar, 'p', 's' )
        last_order = abs(current_net_pos)
        print( "selling to close residual of {} shares".format( current_net_pos ) )
        
        current_net_pos = 0
    elif current_net_pos < 0: # we need to buy
        record_trade( buys, index, ask_price, abs(current_net_pos), current_bar, 'p', 'b' )
        last_order = abs(current_net_pos)
        print( "buying to close residual of {} shares".format( current_net_pos ) )
        current_net_pos = 0
    
    
    # Turn list into dataframe
    buys = pd.DataFrame( buys, columns = [ 'price' , 'shares', 'bar', 'trade_type' ])
    sells = pd.DataFrame(sells, columns = [ 'price' , 'shares', 'bar', 'trade_type' ])
    
    # Now, let's look at some stats
    # clean up all the ticks in which we didn't do trades
    buys = buys.dropna()
    sells = sells.dropna()

    log_message( 'Algo run complete.' )
    
    avg_sell = ( sells.price * sells.shares ).sum() / sells.shares.sum()
    avg_buy = ( buys.price * buys.shares ).sum() / buys.shares.sum()
    PNL = avg_sell - avg_buy

    # assemble results and return
    # TODO: add P&L
    return { 
             'PNL' : PNL,
             'last_order' : last_order,
           }