import pandas as pd
from smartmoneyconcepts import smc

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

def liquidity(
    ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01
) -> DataFrame:
    """
    Liquidity
    Liquidity is when there are multiple highs within a small range of each other.
    or multiple lows within a small range of each other.

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    range_percent: float - the percentage of the range to determine liquidity

    returns:
    Liquidity = 1 if bullish liquidity, -1 if bearish liquidity
    Level = the level of the liquidity
    End = the index of the last liquidity level
    Swept = the index of the candle that swept the liquidity
    """

    swing_highs_lows = swing_highs_lows.copy()

    # Subtract the highest high from the lowest low
    pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * range_percent

    # Initialize arrays for liquidity calculations
    liquidity = np.zeros(len(ohlc), dtype=np.int32)
    liquidity_level = np.zeros(len(ohlc), dtype=np.float32)
    liquidity_end = np.zeros(len(ohlc), dtype=np.int32)
    liquidity_swept = np.zeros(len(ohlc), dtype=np.int32)

    for i in range(len(ohlc)):
        if i >= len(swing_highs_lows):
            print(f"Skipping index {i}, out of bounds for swing_highs_lows.")
            continue
        if swing_highs_lows["HighLow"].iloc[i] == 1:
            high_level = swing_highs_lows["Level"].iloc[i]
            range_low = high_level - pip_range
            range_high = high_level + pip_range
            temp_liquidity_level = [high_level]
            start = i
            end = i
            swept = 0
            for c in range(i + 1, len(ohlc)):
                if c >= len(swing_highs_lows):
                    print(f"Skipping index {c}, out of bounds for swing_highs_lows.")
                    continue
                if (
                    swing_highs_lows["HighLow"].iloc[c] == 1
                    and range_low <= swing_highs_lows["Level"].iloc[c] <= range_high
                ):
                    end = c
                    temp_liquidity_level.append(swing_highs_lows["Level"].iloc[c])
                    swing_highs_lows.at[c, "HighLow"] = 0
                if ohlc["high"].iloc[c] >= range_high:
                    swept = c
                    break
            if len(temp_liquidity_level) > 1:
                average_high = sum(temp_liquidity_level) / len(temp_liquidity_level)
                liquidity[i] = 1
                liquidity_level[i] = average_high
                liquidity_end[i] = end
                liquidity_swept[i] = swept

    # Now do the same for the lows
    for i in range(len(ohlc)):
        if i >= len(swing_highs_lows):
            print(f"Skipping index {i}, out of bounds for swing_highs_lows.")
            continue
        if swing_highs_lows["HighLow"].iloc[i] == -1:
            low_level = swing_highs_lows["Level"].iloc[i]
            range_low = low_level - pip_range
            range_high = low_level + pip_range
            temp_liquidity_level = [low_level]
            start = i
            end = i
            swept = 0
            for c in range(i + 1, len(ohlc)):
                if c >= len(swing_highs_lows):
                    print(f"Skipping index {c}, out of bounds for swing_highs_lows.")
                    continue
                if (
                    swing_highs_lows["HighLow"].iloc[c] == -1
                    and range_low <= swing_highs_lows["Level"].iloc[c] <= range_high
                ):
                    end = c
                    temp_liquidity_level.append(swing_highs_lows["Level"].iloc[c])
                    swing_highs_lows.at[c, "HighLow"] = 0
                if ohlc["low"].iloc[c] <= range_low:
                    swept = c
                    break
            if len(temp_liquidity_level) > 1:
                average_low = sum(temp_liquidity_level) / len(temp_liquidity_level)
                liquidity[i] = -1
                liquidity_level[i] = average_low
                liquidity_end[i] = end
                liquidity_swept[i] = swept

    liquidity = np.where(liquidity != 0, liquidity, np.nan)
    liquidity_level = np.where(~np.isnan(liquidity), liquidity_level, np.nan)
    liquidity_end = np.where(~np.isnan(liquidity), liquidity_end, np.nan)
    liquidity_swept = np.where(~np.isnan(liquidity), liquidity_swept, np.nan)

    liquidity_series = pd.Series(liquidity, name="Liquidity")
    level_series = pd.Series(liquidity_level, name="Level")
    liquidity_end_series = pd.Series(liquidity_end, name="End")
    liquidity_swept_series = pd.Series(liquidity_swept, name="Swept")

    return pd.concat(
        [liquidity_series, level_series, liquidity_end_series, liquidity_swept_series], axis=1
    )

def ob(
    ohlc: DataFrame,
    swing_highs_lows: DataFrame,
    close_mitigation: bool = False,
) -> DataFrame:
    """
    OB - Order Blocks
    This method detects order blocks when there is a high amount of market orders exist on a price range.

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.

    returns:
    OB = 1 if bullish order block, -1 if bearish order block
    Top = top of the order block
    Bottom = bottom of the order block
    OBVolume = volume + 2 last volumes amounts
    Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume,lowVolume))
    """

    swing_highs_lows = swing_highs_lows.copy()

    crossed = np.full(len(ohlc), False, dtype=bool)
    ob = np.zeros(len(ohlc), dtype=np.int32)
    top = np.zeros(len(ohlc), dtype=np.float32)
    bottom = np.zeros(len(ohlc), dtype=np.float32)
    obVolume = np.zeros(len(ohlc), dtype=np.float32)
    lowVolume = np.zeros(len(ohlc), dtype=np.float32)
    highVolume = np.zeros(len(ohlc), dtype=np.float32)
    percentage = np.zeros(len(ohlc), dtype=np.float32)  # Changed to float
    mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
    breaker = np.full(len(ohlc), False, dtype=bool)

    for i in range(len(ohlc)):
        close_index = i
        close_price = ohlc["close"].iloc[close_index]

        # Bullish Order Block
        if len(ob[ob == 1]) > 0:
            for j in range(len(ob) - 1, -1, -1):
                if ob[j] == 1:
                    currentOB = j
                    if breaker[currentOB]:
                        if ohlc.high.iloc[close_index] > top[currentOB]:
                            ob[j] = top[j] = bottom[j] = obVolume[j] = lowVolume[
                                j
                            ] = highVolume[j] = mitigated_index[j] = percentage[
                                j
                            ] = 0.0

                    elif (
                        not close_mitigation
                        and ohlc["low"].iloc[close_index] < bottom[currentOB]
                    ) or (
                        close_mitigation
                        and min(
                            ohlc["open"].iloc[close_index],
                            ohlc["close"].iloc[close_index],
                        )
                        < bottom[currentOB]
                    ):
                        breaker[currentOB] = True
                        mitigated_index[currentOB] = close_index - 1
        last_top_index = None
        for j in range(len(swing_highs_lows["HighLow"])):
            if swing_highs_lows["HighLow"][j] == 1 and j < close_index:
                last_top_index = j
        if last_top_index is not None:
            swing_top_price = ohlc["high"].iloc[last_top_index]
            if close_price > swing_top_price and not crossed[last_top_index]:
                crossed[last_top_index] = True
                obBtm = ohlc["high"].iloc[close_index - 1]
                obTop = ohlc["low"].iloc[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_top_index):
                    obBtm = min(
                        ohlc["low"].iloc[last_top_index + j],
                        obBtm,
                    )
                    if obBtm == ohlc["low"].iloc[last_top_index + j]:
                        obTop = ohlc["high"].iloc[last_top_index + j]
                    obIndex = (
                        last_top_index + j
                        if obBtm == ohlc["low"].iloc[last_top_index + j]
                        else obIndex
                    )

                ob[obIndex] = 1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                    + ohlc["volume"].iloc[close_index - 2]
                )
                lowVolume[obIndex] = ohlc["volume"].iloc[close_index - 2]
                highVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                )
                percentage[obIndex] = (
                    np.min([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                    / np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                ) * 100.0

    for i in range(len(ohlc)):
        close_index = i
        close_price = ohlc["close"].iloc[close_index]

        # Bearish Order Block
        if len(ob[ob == -1]) > 0:
            for j in range(len(ob) - 1, -1, -1):
                if ob[j] == -1:
                    currentOB = j
                    if breaker[currentOB]:
                        if ohlc.low.iloc[close_index] < bottom[currentOB]:
                            ob[j] = top[j] = bottom[j] = obVolume[j] = lowVolume[
                                j
                            ] = highVolume[j] = mitigated_index[j] = percentage[
                                j
                            ] = 0.0

                    elif (
                        not close_mitigation
                        and ohlc["high"].iloc[close_index] > top[currentOB]
                    ) or (
                        close_mitigation
                        and max(
                            ohlc["open"].iloc[close_index],
                            ohlc["close"].iloc[close_index],
                        )
                        > top[currentOB]
                    ):
                        breaker[currentOB] = True
                        mitigated_index[currentOB] = close_index
        last_btm_index = None
        for j in range(len(swing_highs_lows["HighLow"])):
            if swing_highs_lows["HighLow"][j] == -1 and j < close_index:
                last_btm_index = j
        if last_btm_index is not None:
            swing_btm_price = ohlc["low"].iloc[last_btm_index]
            if close_price < swing_btm_price and not crossed[last_btm_index]:
                crossed[last_btm_index] = True
                obBtm = ohlc["low"].iloc[close_index - 1]
                obTop = ohlc["high"].iloc[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_btm_index):
                    obTop = max(ohlc["high"].iloc[last_btm_index + j], obTop)
                    obBtm = (
                        ohlc["low"].iloc[last_btm_index + j]
                        if obTop == ohlc["high"].iloc[last_btm_index + j]
                        else obBtm
                    )
                    obIndex = (
                        last_btm_index + j
                        if obTop == ohlc["high"].iloc[last_btm_index + j]
                        else obIndex
                    )

                ob[obIndex] = -1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                    + ohlc["volume"].iloc[close_index - 2]
                )
                lowVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                )
                highVolume[obIndex] = ohlc["volume"].iloc[close_index - 2]
                percentage[obIndex] = (
                    np.min([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                    / np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                ) * 100.0

    ob = np.where(ob != 0, ob, np.nan)
    top = np.where(~np.isnan(ob), top, np.nan)
    bottom = np.where(~np.isnan(ob), bottom, np.nan)
    obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
    mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
    percentage = np.where(~np.isnan(ob), percentage, np.nan)

    ob_series = pd.Series(ob, name="OB")
    top_series = pd.Series(top, name="Top")
    bottom_series = pd.Series(bottom, name="Bottom")
    obVolume_series = pd.Series(obVolume, name="OBVolume")
    mitigated_index_series = pd.Series(mitigated_index, name="MitigatedIndex")
    percentage_series = pd.Series(percentage, name="Percentage")

    return pd.concat(
        [
            ob_series,
            top_series,
            bottom_series,
            obVolume_series,
            mitigated_index_series,
            percentage_series,
        ],
        axis=1,
    )

# Carregar dados do CSV
file_path = 'input.csv'  # Altere para o caminho correto do seu CSV
df = pd.read_csv(file_path, delimiter=';')
print(df)

# Selecionar apenas as colunas OHLCV principais para passar para o SMC
ohlc = df

# Garantir que os dados estejam indexados pela data
ohlc['date'] = pd.to_datetime(ohlc['date'])
ohlc.set_index('date', inplace=True)

# Aplicar indicadores SMC
# Fair Value Gap
ohlc_fvg = smc.fvg(ohlc, join_consecutive=False)
ohlc = pd.concat([ohlc, ohlc_fvg], axis=1)

# Swing Highs and Lows
ohlc_swing_hl = smc.swing_highs_lows(ohlc, swing_length=50)
ohlc = pd.concat([ohlc, ohlc_swing_hl], axis=1)

# Break of Structure (BOS) & Change of Character (CHoCH)
ohlc_bos_choch = smc.bos_choch(ohlc, ohlc_swing_hl, close_break=True)
ohlc = pd.concat([ohlc, ohlc_bos_choch], axis=1)

# Order Blocks (OB)
ohlc_ob = ob(ohlc=ohlc, swing_highs_lows=ohlc_swing_hl, close_mitigation=False)
ohlc = pd.concat([ohlc, ohlc_ob], axis=1)

# Liquidity
ohlc_liquidity = liquidity(ohlc=ohlc, swing_highs_lows=ohlc_swing_hl, range_percent=0.01)
ohlc = pd.concat([ohlc, ohlc_liquidity], axis=1)

# Salvar o DataFrame enriquecido em um novo CSV
output_file_path = 'output.csv'
ohlc.to_csv(output_file_path)

print(f"Dados enriquecidos salvos em {output_file_path}")
