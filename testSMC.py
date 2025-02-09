import pandas as pd
import plotly.graph_objects as go
import sys
import os
from binance.client import Client
from datetime import datetime
import numpy as np
import time
import imageio
from io import BytesIO
from PIL import Image

sys.path.append(os.path.abspath("../"))
from smartmoneyconcepts.smc import smc

def add_FVG(fig, df):
    for i in range(len(df["FVG"])):
        if not np.isnan(df["FVG"][i]):
            x1 = int(
                df["MitigatedIndex"][i]
                if df["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                # filled Rectangle
                type="rect",
                x0=df.index[i],
                y0=df["Top"][i],
                x1=df.index[x1],
                y1=df["Bottom"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="yellow",
                opacity=0.2,
            )
            mid_x = round((i + x1) / 2)
            mid_y = (df["Top"][i] + df["Bottom"][i]) / 2
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="FVG",
                    textposition="middle center",
                    textfont=dict(color='rgba(255, 255, 255, 0.4)', size=8),
                )
            )
    return fig


def add_swing_highs_lows(fig, df):
    indexs = []
    level = []
    for i in range(len(df)):
        if not np.isnan(df["HighLow"][i]):
            indexs.append(i)
            level.append(df["Level"][i])

    # plot these lines on a graph
    for i in range(len(indexs) - 1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[indexs[i]], df.index[indexs[i + 1]]],
                y=[level[i], level[i + 1]],
                mode="lines",
                line=dict(
                    color=(
                        "rgba(0, 128, 0, 0.2)"
                        if df["HighLow"][indexs[i]] == -1
                        else "rgba(255, 0, 0, 0.2)"
                    ),
                ),
            )
        )

    return fig


def add_bos_choch(fig, df):
    for i in range(len(df["BOS"])):
        if not np.isnan(df["BOS"][i]):
            # add a label to this line
            mid_x = round((i + int(df["BrokenIndex"][i])) / 2)
            mid_y = df["Level"][i]
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(df["BrokenIndex"][i])]],
                    y=[df["Level"][i], df["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="rgba(255, 165, 0, 0.2)",
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="BOS",
                    textposition="top center" if df["BOS"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(255, 165, 0, 0.4)", size=8),
                )
            )
        if not np.isnan(df["CHOCH"][i]):
            # add a label to this line
            mid_x = round((i + int(df["BrokenIndex"][i])) / 2)
            mid_y = df["Level"][i]
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(df["BrokenIndex"][i])]],
                    y=[df["Level"][i], df["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="rgba(0, 0, 255, 0.2)",
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="CHOCH",
                    textposition="top center" if df["CHOCH"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(0, 0, 255, 0.4)", size=8),
                )
            )

    return fig


def add_OB(fig, df):
    def format_volume(volume):
        if volume >= 1e12:
            return f"{volume / 1e12:.3f}T"
        elif volume >= 1e9:
            return f"{volume / 1e9:.3f}B"
        elif volume >= 1e6:
            return f"{volume / 1e6:.3f}M"
        elif volume >= 1e3:
            return f"{volume / 1e3:.3f}k"
        else:
            return f"{volume:.2f}"

    for i in range(len(df["OB"])):
        if df["OB"][i] == 1:
            x1 = int(
                df["MitigatedIndex"][i]
                if df["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=df["Bottom"][i],
                x1=df.index[x1],
                y1=df["Top"][i],
                line=dict(color="Purple"),
                fillcolor="Purple",
                opacity=0.2,
                name="Bullish OB",
                # Remove legendgroup property
                # legendgroup="bullish ob",
                # showlegend=True,
            )

            if df["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (df["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (df["Bottom"][i] + df["Top"][i]) / 2
            volume_text = format_volume(df["OBVolume"][i])
            # Add annotation text
            annotation_text = f'OB: {volume_text} ({df["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )

    for i in range(len(df["OB"])):
        if df["OB"][i] == -1:
            x1 = int(
                df["MitigatedIndex"][i]
                if df["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=df["Bottom"][i],
                x1=df.index[x1],
                y1=df["Top"][i],
                line=dict(color="Purple"),
                fillcolor="Purple",
                opacity=0.2,
                name="Bearish OB",
                # Remove legendgroup property
                # legendgroup="bearish ob",
                # showlegend=True,
            )

            if df["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (df["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (df["Bottom"][i] + df["Top"][i]) / 2
            volume_text = format_volume(df["OBVolume"][i])
            # Add annotation text
            annotation_text = f'OB: {volume_text} ({df["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )
    return fig


def add_liquidity(fig, df):
    # draw a line horizontally for each liquidity level
    for i in range(len(df["Liquidity"])):
        if not np.isnan(df["Liquidity"][i]):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(df["End"][i])]],
                    y=[df["Level"][i], df["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="rgba(255, 165, 0, 0.2)",
                    ),
                )
            )
            mid_x = round((i + int(df["End"][i])) / 2)
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[df["Level"][i]],
                    mode="text",
                    text="Liquidity",
                    textposition="top center" if df["Liquidity"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(255, 165, 0, 0.4)", size=8),
                )
            )
        if df["Swept"][i] != 0 and not np.isnan(df["Swept"][i]):
            # draw a red line between the end and the swept point
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.index[int(df["End"][i])],
                        df.index[int(df["Swept"][i])],
                    ],
                    y=[
                        df["Level"][i],
                        (
                            df["high"].iloc[int(df["Swept"][i])]
                            if df["Liquidity"][i] == 1
                            else df["low"].iloc[int(df["Swept"][i])]
                        ),
                    ],
                    mode="lines",
                    line=dict(
                        color="rgba(255, 0, 0, 0.2)",
                    ),
                )
            )
            mid_x = round((i + int(df["Swept"][i])) / 2)
            mid_y = (
                df["Level"][i]
                + (
                    df["high"].iloc[int(df["Swept"][i])]
                    if df["Liquidity"][i] == 1
                    else df["low"].iloc[int(df["Swept"][i])]
                )
            ) / 2
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="Liquidity Swept",
                    textposition="top center" if df["Liquidity"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(255, 0, 0, 0.4)", size=8),
                )
            )
    return fig


def add_previous_high_low(fig, df, previous_high_low_data):
    high = previous_high_low_data["PreviousHigh"]
    low = previous_high_low_data["PreviousLow"]

    # create a list of all the different high levels and their indexes
    high_levels = []
    high_indexes = []
    for i in range(len(high)):
        if not np.isnan(high[i]) and high[i] != (high_levels[-1] if len(high_levels) > 0 else None):
            high_levels.append(high[i])
            high_indexes.append(i)

    low_levels = [] 
    low_indexes = []
    for i in range(len(low)):
        if not np.isnan(low[i]) and low[i] != (low_levels[-1] if len(low_levels) > 0 else None):
            low_levels.append(low[i])
            low_indexes.append(i)

    # plot these lines on a graph
    for i in range(len(high_indexes)-1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[high_indexes[i]], df.index[high_indexes[i+1]]],
                y=[high_levels[i], high_levels[i]],
                mode="lines",
                line=dict(
                    color="rgba(255, 255, 255, 0.2)",
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index[high_indexes[i+1]]],
                y=[high_levels[i]],
                mode="text",
                text="PH",
                textposition="top center",
                textfont=dict(color="rgba(255, 255, 255, 0.4)", size=8),
            )
        )

    for i in range(len(low_indexes)-1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[low_indexes[i]], df.index[low_indexes[i+1]]],
                y=[low_levels[i], low_levels[i]],
                mode="lines",
                line=dict(
                    color="rgba(255, 255, 255, 0.2)",
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index[low_indexes[i+1]]],
                y=[low_levels[i]],
                mode="text",
                text="PL",
                textposition="bottom center",
                textfont=dict(color="rgba(255, 255, 255, 0.4)", size=8),
            )
        )

    return fig


def add_sessions(fig, df, sessions):
    for i in range(len(sessions["Active"])-1):
        if sessions["Active"][i] == 1:
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=sessions["Low"][i],
                x1=df.index[i + 1],
                y1=sessions["High"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="#16866E",
                opacity=0.2,
            )
    return fig


def add_retracements(fig, df, retracements):
    for i in range(len(retracements)):
        if (
            (
                (
                    retracements["Direction"].iloc[i + 1]
                    if i < len(retracements) - 1
                    else 0
                )
                != retracements["Direction"].iloc[i]
                or i == len(retracements) - 1
            )
            and retracements["Direction"].iloc[i] != 0
            and (
                retracements["Direction"].iloc[i + 1]
                if i < len(retracements) - 1
                else retracements["Direction"].iloc[i]
            )
            != 0
        ):
            fig.add_annotation(
                x=df.index[i],
                y=(
                    df["high"].iloc[i]
                    if retracements["Direction"].iloc[i] == -1
                    else df["low"].iloc[i]
                ),
                xref="x",
                yref="y",
                text=f"C:{retracements['CurrentRetracement%'].iloc[i]}%<br>D:{retracements['DeepestRetracement%'].iloc[i]}%",
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )
    return fig


# get the data
def import_data(symbol, start_str, timeframe):
    client = Client()
    start_str = str(start_str)
    end_str = f"{datetime.now()}"
    df = pd.DataFrame(
        client.get_historical_klines(
            symbol=symbol, interval=timeframe, start_str=start_str, end_str=end_str
        )
    ).astype(float)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    return df


#df = import_data("BTCUSDT", "2024-04-01", "15m")
#df = df.iloc[-500:]

file_path = 'input.csv'  # Altere para o caminho correto do seu CSV
df = pd.read_csv(file_path, delimiter=';')

def fig_to_buffer(fig):
    fig_bytes = fig.to_image(format="png")
    fig_buffer = BytesIO(fig_bytes)
    fig_image = Image.open(fig_buffer)
    return np.array(fig_image)


gif = []

window = 100
for pos in range(window, len(df)):
    window_df = df.iloc[pos - window : pos]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=window_df.index,
                open=window_df["open"],
                high=window_df["high"],
                low=window_df["low"],
                close=window_df["close"],
                increasing_line_color="#77dd76",
                decreasing_line_color="#ff6962",
            )
        ]
    )

    '''fvg_data = smc.fvg(window_df, join_consecutive=True)
    swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=5)
    bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
    ob_data = smc.ob(window_df, swing_highs_lows_data)
    liquidity_data = smc.liquidity(window_df, swing_highs_lows_data)'''

    fig = add_FVG(fig, window_df)
    fig = add_swing_highs_lows(fig, window_df)
    fig = add_bos_choch(fig, window_df)
    fig = add_OB(fig, window_df)
    fig = add_liquidity(fig, window_df)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(paper_bgcolor="rgba(12, 14, 18, 1)")
    fig.update_layout(font=dict(color="white"))

    # reduce the size of the image
    fig.update_layout(width=500, height=300)

    gif.append(fig_to_buffer(fig))

# save the gif
imageio.mimsave("test.gif", gif, duration=1)