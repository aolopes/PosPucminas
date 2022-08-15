# Uteis:
import time
import datetime

# Pandas:
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)


def reader_lol(
    df: pd.DataFrame,
    version: int = 0,
    verbose_data: bool = True
) -> pd.DataFrame:
    df = df.loc[df['position'] != 'team']

    df_total = pd.DataFrame()

    start = time.time()

    positions = ['top', 'jng', 'mid', 'bot', 'sup']

    gameid_list = df['gameid'].unique()

    # raise Exception("NotImplemented")
    for index_game, gameid in enumerate(gameid_list):

        df_game = df.loc[df['gameid'] == gameid]

        column_names = [
            'game_id',
            'blue_top',
            'blue_jng',
            'blue_mid',
            'blue_bot',
            'blue_sup',
            'red_top',
            'red_jng',
            'red_mid',
            'red_bot',
            'red_sup',
        ]
        if version == 1:
            column_names += [
                'player_blue_top',
                'player_blue_jng',
                'player_blue_mid',
                'player_blue_bot',
                'player_blue_sup',
                'player_red_top',
                'player_red_jng',
                'player_red_mid',
                'player_red_bot',
                'player_red_sup',
            ]

        column_names.append('win_blue')

        i, j = 0, 0
        df2_data = {}

        for column in column_names:
            df2_data[column] = []

        for index, row in df_game.iterrows():

            if(len(df2_data['game_id']) == 0):
                df2_data['game_id'].append(row['gameid'])

            offset = 1
            side = 'blue'
            if(row['side'] == 'Blue'):
                if(len(df2_data['win_blue']) == 0):
                    if row['result'] == 0:
                        df2_data['win_blue'].append(0)
                    else:
                        df2_data['win_blue'].append(1)

            else:
                offset = 6
                side = 'red'

            i = row['participantid']

            n = side + '_' + positions[i-offset]
            df2_data[n].append(str(row['champion']))
            if version == 1:
                df2_data[f"player_{n}"].append(str(row['playerid']))
        if verbose_data:
            print(df2_data)

        df2 = pd.DataFrame(df2_data)

        df_total = pd.concat([df_total, df2])

        current = time.time()
        decorrido = current - start
        tempo_total = decorrido * df.shape[0] / (index_game + 1)

        if verbose_data:
            print(
                "Tempo decorrido: ", str(datetime.timedelta(seconds=decorrido)) +
                " / " + str(datetime.timedelta(seconds=tempo_total))
            )

    df_total.reset_index(inplace=True)

    # if verbose_data:
    #     print("\n\nResultado: ")
    #     print(df_total)

    return df_total


def transform_columns_lol(
    df_total: pd.DataFrame,
    version: int = 0,
    verbose_data: bool = True
) -> pd.DataFrame:
    df_total.drop(['index', 'game_id'], axis=1, inplace=True)
    return df_total
