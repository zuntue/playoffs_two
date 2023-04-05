import pandas as pd

games = pd.read_csv("games.csv")
games_details = pd.read_csv("games_details.csv")

games_2022 = games[games['SEASON'] == 2022]

games_complete = games_2022.copy()
games_complete['OREB_home'] = pd.Series(dtype='int64')
games_complete['OREB_away'] = pd.Series(dtype='int64')
games_complete['spread'] = pd.Series(dtype='float64')


def calculate_total_team_OREB_and_spread(_game_id, _home_team_id, _away_team_id):
    game_of_interest = games_details.loc[(games_details['GAME_ID'] == _game_id)]
    home_total = game_of_interest.loc[game_of_interest['TEAM_ID'] == _home_team_id]['OREB'].sum()
    away_total = game_of_interest.loc[game_of_interest['TEAM_ID'] == _away_team_id]['OREB'].sum()
    game_row = games_2022.loc[games_2022['GAME_ID'] == _game_id]
    spread = game_row['PTS_home'].item() - game_row['PTS_away'].item()
    return int(home_total), int(away_total), spread

for index, row in games_2022.iterrows():
    game_id = row['GAME_ID']
    home_id = row['HOME_TEAM_ID']
    away_id = row['VISITOR_TEAM_ID']
    oreb_home, oreb_away, spread = calculate_total_team_OREB_and_spread(game_id, home_id, away_id)
    games_complete.loc[index, 'OREB_home'] = oreb_home
    games_complete.loc[index, 'OREB_away'] = oreb_away
    games_complete.loc[index, 'spread'] = spread

    if index % 20 == 0:
        print(index)

games_complete['PTS_total'] = games_complete['PTS_home'] + games_complete['PTS_away']
games_complete['OREB_total'] = games_complete['OREB_home'] + games_complete['PTS_away']


games_complete.to_csv('games_complete.csv', index=False)
