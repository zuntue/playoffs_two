import csv

import pandas as pd

games = pd.read_csv("./scraped_data/games.csv")
games_details = pd.read_csv("./scraped_data/games_details.csv")

games_2022 = games[games['SEASON'] == 2022]
games_2022 = games_2022[games_2022['GAME_DATE_EST'] >= '2023-04-04']
games_2022 = games_2022[games_2022['GAME_DATE_EST'] <= '2023-04-09']


games_results = games_2022.copy()
games_results['OREB_home'] = pd.Series(dtype='int64')
games_results['OREB_away'] = pd.Series(dtype='int64')
games_results['spread'] = pd.Series(dtype='float64')


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
    games_results.loc[index, 'OREB_home'] = oreb_home
    games_results.loc[index, 'OREB_away'] = oreb_away
    games_results.loc[index, 'spread'] = spread


games_results['PTS_total'] = games_results['PTS_home'] + games_results['PTS_away']
games_results['OREB_total'] = games_results['OREB_home'] + games_results['OREB_away']

# keep only certain columns
columns_to_keep = ['GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'spread', 'PTS_total', 'OREB_total']
games_results = games_results.drop(columns=[col for col in games_results.columns if col not in columns_to_keep])

games_results.to_csv('games_results.csv', index=False)

future_games = [{'Home': 'Charlotte Hornets', 'Away': 'Toronto Raptors'},
                {'Home': 'Washington Wizards', 'Away': 'Milwaukee Bucks'},
                {'Home': 'Orlando Magic', 'Away': 'Cleveland Cavaliers'},
                {'Home': 'Detroit Pistons', 'Away': 'Miami Heat'},
                {'Home': 'Brooklyn Nets', 'Away': 'Minnesota Timberwolves'},
                {'Home': 'Houston Rockets', 'Away': 'Denver Nuggets'},
                {'Home': 'Memphis Grizzlies', 'Away': 'Portland Trail Blazers'},
                {'Home': 'New Orleans Pelicans', 'Away': 'Sacramento Kings'},
                {'Home': 'Philadelphia 76ers', 'Away': 'Boston Celtics'},
                {'Home': 'Chicago Bulls', 'Away': 'Atlanta Hawks'},
                {'Home': 'Utah Jazz', 'Away': 'Los Angeles Lakers'},
                {'Home': 'Golden State Warriors', 'Away': 'Oklahoma City Thunder'},
                {'Home': 'Phoenix Suns', 'Away': 'San Antonio Spurs'},
                {'Home': 'Detroit Pistons', 'Away': 'Brooklyn Nets'},
                {'Home': 'Indiana Pacers', 'Away': 'New York Knicks'},
                {'Home': 'Atlanta Hawks', 'Away': 'Washington Wizards'},
                {'Home': 'Boston Celtics', 'Away': 'Toronto Raptors'},
                {'Home': 'Milwaukee Bucks', 'Away': 'Chicago Bulls'},
                {'Home': 'New Orleans Pelicans', 'Away': 'Memphis Grizzlies'},
                {'Home': 'Dallas Mavericks', 'Away': 'Sacramento Kings'},
                {'Home': 'Los Angeles Clippers', 'Away': 'Los Angeles Lakers'},
                {'Home': 'Orlando Magic', 'Away': 'Cleveland Cavaliers'},
                {'Home': 'Philadelphia 76ers', 'Away': 'Miami Heat'},
                {'Home': 'San Antonio Spurs', 'Away': 'Portland Trail Blazers'},
                {'Home': 'Utah Jazz', 'Away': 'Oklahoma City Thunder'},
                {'Home': 'Phoenix Suns', 'Away': 'Denver Nuggets'},
                {'Home': 'Charlotte Hornets', 'Away': 'Houston Rockets'},
                {'Home': 'Washington Wizards', 'Away': 'Miami Heat'},
                {'Home': 'Indiana Pacers', 'Away': 'Detroit Pistons'},
                {'Home': 'Brooklyn Nets', 'Away': 'Orlando Magic'},
                {'Home': 'Boston Celtics', 'Away': 'Toronto Raptors'},
                {'Home': 'Atlanta Hawks', 'Away': 'Philadelphia 76ers'},
                {'Home': 'Milwaukee Bucks', 'Away': 'Memphis Grizzlies'},
                {'Home': 'New Orleans Pelicans', 'Away': 'New York Knicks'},
                {'Home': 'Dallas Mavericks', 'Away': 'Chicago Bulls'},
                {'Home': 'Sacramento Kings', 'Away': 'Golden State Warriors'},
                {'Home': 'Los Angeles Lakers', 'Away': 'Phoenix Suns'},
                {'Home': 'Utah Jazz', 'Away': 'Denver Nuggets'},
                {'Home': 'Los Angeles Clippers', 'Away': 'Portland Trail Blazers'},
                {'Home': 'San Antonio Spurs', 'Away': 'Minnesota Timberwolves'},
                {'Home': 'Boston Celtics', 'Away': 'Atlanta Hawks'},
                {'Home': 'Brooklyn Nets', 'Away': 'Philadelphia 76ers'},
                {'Home': 'Chicago Bulls', 'Away': 'Detroit Pistons'},
                {'Home': 'Cleveland Cavaliers', 'Away': 'Charlotte Hornets'},
                {'Home': 'Miami Heat', 'Away': 'Orlando Magic'},
                {'Home': 'New York Knicks', 'Away': 'Indiana Pacers'},
                {'Home': 'Toronto Raptors', 'Away': 'Milwaukee Bucks'},
                {'Home': 'Washington Wizards', 'Away': 'Houston Rockets'},
                {'Home': 'Dallas Mavericks', 'Away': 'San Antonio Spurs'},
                {'Home': 'Denver Nuggets', 'Away': 'Sacramento Kings'},
                {'Home': 'Los Angeles Lakers', 'Away': 'Utah Jazz'},
                {'Home': 'Minnesota Timberwolves', 'Away': 'New Orleans Pelicans'},
                {'Home': 'Oklahoma City Thunder', 'Away': 'Memphis Grizzlies'},
                {'Home': 'Phoenix Suns', 'Away': 'Los Angeles Clippers'},
                {'Home': 'Portland Trail Blazers', 'Away': 'Golden State Warriors'}
                ]

team_name_to_id = {
    'Atlanta Hawks': 1610612737,
    'Boston Celtics': 1610612738,
    'Brooklyn Nets': 1610612751,
    'Charlotte Hornets': 1610612766,
    'Chicago Bulls': 1610612741,
    'Cleveland Cavaliers': 1610612739,
    'Dallas Mavericks': 1610612742,
    'Denver Nuggets': 1610612743,
    'Detroit Pistons': 1610612765,
    'Golden State Warriors': 1610612744,
    'Houston Rockets': 1610612745,
    'Indiana Pacers': 1610612754,
    'Los Angeles Clippers': 1610612746,
    'Los Angeles Lakers': 1610612747,
    'Memphis Grizzlies': 1610612763,
    'Miami Heat': 1610612748,
    'Milwaukee Bucks': 1610612749,
    'Minnesota Timberwolves': 1610612750,
    'New Orleans Pelicans': 1610612740,
    'New York Knicks': 1610612752,
    'Oklahoma City Thunder': 1610612760,
    'Orlando Magic': 1610612753,
    'Philadelphia 76ers': 1610612755,
    'Phoenix Suns': 1610612756,
    'Portland Trail Blazers': 1610612757,
    'Sacramento Kings': 1610612758,
    'San Antonio Spurs': 1610612759,
    'Toronto Raptors': 1610612761,
    'Utah Jazz': 1610612762,
    'Washington Wizards': 1610612764
}

# Predictions for the future games
results = []

for game in future_games:
    home_team_id = team_name_to_id[game['Home']]
    away_team_id = team_name_to_id[game['Away']]

    matching_rows = games_results[(games_results['HOME_TEAM_ID'] == home_team_id) & (games_results['VISITOR_TEAM_ID'] == away_team_id)]

    results.append({
        'Away': game['Away'],
        'Home': game['Home'],
        'Spread': matching_rows['spread'].iloc[0] if not matching_rows.empty else 0,
        'Total': matching_rows['PTS_total'].iloc[0] if not matching_rows.empty else 0,
        'OREB': matching_rows['OREB_total'].iloc[0] if not matching_rows.empty else 0
    })

for result in results:
    print(f"{result['Home']} vs {result['Away']}:")
    print(f"Spread: {result['Spread']}")
    print(f"PTS_total: {result['Total']}")
    print(f"OREB_total: {result['OREB']}")
    print("-------------------------------------")


# Save predictions to CSV
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Away', 'Home', 'Spread', 'Total', 'OREB']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)

