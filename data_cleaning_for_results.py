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

# Sort the DataFrame by the 'GAME_DATE_EST' and 'index' columns in ascending order
games_results_sorted = games_results.sort_values(by=['GAME_DATE_EST'])


future_games = [
    {'Date': '2023-04-04', 'Away': 'Toronto Raptors', 'Home': 'Charlotte Hornets'},
    {'Date': '2023-04-04', 'Away': 'Milwaukee Bucks', 'Home': 'Washington Wizards'},
    {'Date': '2023-04-04', 'Away': 'Cleveland Cavaliers', 'Home': 'Orlando Magic'},
    {'Date': '2023-04-04', 'Away': 'Miami Heat', 'Home': 'Detroit Pistons'},
    {'Date': '2023-04-04', 'Away': 'Minnesota Timberwolves', 'Home': 'Brooklyn Nets'},
    {'Date': '2023-04-04', 'Away': 'Denver Nuggets', 'Home': 'Houston Rockets'},
    {'Date': '2023-04-04', 'Away': 'Portland Trail Blazers', 'Home': 'Memphis Grizzlies'},
    {'Date': '2023-04-04', 'Away': 'Sacramento Kings', 'Home': 'New Orleans Pelicans'},
    {'Date': '2023-04-04', 'Away': 'Boston Celtics', 'Home': 'Philadelphia 76ers'},
    {'Date': '2023-04-04', 'Away': 'Atlanta Hawks', 'Home': 'Chicago Bulls'},
    {'Date': '2023-04-04', 'Away': 'Los Angeles Lakers', 'Home': 'Utah Jazz'},
    {'Date': '2023-04-04', 'Away': 'Oklahoma City Thunder', 'Home': 'Golden State Warriors'},
    {'Date': '2023-04-04', 'Away': 'San Antonio Spurs', 'Home': 'Phoenix Suns'},
    {'Date': '2023-04-05', 'Away': 'Brooklyn Nets', 'Home': 'Detroit Pistons'},
    {'Date': '2023-04-05', 'Away': 'New York Knicks', 'Home': 'Indiana Pacers'},
    {'Date': '2023-04-05', 'Away': 'Washington Wizards', 'Home': 'Atlanta Hawks'},
    {'Date': '2023-04-05', 'Away': 'Toronto Raptors', 'Home': 'Boston Celtics'},
    {'Date': '2023-04-05', 'Away': 'Chicago Bulls', 'Home': 'Milwaukee Bucks'},
    {'Date': '2023-04-05', 'Away': 'Memphis Grizzlies', 'Home': 'New Orleans Pelicans'},
    {'Date': '2023-04-05', 'Away': 'Sacramento Kings', 'Home': 'Dallas Mavericks'},
    {'Date': '2023-04-05', 'Away': 'Los Angeles Lakers', 'Home': 'Los Angeles Clippers'},
    {'Date': '2023-04-06', 'Away': 'Cleveland Cavaliers', 'Home': 'Orlando Magic'},
    {'Date': '2023-04-06', 'Away': 'Miami Heat', 'Home': 'Philadelphia 76ers'},
    {'Date': '2023-04-06', 'Away': 'Portland Trail Blazers', 'Home': 'San Antonio Spurs'},
    {'Date': '2023-04-06', 'Away': 'Oklahoma City Thunder', 'Home': 'Utah Jazz'},
    {'Date': '2023-04-06', 'Away': 'Denver Nuggets', 'Home': 'Phoenix Suns'},
    {'Date': '2023-04-07', 'Away': 'Houston Rockets', 'Home': 'Charlotte Hornets'},
    {'Date': '2023-04-07', 'Away': 'Miami Heat', 'Home': 'Washington Wizards'},
    {'Date': '2023-04-07', 'Away': 'Detroit Pistons', 'Home': 'Indiana Pacers'},
    {'Date': '2023-04-07', 'Away': 'Orlando Magic', 'Home': 'Brooklyn Nets'},
    {'Date': '2023-04-07', 'Away': 'Toronto Raptors', 'Home': 'Boston Celtics'},
    {'Date': '2023-04-07', 'Away': 'Philadelphia 76ers', 'Home': 'Atlanta Hawks'},
    {'Date': '2023-04-07', 'Away': 'Memphis Grizzlies', 'Home': 'Milwaukee Bucks'},
    {'Date': '2023-04-07', 'Away': 'New York Knicks', 'Home': 'New Orleans Pelicans'},
    {'Date': '2023-04-07', 'Away': 'Chicago Bulls', 'Home': 'Dallas Mavericks'},
    {'Date': '2023-04-07', 'Away': 'Golden State Warriors', 'Home': 'Sacramento Kings'},
    {'Date': '2023-04-07', 'Away': 'Phoenix Suns', 'Home': 'Los Angeles Lakers'},
    {'Date': '2023-04-08', 'Away': 'Denver Nuggets', 'Home': 'Utah Jazz'},
    {'Date': '2023-04-08', 'Away': 'Portland Trail Blazers', 'Home': 'Los Angeles Clippers'},
    {'Date': '2023-04-08', 'Away': 'Minnesota Timberwolves', 'Home': 'San Antonio Spurs'},
    {'Date': '2023-04-09', 'Away': 'Atlanta Hawks', 'Home': 'Boston Celtics'},
    {'Date': '2023-04-09', 'Away': 'Philadelphia 76ers', 'Home': 'Brooklyn Nets'},
    {'Date': '2023-04-09', 'Away': 'Detroit Pistons', 'Home': 'Chicago Bulls'},
    {'Date': '2023-04-09', 'Away': 'Charlotte Hornets', 'Home': 'Cleveland Cavaliers'},
    {'Date': '2023-04-09', 'Away': 'Orlando Magic', 'Home': 'Miami Heat'},
    {'Date': '2023-04-09', 'Away': 'Indiana Pacers', 'Home': 'New York Knicks'},
    {'Date': '2023-04-09', 'Away': 'Milwaukee Bucks', 'Home': 'Toronto Raptors'},
    {'Date': '2023-04-09', 'Away': 'Houston Rockets', 'Home': 'Washington Wizards'},
    {'Date': '2023-04-09', 'Away': 'San Antonio Spurs', 'Home': 'Dallas Mavericks'},
    {'Date': '2023-04-09', 'Away': 'Sacramento Kings', 'Home': 'Denver Nuggets'},
    {'Date': '2023-04-09', 'Away': 'Utah Jazz', 'Home': 'Los Angeles Lakers'},
    {'Date': '2023-04-09', 'Away': 'New Orleans Pelicans', 'Home': 'Minnesota Timberwolves'},
    {'Date': '2023-04-09', 'Away': 'Memphis Grizzlies', 'Home': 'Oklahoma City Thunder'},
    {'Date': '2023-04-09', 'Away': 'Los Angeles Clippers', 'Home': 'Phoenix Suns'},
    {'Date': '2023-04-09', 'Away': 'Golden State Warriors', 'Home': 'Portland Trail Blazers'},
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

# Create a new DataFrame with the same columns as games_results_sorted plus the 'Home', 'Away', and 'Date' columns
future_games_df = pd.DataFrame(columns=games_results_sorted.columns.tolist() + ['Home', 'Away', 'Date'])

# Iterate through the future_games list to populate the future_games_df DataFrame
for game in future_games:
    home_team_id = team_name_to_id[game['Home']]
    away_team_id = team_name_to_id[game['Away']]

    # Find the matching row in games_results_sorted based on the team IDs and the game date
    matching_row = games_results_sorted.loc[(games_results_sorted['HOME_TEAM_ID'] == home_team_id) &
                                            (games_results_sorted['VISITOR_TEAM_ID'] == away_team_id) &
                                            (games_results_sorted['GAME_DATE_EST'] == game['Date'])].copy()

    if not matching_row.empty:
        # Set the Home, Away, and Date columns
        matching_row['Home'] = game['Home']
        matching_row['Away'] = game['Away']
        matching_row['Date'] = game['Date']

        # Concatenate the matching row to future_games_df
        future_games_df = pd.concat([future_games_df, matching_row], ignore_index=True)

results = []

# Iterate through the future_games_df DataFrame to create the results list
for index, row in future_games_df.iterrows():
    results.append({
        'Away': row['Away'],
        'Home': row['Home'],
        'Spread': row['spread'],
        'Total': row['PTS_total'],
        'OREB': row['OREB_total']
    })

for result in results:
    print(f"{result['Home']} vs {result['Away']}:")
    print(f"Spread: {result['Spread']}")
    print(f"PTS_total: {result['Total']}")
    print(f"OREB_total: {result['OREB']}")
    print("-------------------------------------")


# Save results to CSV
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Away', 'Home', 'Spread', 'Total', 'OREB']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)

