import pandas as pd
import csv

from model_v_4 import generate_game_features, games_data, depth_of_games, scaler, y_scaler, model

# Load the team ID dataset into a Pandas DataFrame
team_id_data = pd.read_csv('teams.csv', header=None)
team_id_data.columns = ['league_id', 'team_id', 'min_year', 'max_year', 'abbreviation', 'team_name', 'year_founded',
                        'city', 'arena', 'arena_capacity', 'owner', 'general_manager', 'head_coach',
                        'd_league_affiliation']

# Create a dictionary mapping team names to IDs
team_id_dict = dict(zip(team_id_data['team_name'], team_id_data['team_id']))

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
predictions = []

for game in future_games:
    home_team_id = team_name_to_id[game['Home']]
    away_team_id = team_name_to_id[game['Away']]
    future_game_features = generate_game_features(games_data, home_team_id, away_team_id,
                                                  n_games=depth_of_games).reshape(1, -1)
    future_game_features_scaled = scaler.transform(future_game_features)
    future_game_prediction = y_scaler.inverse_transform(model.predict(future_game_features_scaled))
    predictions.append({
        'Home': game['Home'],
        'Away': game['Away'],
        'Spread': future_game_prediction[0][0],
        'PTS_total': future_game_prediction[0][1],
        'OREB_total': future_game_prediction[0][2]
    })

for prediction in predictions:
    print(f"{prediction['Home']} vs {prediction['Away']}:")
    print(f"Spread: {prediction['Spread']}")
    print(f"PTS_total: {prediction['PTS_total']}")
    print(f"OREB_total: {prediction['OREB_total']}")
    print("-------------------------------------")


# Save predictions to CSV
with open('predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['Home', 'Away', 'Spread', 'PTS_total', 'OREB_total']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for prediction in predictions:
        writer.writerow(prediction)
