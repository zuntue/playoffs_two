import pandas as pd

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
                {'Home': 'Chicago Bulls', 'Away': 'Detroit Pistons'}
                ]

# Load the original DataFrame
games_data = pd.read_csv('games_data.csv')

# Map the team IDs to the Home and Away columns
games_data['Home_ID'] = games_data['Home'].map(team_id_dict)
games_data['Away_ID'] = games_data['Away'].map(team_id_dict)

# Print the updated DataFrame
print(games_data.head())
