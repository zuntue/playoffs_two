# Load the NBA game data into a Pandas DataFrame
from time import sleep

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from joblib import parallel_backend
from sklearn.metrics import make_scorer, mean_squared_error


def custom_scorer(y_true, y_pred, **kwargs):
    mse = mean_squared_error(y_true, y_pred)
    pbar.update(1)
    return -mse  # Return the negative value since GridSearchCV maximizes the score


wrapped_scorer = make_scorer(custom_scorer, greater_is_better=True)

pd.set_option('display.max_columns', 30)

games_data = pd.read_csv('games_complete.csv')


def generate_game_features(games_data, home_team_id, visitor_team_id, game_date='2023-04-04', n_games=5,
                           weight_n_games=0.7,
                           weight_h2h=0.3):
    # Filter games_data to include only the last n games for each team
    home_team_games = games_data[
        ((games_data['HOME_TEAM_ID'] == home_team_id) | (games_data['VISITOR_TEAM_ID'] == home_team_id)) & (
                games_data['GAME_DATE_EST'] < game_date)].head(n_games)
    visitor_team_games = games_data[
        ((games_data['HOME_TEAM_ID'] == visitor_team_id) | (games_data['VISITOR_TEAM_ID'] == visitor_team_id)) & (
                games_data['GAME_DATE_EST'] < game_date)].head(
        n_games)


    # Filter games_data to include only head-to-head games from this season
    head_to_head_games = games_data[
        (((games_data['HOME_TEAM_ID'] == home_team_id) & (games_data['VISITOR_TEAM_ID'] == visitor_team_id)) | (
                (games_data['HOME_TEAM_ID'] == visitor_team_id) & (
                games_data['VISITOR_TEAM_ID'] == home_team_id))) & (games_data['GAME_DATE_EST'] < game_date)]

    def get_stats_from_games(games, team_id):
        stats = np.empty((len(games), 7))
        index_to_use = 0
        for index, row in games.iterrows():
            stats_row = []
            for stat in ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
                         'OREB_home',
                         'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
                         'OREB_away']:
                if (row['HOME_TEAM_ID'] == team_id) and stat.endswith('_home'):
                    stats_row.append(row[stat])
                elif (row['VISITOR_TEAM_ID'] == team_id) and stat.endswith('_away'):
                    stats_row.append(row[stat])

                # Assign the values from stats_row to the corresponding row in the stats array
            stats[index_to_use] = stats_row
            index_to_use += 1
        return stats

    # ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
    #  'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away']
    #
    home_team_stats = get_stats_from_games(home_team_games, home_team_id)
    visitor_team_stats = get_stats_from_games(visitor_team_games, visitor_team_id)

    #
    # # This is wrong because it can't tell whether the team we are interested in was home or away
    # # Calculate the average stats for head-to-head games, considering the home and away teams
    #
    # # Calculate the weighted features
    # weighted_features = []
    #
    # for stat in ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
    #              'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away']:
    #     if stat.endswith('_home'):
    #         weighted_stat = weight_n_games * home_team_stats[stat] + weight_h2h * h2h_home_stats[stat]
    #     else:
    #         weighted_stat = weight_n_games * visitor_team_stats[stat] + weight_h2h * h2h_visitor_stats[stat]
    #     weighted_features.append(weighted_stat)
    #
    # future_game_features = np.array(weighted_features)

    # print("np.concatenate((home_team_stats.mean(axis=0), visitor_team_stats.mean(axis=0)), axis=0): ", np.concatenate((home_team_stats.mean(axis=0), visitor_team_stats.mean(axis=0)), axis=0))
    return np.concatenate((np.nanmean(home_team_stats, axis=0), np.nanmean(visitor_team_stats, axis=0))) # a (2, 7) array


depth_of_games = 60
max_index = 1000
X_data_array = np.empty((max_index, 14))

for index, row in games_data.iterrows():
    if index < 1000:
        # print("generate_game_features(str(row['HOME_TEAM_ID']), str(row['VISITOR_TEAM_ID'])): ",  generate_game_features(games_data, row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']))
        X_data_array[index] = generate_game_features(games_data, row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID'],
                                                     row['GAME_DATE_EST'], depth_of_games)

# Create a DataFrame from the NumPy array
X_data = pd.DataFrame(X_data_array.reshape(max_index, -1))

# Define the input variables (X) and target variable (Y)
# print("X_data: ", X_data)
# print("-------------------------------------\n\n\n")

X = X_data  # two dimensional
Y = games_data[['spread', 'PTS_total', 'OREB_total']].head(1000)

# Normalize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

sleep(0.1)  # so that print finishes before status bar is printed

model = xgb.XGBRegressor()

param_grid = {
    'objective': ['reg:squarederror'],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.005, 0.01, 0.02],
    'n_estimators': [200, 300, 400, 500],
    'n_jobs': [-1],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=wrapped_scorer)

# Calculate the total number of iterations
total_iterations = len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(
    param_grid['n_estimators']) * 5  # times 5 because cv = 5

# Create a progress bar
with tqdm(total=total_iterations) as pbar:
    # Use joblib to run the grid search in parallel
    # Fit the grid search object to the data
    grid.fit(X_train, Y_train)

best_params = grid.best_params_
print("Best parameters found: ", best_params)

model = xgb.XGBRegressor(**best_params)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("Mean squared error: ", mse)
score = model.score(X_test, Y_test)
print("Model score: ", score)
# print("input importance: ", model.feature_importances_)
print("-------------------------------------\n\n\n")

print(scaler.inverse_transform(X_test[:10, :]))
print(y_scaler.inverse_transform(Y_test[:10, :]))
print(y_scaler.inverse_transform(y_pred[:10, :]))
# print("-------------------------------------\n\n\n")

home_team_id = 1610612766  # Charlotte Hornets
# home_team_id = 1610612738  # Boston Celtics
visitor_team_id = 1610612761  # Toronto Raptors

# Example: Predicting the spread, PTS_total, and OREB_total for a future game
future_game_features = generate_game_features(games_data, home_team_id, visitor_team_id,
                                              n_games=depth_of_games).reshape(1, -1)
# Scale the features using the provided scaler
future_game_features_scaled = scaler.transform(future_game_features)
future_game_prediction = y_scaler.inverse_transform(model.predict(future_game_features_scaled))

print("Spread: ", future_game_prediction[0][0])
print("PTS_total: ", future_game_prediction[0][1])
print("OREB_total: ", future_game_prediction[0][2])
