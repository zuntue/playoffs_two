from time import sleep

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from tqdm import tqdm


def custom_scorer(y_true, y_pred):
    pbar.update(1)
    return -mean_squared_error(y_true, y_pred)  # Return the negative value since GridSearchCV maximizes the score


def get_stats_from_games(games, team_id):
    stats = np.empty((len(games), 7))
    index_to_use = 0

    for index, row in games.iterrows():
        stats_row = []
        for stat in ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
                     'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away']:
            if (row['HOME_TEAM_ID'] == team_id) and stat.endswith('_home'):
                stats_row.append(row[stat])
            elif (row['VISITOR_TEAM_ID'] == team_id) and stat.endswith('_away'):
                stats_row.append(row[stat])

        stats[index_to_use] = stats_row
        index_to_use += 1

    return stats


def generate_game_features(games_data, home_team_id, visitor_team_id, game_date='2023-04-04', n_games=5):
    # Filter games_data to include only the last n games for each team
    home_team_games = games_data[
        ((games_data['HOME_TEAM_ID'] == home_team_id) | (games_data['VISITOR_TEAM_ID'] == home_team_id)) & (
                games_data['GAME_DATE_EST'] < game_date)].head(n_games)
    visitor_team_games = games_data[
        ((games_data['HOME_TEAM_ID'] == visitor_team_id) | (games_data['VISITOR_TEAM_ID'] == visitor_team_id)) & (
                games_data['GAME_DATE_EST'] < game_date)].head(n_games)

    home_team_stats = get_stats_from_games(home_team_games, home_team_id)
    visitor_team_stats = get_stats_from_games(visitor_team_games, visitor_team_id)

    return np.concatenate(
        (np.nanmean(home_team_stats, axis=0), np.nanmean(visitor_team_stats, axis=0)))  # a (2, 7) array


games_data = pd.read_csv('games_complete.csv')

depth_of_games = 60
max_index = 1000
X_data_array = np.empty((max_index, 14))

for index, row in games_data.iterrows():
    if index < max_index:
        X_data_array[index] = generate_game_features(games_data, row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID'],
                                                     row['GAME_DATE_EST'], depth_of_games)

X_data = pd.DataFrame(X_data_array.reshape(max_index, -1))

X = X_data
Y = games_data[['spread', 'PTS_total', 'OREB_total']].head(max_index)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
sleep(0.1)

wrapped_scorer = make_scorer(custom_scorer, greater_is_better=True)
model = xgb.XGBRegressor()

param_grid = {
    'objective': ['reg:squarederror'],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.005, 0.01, 0.02],
    'n_estimators': [200, 300, 500],
    'n_jobs': [-1],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=wrapped_scorer)

total_iterations = len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(
    param_grid['n_estimators']) * 5  # times 5 because cv = 5

with tqdm(total=total_iterations) as pbar:
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

print("-------------------------------------\n\n\n")

print(scaler.inverse_transform(X_test[:10, :]))
print(y_scaler.inverse_transform(Y_test[:10, :]))
print(y_scaler.inverse_transform(y_pred[:10, :]))

# Example: Predicting the spread, PTS_total, and OREB_total for a future game
home_team_id = 1610612766  # Charlotte Hornets
visitor_team_id = 1610612761  # Toronto Raptors

future_game_features = generate_game_features(games_data, home_team_id, visitor_team_id,
                                              n_games=depth_of_games).reshape(1, -1)

future_game_features_scaled = scaler.transform(future_game_features)
future_game_prediction = y_scaler.inverse_transform(model.predict(future_game_features_scaled))

print("Spread: ", future_game_prediction[0][0])
print("PTS_total: ", future_game_prediction[0][1])
print("OREB_total: ", future_game_prediction[0][2])
