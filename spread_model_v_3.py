# Load the NBA game data into a Pandas DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb

pd.set_option('display.max_columns', 14)

games_data = pd.read_csv('games_complete.csv')


def generate_game_features(games_data, home_team_id, visitor_team_id, game_date='2023-04-04', n_games=5):
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

    # print("Number of head to head games: ", len(head_to_head_games))
    # print("head to head games: ", head_to_head_games)

    if len(head_to_head_games) > 0:
        # Calculate the average stats for head-to-head games, considering the home and away teams
        h2h_home_stats = head_to_head_games[(head_to_head_games['HOME_TEAM_ID'] == home_team_id) | (
                head_to_head_games['VISITOR_TEAM_ID'] == home_team_id)].mean(numeric_only=True)
        h2h_visitor_stats = head_to_head_games[(head_to_head_games['VISITOR_TEAM_ID'] == visitor_team_id) | (
                head_to_head_games['HOME_TEAM_ID'] == visitor_team_id)].mean(numeric_only=True)

    home_team_stats = home_team_games[
        ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
         'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away']]
    away_team_stats = visitor_team_games[
        ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
         'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away']]

    future_game_features = np.concatenate((home_team_stats, away_team_stats), axis=0)

    return future_game_features  # a (10, 14) array

depth_of_games = 5
max_index = 1000
X_data_array = np.empty((max_index, depth_of_games*2, 14))

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

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

model = xgb.XGBRegressor()

param_grid = {
    'objective': ['reg:squarederror'],
    'max_depth': [3, 5],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [500, 2000, 4000],
    'n_jobs': [-1],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
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


print(Y_test.head(5))
print(y_pred[:5, :])
# print("-------------------------------------\n\n\n")

home_team_id = 1610612766  # Charlotte Hornets
# home_team_id = 1610612738  # Boston Celtics
visitor_team_id = 1610612761  # Toronto Raptors

# home_team_id = 1610612761  # Toronto Raptors
# visitor_team_id = 1610612766  # Charlotte Hornets

# Example: Predicting the spread, PTS_total, and OREB_total for a future game
future_game_features = generate_game_features(games_data, home_team_id, visitor_team_id, n_games=depth_of_games).reshape(1, -1)
# Scale the features using the provided scaler
future_game_features_scaled = scaler.transform(future_game_features)
future_game_prediction = model.predict(future_game_features_scaled)

print("Spread: ", future_game_prediction[0][0])
print("PTS_total: ", future_game_prediction[0][1])
print("OREB_total: ", future_game_prediction[0][2])
