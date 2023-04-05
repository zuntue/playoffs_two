# Load the NBA game data into a Pandas DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb

pd.set_option('display.max_columns', 14)

games_data = pd.read_csv('games_complete.csv')


def generate_game_features(games_data, home_team_id, visitor_team_id, n=2, weight_n_games=0.7,
                           weight_h2h=0.3):

    # Filter games_data to include only the last n games for each team
    home_team_games = games_data[
        (games_data['HOME_TEAM_ID'] == home_team_id) | (games_data['VISITOR_TEAM_ID'] == home_team_id)].head(
        n)  # need to use .head() due to the most recent games being at the top of the data
    visitor_team_games = games_data[
        (games_data['HOME_TEAM_ID'] == visitor_team_id) | (games_data['VISITOR_TEAM_ID'] == visitor_team_id)].head(
        n)  # need to use .head() due to the most recent games being at the top of the data

    # Calculate the average stats for each team
    home_team_stats = home_team_games.mean(numeric_only=True)
    visitor_team_stats = visitor_team_games.mean(numeric_only=True)

    # Filter games_data to include only head-to-head games from this season
    head_to_head_games = games_data[
        ((games_data['HOME_TEAM_ID'] == home_team_id) & (games_data['VISITOR_TEAM_ID'] == visitor_team_id)) | (
                (games_data['HOME_TEAM_ID'] == visitor_team_id) & (
                games_data['VISITOR_TEAM_ID'] == home_team_id))]

    print("Number of head to head games: ", len(head_to_head_games))
    print("head to head games: ", head_to_head_games)

    # Calculate the average stats for head-to-head games, considering the home and away teams
    h2h_home_stats = head_to_head_games[(head_to_head_games['HOME_TEAM_ID'] == home_team_id) | (
            head_to_head_games['VISITOR_TEAM_ID'] == home_team_id)].mean(numeric_only=True)
    h2h_visitor_stats = head_to_head_games[(head_to_head_games['VISITOR_TEAM_ID'] == visitor_team_id) | (
            head_to_head_games['HOME_TEAM_ID'] == visitor_team_id)].mean(numeric_only=True)
    # Calculate the weighted features
    weighted_features = []
    for stat in ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
                 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away']:
        if stat.endswith('_home'):
            weighted_stat = weight_n_games * home_team_stats[stat] + weight_h2h * h2h_home_stats[stat]
        else:
            weighted_stat = weight_n_games * visitor_team_stats[stat] + weight_h2h * h2h_visitor_stats[stat]
        weighted_features.append(weighted_stat)

    future_game_features = np.array(weighted_features)

    print(['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'OREB_home',
           'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'OREB_away'])
    print(future_game_features)

    print("-------------------------------------\n\n\n")
    return future_game_features


X_data = games_data.copy()

for index, row in games_data.iterrows():
    print("generate_game_features(str(row['HOME_TEAM_ID']), str(row['VISITOR_TEAM_ID'])): ",  generate_game_features(games_data, row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']))
    X_data.loc[index, ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home',
                'REB_home', 'OREB_home', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away',
                'REB_away', 'OREB_away']] = generate_game_features(games_data, row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID'])


# Define the input variables (X) and target variable (Y)
X = X_data
Y = games_data[['spread', 'PTS_total', 'OREB_total']]

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
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [600],
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
print("input importance: ", model.feature_importances_)

print(X_test[:5, :])
print(Y_test.head(5))
print(y_pred[:5, :])



home_team_id = 1610612766  # Charlotte Hornets
# home_team_id = 1610612738  # Boston Celtics
visitor_team_id = 1610612761  # Toronto Raptors

# home_team_id = 1610612761  # Toronto Raptors
# visitor_team_id = 1610612766  # Charlotte Hornets

# Example: Predicting the spread, PTS_total, and OREB_total for a future game
future_game_features = generate_game_features(games_data, scaler, home_team_id, visitor_team_id)
# Scale the features using the provided scaler
future_game_features_scaled = scaler.transform(future_game_features)
future_game_prediction = model.predict(future_game_features_scaled)

print("Spread: ", future_game_prediction[0][0])
print("PTS_total: ", future_game_prediction[0][1])
print("OREB_total: ", future_game_prediction[0][2])
