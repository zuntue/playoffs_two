import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the NBA game data into a Pandas DataFrame
games_data = pd.read_csv('games_with_OREB.csv')

# Define the input variables (X) and target variable (Y)
X = games_data[['team_record', 'avg_pts_per_game', 'shooting_pct', 'home_status']]
Y = games_data['spread']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


# Load the NBA game data into a Pandas DataFrame
nba_data = pd.read_csv('nba_data.csv')

# Define the input variables (X) and target variable (Y)
X = nba_data[['team_record', 'avg_pts_per_game', 'shooting_pct', 'home_status']]
Y = nba_data['spread']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'n_jobs': -1,
}

model = xgb.XGBRegressor(**params)

model.fit(X_train, y_train)

# param_grid = {
#     'max_depth': [3, 4, 5, 6],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [50, 100, 150],
# }
#
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# grid.fit(X_train, y_train)
#
# best_params = grid.best_params_
# print("Best parameters found: ", best_params)

# model = xgb.XGBRegressor(**best_params)
# model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)