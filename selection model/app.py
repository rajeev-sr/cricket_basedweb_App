from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# [Include all your existing code here - file_map, bat_features, safe_numeric, load_and_prepare_data, train_and_predict]
# --- File mapping and features ---
file_map = {
    # Test
    'test_international.csv':('Test', 'both'),
    'duleep_runs_test.csv': ('Test', 'bat'),
    'ranji_runs_test.csv': ('Test', 'bat'),
    'duleep_wickets_test.csv': ('Test', 'bowl'),
    'ranji_wickets_test.csv': ('Test', 'bowl'),
    # ODI
    'odi_international.csv': ('ODI', 'both'),
    'u23_runs_ODI.csv':('ODI', 'bat'),
    'u23_wickets_ODI.csv':('ODI', 'bowl'),
    'vijay_runs_ODI.csv': ('ODI', 'bat'),
    'vijay_wickets_ODI.csv': ('ODI', 'bowl'),
    'cooch_runs_ODI.csv': ('ODI', 'bat'),
    'cooch_wickets_ODI.csv': ('ODI', 'bowl'),
    # T20
    't20_international.csv': ('T20', 'both'),
    'mushtaq_runs_t20.csv': ('T20', 'bat'),
    'mushtaq_wickets_t20.csv': ('T20', 'bowl')
}




bat_features = ['R', 'AVG', '100', '50']
bowl_features = ['WKTS', 'AVG', 'ECON', 'SR', '4W', '5W']

def safe_numeric(col):
    """Convert column to numeric, replace '-', missing, or bad values with 0."""
    return pd.to_numeric(col.replace('-', np.nan), errors='coerce').fillna(0)

def load_and_prepare_data():
    all_data = []
    for file, (fmt, typ) in file_map.items():
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue
        df['Format'] = fmt
        df['Type'] = typ
        df['Source'] = file

        # Standardize and clean batting columns
        if typ == 'bat' or typ == 'both':
            # Try to map to standard batting columns
            bat_map = {
                'PLAYER': ['PLAYER', 'Name', 'Known As'],
                'R': ['R', 'Runs'],
                'AVG': ['AVG', 'Bat Avg'],
                '100': ['100', '100s'],
                '50': ['50', '50s'],
                '4S': ['4S'],
                '6S': ['6S']
            }
            bdf = pd.DataFrame()
            for std_col, options in bat_map.items():
                for opt in options:
                    if opt in df.columns:
                        bdf[std_col] = df[opt]
                        break
                else:
                    bdf[std_col] = 0
            bdf['Format'] = fmt
            bdf['Type'] = 'bat'
            bdf['Source'] = file
            for col in bat_features:
                bdf[col] = safe_numeric(bdf[col])
            bdf['PLAYER'] = bdf['PLAYER'].astype(str)
            all_data.append(bdf)

        # Standardize and clean bowling columns
        if typ == 'bowl' or typ == 'both':
            bowl_map = {
                'PLAYER': ['PLAYER', 'Name', 'Known As'],
                'WKTS': ['WKTS', 'Wkts'],
                'AVG': ['AVG', 'Bowl Avg'],
                'ECON': ['ECON', 'E/R'],
                'SR': ['SR'],
                '4W': ['4W'],
                '5W': ['5W']
            }
            bdf = pd.DataFrame()
            for std_col, options in bowl_map.items():
                for opt in options:
                    if opt in df.columns:
                        bdf[std_col] = df[opt]
                        break
                else:
                    bdf[std_col] = 0
            bdf['Format'] = fmt
            bdf['Type'] = 'bowl'
            bdf['Source'] = file
            for col in bowl_features:
                bdf[col] = safe_numeric(bdf[col])
            bdf['PLAYER'] = bdf['PLAYER'].astype(str)
            all_data.append(bdf)
    if not all_data:
        raise ValueError("No data loaded. Please check your CSV files.")
    return pd.concat(all_data, ignore_index=True)

def train_and_predict(format_type, top_n=10):
    data = load_and_prepare_data()
    data = data[data['Format'] == format_type]

    # Batting
    bat_data = data[data['Type'] == 'bat'].copy()
    bat_data = bat_data.dropna(subset=['PLAYER'])
    X_bat = bat_data[bat_features].fillna(0)
    y_bat = bat_data['R'].fillna(0)
    if len(X_bat) > 5:
        scaler_bat = StandardScaler()
        X_bat_scaled = scaler_bat.fit_transform(X_bat)
        model_bat = RandomForestRegressor(n_estimators=100, random_state=42)
        model_bat.fit(X_bat_scaled, y_bat)
        bat_data['Predicted'] = model_bat.predict(X_bat_scaled)
        top_bat = bat_data[['PLAYER', 'Predicted']].sort_values('Predicted', ascending=False).head(top_n)
    else:
        top_bat = bat_data[['PLAYER', 'R']].sort_values('R', ascending=False).head(top_n).rename(columns={'R': 'Predicted'})

    # Bowling
    bowl_data = data[data['Type'] == 'bowl'].copy()
    bowl_data = bowl_data.dropna(subset=['PLAYER'])
    X_bowl = bowl_data[bowl_features].fillna(0)
    y_bowl = bowl_data['WKTS'].fillna(0)
    if len(X_bowl) > 5:
        scaler_bowl = StandardScaler()
        X_bowl_scaled = scaler_bowl.fit_transform(X_bowl)
        model_bowl = RandomForestRegressor(n_estimators=100, random_state=42)
        model_bowl.fit(X_bowl_scaled, y_bowl)
        bowl_data['Predicted'] = model_bowl.predict(X_bowl_scaled)
        top_bowl = bowl_data[['PLAYER', 'Predicted']].sort_values('Predicted', ascending=False).head(top_n)
    else:
        top_bowl = bowl_data[['PLAYER', 'WKTS']].sort_values('WKTS', ascending=False).head(top_n).rename(columns={'WKTS': 'Predicted'})

    print(f"\nTop {top_n} Batters for {format_type}:")
    print(top_bat.to_string(index=False))
    print(f"\nTop {top_n} Bowlers for {format_type}:")
    print(top_bowl.to_string(index=False))

# Example usage:
train_and_predict('Test', top_n=10)  # For Test
train_and_predict('ODI', top_n=10)   # For ODI
train_and_predict('T20', top_n=10)   # For T20


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        format_type = request.form.get('format')
    else:
        format_type = request.args.get('format', 'Test')
    
    try:
        data = load_and_prepare_data()
        format_data = data[data['Format'] == format_type]
        
        # Get top 5 batters and bowlers
        batters = format_data[format_data['Type'] == 'bat'] \
            .sort_values('R', ascending=False).head(10)[['PLAYER', 'R', 'AVG']]
        bowlers = format_data[format_data['Type'] == 'bowl'] \
            .sort_values('WKTS', ascending=False).head(10)[['PLAYER', 'WKTS', 'ECON']]
        
        # Get model predictions
        train_and_predict(format_type)  # This will print to console
        return render_template('results.html', 
                             format=format_type,
                             batters=batters.to_dict('records'),
                             bowlers=bowlers.to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
