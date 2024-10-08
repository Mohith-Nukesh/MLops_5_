import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load and preprocess dataset
df=pd.read_csv('data/hour.csv')
int_columns=df.select_dtypes(include=['int']).columns
df[int_columns]=df[int_columns].astype('float64')
df['day_night']=df['hr'].apply(lambda x:'day' if 6<=x<=18 else 'night')
df.drop(['instant','casual','registered','dteday'],axis=1,inplace=True)

# Separate features and target
X=df.drop(columns=['cnt'])
y=df['cnt']

# Create pipeline for numerical features
numerical_features=['temp','hum','windspeed']
numerical_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',MinMaxScaler())
])

# Create pipeline for categorical features
categorical_features=['season','weathersit','day_night']
categorical_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(sparse_output=False,drop='first'))
])

# Preprocess features
X[numerical_features]=numerical_pipeline.fit_transform(X[numerical_features])
X_encoded=categorical_pipeline.fit_transform(X[categorical_features])
X_encoded=pd.DataFrame(X_encoded,columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features))
X=pd.concat([X.drop(columns=categorical_features),X_encoded],axis=1)

# Split the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Bike_Sharing_Model_Tracking")

# Initialize best model tracking variables
best_model=None
best_mse=float('inf')
best_r2=0.0
model_results={}

# Function to train and log models
def log_model_performance(model,model_name):
    global best_model,best_mse,best_r2,model_results
    with mlflow.start_run(run_name=model_name):
        # Fit model and predict
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

        # Calculate performance metrics
        mse=mean_squared_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)

        # Log metrics and parameters
        mlflow.log_param("model_type",model_name)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("r2",r2)

        # Log model with signature
        signature=infer_signature(X_train,model.predict(X_train))
        mlflow.sklearn.log_model(model,model_name,signature=signature)

        # Store model results
        model_results[model_name]={'mse':mse,'r2':r2}

        # Update best model if performance improves
        if mse<best_mse:
            best_mse=mse
            best_model=model_name
        if r2>best_r2:
            best_r2=r2

    mlflow.end_run()

# Train and log linear regression model
log_model_performance(LinearRegression(),"Linear_Regression")

# Train and log random forest model
log_model_performance(RandomForestRegressor(n_estimators=100,random_state=42),"Random_Forest")

# Display model comparison
print("\nModel Performance Comparison:")
for model_name,metrics in model_results.items():
    print(f"{model_name}: MSE={metrics['mse']}, R2={metrics['r2']}")

# Register the best-performing model
if best_model:
    print(f"\nBest Model: {best_model}, MSE={best_mse}")
    with mlflow.start_run(run_name=f"Best Model Registration: {best_model}"):
        mlflow.log_param("model_type",best_model)
        mlflow.log_metric("mse",best_mse)
        mlflow.log_metric("r2",best_r2)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{best_model}","Best_Bike_Sharing_Model")
    mlflow.end_run()
