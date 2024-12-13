# Base image to start with
FROM ghcr.io/mlflow/mlflow:v2.0.1

# Install gcc because we will need it to build some of the
# Python dependencies
RUN apt-get update && apt-get install -y gcc

# Update pip (Python's package manager) and install virtualenv
RUN python -m pip install --upgrade pip && pip install virtualenv

# Copy in our model
COPY RandomForest /RandomForest

# Install the model requirements
COPY requirements.txt /RandomForest/requirements.txt
RUN pip install -r /RandomForest/requirements.txt

# Tell it how to run the model
CMD ["mlflow", "models", "serve", "-h", "0.0.0.0", "-m", "RandomForest", "--env-manager=local"]
