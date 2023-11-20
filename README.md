# Receipt Count Prediction

The Following code predicts the count of receipts for any month in 2023. In addition, the visualization presented also presents day-by-day prediction receipt count.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of [Docker](https://www.docker.com/).

## Installation

To install the project, follow these steps:

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/saikrishna-prathapaneni/fetch_prediction.git
    ```

2. Navigate to the project directory:
    ```
    cd fetch_prediction
    ```

## Usage

### Running the Flask App

To run the Flask app in a Docker container:

1. Build the Docker image:
    ```
    docker build -t fetch_receipt:latest .
    ```

2. Run the Docker container:
    ```
    docker run -p 5000:5000 fetch_receipt:latest
    ```

Visit `http://localhost:5000` in your web browser to access the app.

### Training the Model

To train the model, you can run the `train.py` script in a Docker container:

1. Ensure you have the necessary `data_daily` available in the parent directory.

2. Run the Docker container with a volume to persist model weights:
    ```
    docker run -v $(pwd)/model_weights:/usr/src/app/model_weights -it fetch_receipt:latest python train.py
    ```

## Development

For development purposes, you can mount the whole project directory to the container to reflect live changes:

