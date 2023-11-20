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

The app requests for a date to be entered, pass the appropriate date to get the predictions.

### Training the Model

The mode is trained and model weights are present in `model_weights` folder, yet if you want to train the model, you can run the `train.py [OPTIONS]` script in a Docker container:

1. Ensure you have the necessary `data_daily` available in the parent directory.

2. Run the Docker container with a volume to persist model weights:
    ```
    docker run -v $(pwd)/model_weights:/usr/src/app/model_weights -it fetch_receipt:latest python train.py
    ```
    
### Parameter for Training the Model
1.  `--datafile` (default: "data_daily.csv")
    Description: Specify the location of the data file.
2.  `--include_l2reg` (default: False)
    Description: Use L2 regularization during model training.
4.  `--include_l1reg` (default: False)
    Description: Use L1 regularization during model training.
5.  `--l1_lambda` (default: 0.001)
    Description: Set the value of lambda for L1 regularization.
6.  `--l2_lambda` (default: 0.003)
    Description: Set the value of lambda for L2 regularization.
7.  `--lr` (default: 0.01)
    Description: Set the learning rate for training the model.
8.  `--batch_size` (default: 8)
    Description: Specify the batch size for training the model.
9.  `--epochs` (default: 20)
    Description: Set the number of epochs for training the model.
## Further Development

1. While the Model used here is simple, yet the features extracted are highly correlated with the variable of interest with a **added interpretability** nature associated to it
2. The Model used here is Linear Regression which not suitable for Time series analysis (Testing with LSTM, XGboost is necessary) yet due to constraint in the data size available, I pursuied Linear Regreesion with regualrisers in place, with some great features
3. Abalation study is not done with all the epoch sizes and learning rates in addition to regalarisers, comprehensive evaluation is needed here



