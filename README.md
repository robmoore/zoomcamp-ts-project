# Machine Learning Zoomcamp Third Project

## Introduction

The organization [datatalks.club](https://datatalks.club) has provided an online course
entitled [Machine Learning Zoomcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html)
to provide a practical introduction to machine learning. The course is based on the book
[Machine Learning Bookcamp](https://www.manning.com/books/machine-learning-bookcamp) by
[Alexey Grigorev](https://alexeygrigorev.com/).

This project was undertaken as a project for the Machine Learning Zoomcamp course.

## Project description

This project grew out of an interest in using machine learning to predict the movement of a stock using only its prior
history. Initial research took me to a paper
entitled [Comparative Analysis of Deep Learning Models for Multi-Step Prediction of Financial Time Series](https://doi.org/10.3844/jcssp.2020.1401.1416)
, which reviews the performance of several different deep learning models on various indexes, among them the S&P 500.

The authors, Saugat Aryal et al., suggest two recent innovations (namely, Temporal Convolutional Networks (TCN) and
Neural basis expansion analysis for interpretable time series (N-BEATS)) significantly outperform more established ones.
Notably, TCN and N-BEATS perform exceptionally well on univariate S&P 500 index data. I decided to use the approach
followed in that paper to see if I could reproduce the results and learn about working with time-series data in a
machine learning context.

The authors follow an approach of evaluating the performance of a model using a lookback period or window to predict the
performance over the following period. They use a window of 16 days to forecast the subject's performance over the next
2, 3, 5, 7, and 10 days.

Furthermore, the authors use a sequence to sequence or multiple-input multiple-output (MIMO) approach to processing the
data. This method forecasts each day (t+1 to t+10) simultaneously, producing a single output containing each day (ten
values in total) rather than independently retraining and evaluating the model for each forecast period.

The paper discusses utilizing a walk-forward validation approach to evaluating the results using a sliding window that
moves day-by-day to arrive at an overall performance value for the entire period.

[Notebooks](https://github.com/itsaugat/time-series-prediction) by the primary author of the above-mentioned paper were
instrumental in understanding the details of the paper and replicating its approach. In following the specifics of the
sequence to sequence model and walk-forward validation, I was greatly assisted by the
article [Multi-Step LSTM Time Series Forecasting Models for Power Usage](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/)
by Jason Brownlee. Finally, the example of implementing time-series analysis in
the [Tensorflow tutorial on time-series analysis](https://www.tensorflow.org/tutorials/structured_data/time_series) and
its sample code (particularly the window generator) proved invaluable.

My findings suggest that while these approaches work well to minimize the error in predicting future prices, they do not
perform better than the baseline approach of using the value of the last day in the lookback period as the expected
value. Moreover, they appear to mimic the 'last' baseline method described in the project notebooks in arriving at a prediction. This outcome suggests that
the random-walk nature of the data results in a model that is no better than a naive prediction.

Future directions include incorporating other signals into the data (ie, multivariate models). Examples of such data include technical
signals (eg, those implemented by [ta-lib](https://mrjbq7.github.io/ta-lib/funcs.html)) and external sources, such
as [market sentiment indicators](https://www.investopedia.com/terms/m/marketsentiment.asp)
or traditional/social media. A detailed look at the later approach can be found in [Sentiment correlation in financial news networks and associated market movements](https://www.nature.com/articles/s41598-021-82338-6).

## Project tour

Some files of interest:

- [notebook-eda.ipynb](notebook-eda.ipynb): The Jupyter notebook used to perform EDA.
- [notebook-model.ipynb](notebook-eda.ipynb): The Jupyter notebook used to select and tune models.
- [train.py](train.py): Produces the model files used by the service to perform predictions.
- [predict.py](predict.py): Implementation of the prediction service using [Flask](https://flask.palletsprojects.com/).
- [models.py](models.py): Produces the models used in the Jupyter notebook.
- [utils.py](utils.py): Utility functions used in the Jupyter notebook and other modules.
- [constants.py](constants.py): Constant values used across modules.
- [predict_client.py](predict_client.py): An example client used to request predictions.
- [Pipfile](Pipfile): Defines the project dependencies.
- [Procfile](Procfile): Used by [Heroku](https://heroku.com) to run the prediction service.
- [data](data/gspc.csv): Example of data used to train the model.

## Getting started

This project requires [Docker](https://docs.docker.com/get-docker/) and
optionally [`make`](https://www.gnu.org/software/make/).
`make` is available typically readily available in Linux (in Debian, try `sudo apt-get install make`) and can be
installed on other OSs using their respective toolsets:

- Mac: [Homebrew](https://brew.sh/) or [xcode](https://apps.apple.com/us/app/xcode/)
- Windows: [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/)
  or [Chocolatey](https://chocolatey.org/)

The following assumes that you have `make` installed. However, `make` is simply a wrapper to ease execution of the
commands using Python or Docker, so please consult the [`Makefile`](Makefile) to see the underlying commands used to
work with the project.

To build the Docker image for the product, run `make build`.

To run the prediction service, run `make run-service`.

To run a Python client that makes example requests to the service, run `make run-client-local` to make a request to a
local version of the prediction service or `make run-client-remote` if you'd like to make requests to the service
running in
[Heroku](https://heroku.com).

The data used to train the model is available using the [yahooquery](https://pypi.org/project/yahooquery/) library.
Basic usage is provided in the [yahooquery documentation](https://yahooquery.dpguthrie.com/guide/ticker/intro/).

Binary versions of the model are available in [`bin`](bin). However, they can be regenerated using `make bin`.

## Deployment

The prediction service is available at https://zoomcamp-ts.herokuapp.com/ . It can be deployed using
`make heroku-deploy` and requires installation of the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).
Authentication is required to run `make heroku-deploy`. To authenticate, use `make heroku-login` and you will be
prompted to authenticate using a web browser. However, deployment is not necessary for testing as the service is
available publicly (see the notes regarding `make run-client-remote` above for how to make a request). The
file [`Procfile`](Procfile)
is included for Heroku's sake so it knows how to start the service (see the Heroku docs on
[Procfile](https://devcenter.heroku.com/articles/procfile) for further detail).

## Usage example

The service is available at https://zoomcamp-ts.herokuapp.com/predict. A request should contain the non-target variables
for a 16 day period. It is comprised of an array of 16 float values. The response contains the predicted price over a
10-day period.

The [example client](predict_client.py) sends a request using a random entry from the test data set. The following is an
example request and response made by the client.

### Request

```json
[
  4141.58984375,
  4124.66015625,
  4170.419921875,
  4185.47021484375,
  4163.259765625,
  4134.93994140625,
  4173.419921875,
  4134.97998046875,
  4180.169921875,
  4187.6201171875,
  4186.72021484375,
  4183.18017578125,
  4211.47021484375,
  4181.169921875,
  4192.66015625,
  4164.66015625
]
```

### Response

```json
[
  4462.90283203125,
  4216.52197265625,
  3968.47265625,
  3794.968994140625,
  4422.8388671875,
  5239.6044921875,
  3182.36572265625,
  3340.00634765625,
  5933.65869140625,
  3798.63623046875
]; actual prices: [4167.58984375, 4201.6201171875, 4232.60009765625, 4188.43017578125, 4152.10009765625, 4063.0400390625, 4112.5, 4173.85009765625, 4163.2900390625, 4127.830078125]
```

## Dependencies

The project uses [Pipenv](https://pipenv.pypa.io/) to manage its dependencies. When used outside of Docker, the
dependencies can be installed via `pipenv install` and the environment can be used via `pipenv shell`. To view the
dependencies, see [`Pipfile`](Pipfile).