# Machine Learning Zoomcamp Final Project

## Introduction

The organization [datatalks.club](https://datatalks.club) has provided an online course
entitled [Machine Learning Zoomcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html)
to provide a practical introduction to machine learning. The course is based on the book
[Machine Learning Bookcamp](https://www.manning.com/books/machine-learning-bookcamp) by
[Alexey Grigorev](https://alexeygrigorev.com/).

This project was undertaken as a final project for the Machine Learning Zoomcamp course.

## Project description

...

Some files of interest:

- [notebook-eda.ipynb](notebook-eda.ipynb): The Jupyter notebook used to perform EDA. 
- [notebook-model.ipynb](notebook-eda.ipynb): The Jupyter notebook used to select and tune models.
- [train.py](train.py): Produces the model binary used by the service to perform predictions.
- [predict.py](predict.py): Implementation of the prediction service using [Flask](https://flask.palletsprojects.com/).
- [predict_client.py](predict_client.py): An example client used to request predictions.
- [Pipfile](Pipfile): Defines the project dependencies.
- [Procfile](Procfile): Used by [Heroku](https://heroku.com) to run the prediction service.

## Getting started

This project requires [Docker](https://docs.docker.com/get-docker/) and
optionally [`make`](https://www.gnu.org/software/make/).
`make` is available typically readily available in Linux (in Debian, try `sudo apt-get install make`) and can be installed on other OSs using their respective
toolsets:

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

The data used to train the model is available using the [yahooquery](https://pypi.org/project/yahooquery/) library. Basic usage is provided in the [yahooquery documentation](https://yahooquery.dpguthrie.com/guide/ticker/intro/). 

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
for a 16 day period. It is comprised of an array of 16 float values. The response contains the predicted price over a 10-day period.

The [example client](predict_client.py) sends a request using a random entry from the test data set. The following is
an example request and response made by the client.

### Request

```json

```

### Response

```json
```

## Dependencies

The project uses [Pipenv](https://pipenv.pypa.io/) to manage its dependencies. When used outside of Docker, the
dependencies can be installed via `pipenv install` and the environment can be used via `pipenv shell`. To view the
dependencies, see [`Pipfile`](Pipfile).