# Implementation of Robust H-UCRL Algorithm


[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/rhucrl/master?label=master%20build%20and%20test&token=fa2c21d3aa7c7b3e2b6a51aa824e135bd2f85b31)](https://app.circleci.com/pipelines/github/sebascuri/rhucrl)
[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/rhucrl/dev?label=dev%20build%20and%20test&token=fa2c21d3aa7c7b3e2b6a51aa824e135bd2f85b31)](https://app.circleci.com/pipelines/github/sebascuri/rhucrl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/hug/)


To install create a conda environment:
```bash
$ conda create -n rhucrl python=3.7
$ conda activate rhucrl
```

```bash
$ pip install -e .[test,logging,experiments]
```

For Mujoco (license required) Run:
```bash
$ pip install -e .[mujoco]
```

On clusters run:
```bash
$ sudo apt-get install -y --no-install-recommends --quiet build-essential libopenblas-dev python-opengl xvfb xauth
```


## Running an experiment.
```bash
$ python exps/run $ENVIRONMENT $AGENT
```

For help, see
```bash
$ python exps/run.py --help
```

## Pre Commit
install pre-commit with
```bash
$ pip install pre-commit
$ pre-commit install
```

Run pre-commit with
```bash
$ pre-commit run --all-files
```


## CIRCLE-CI

To run locally circleci run:
```bash
$ circleci config process .circleci/config.yml > process.yml
$ circleci local execute -c process.yml --job test
```

## Goals
Environment goals are passed to the agent through agent.set_goal(goal).
If a goal moves during an episode, then include it in the observation space of the environment.
If a goal is to follow a trajectory, it might be a good idea to encode it in the reward model.

## Policies
Continuous Policies are "bounded" between [-1, 1] via a tanh transform unless otherwise defined.
For environments with action spaces with different bounds, up(down)-scale the action after sampling it.
