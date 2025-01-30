# Testing 🔎

To test our code, we adopted:

- **Deepchecks**: a library for diagnosing and monitoring deep learning models.

- **Pytest**: a tool for writing and running unit tests.

## Deepchecks

The [**Deepchecks library**](https://docs.deepchecks.com/stable/getting-started/welcome.html) is used in [this notebook](./deepchecks_evaluation.ipynb).

Since the dataset used for training or testing our model consists of images and audio files, it does not match the formats supported by Deepchecks (e.g., tabular data). To work around this, we structured a subset of our image dataset as if it were used in a classification task, allowing us to perform some checks on the images.

Specifically, we run the `ImageDatasetDrift()` and the `train_test_validation()` checks.

## Pytest

The [**Pytest library**](https://docs.pytest.org/en/stable/) is used in [this script](./pytest_tests.py) to run various tests ensuring proper functionality.

Executing the command:

```bash
python -m pytest art2cook/src/qa/pytest_tests.py
```

will run all the tests present within the specified script.
