# DVC (Data Version Control) 📁

The [**DVC library**](https://dvc.org/doc) is used in our project to manage all the data needed for training and testing our model. DVC supports various remote storage options, including Amazon S3, Azure, and Google Drive. In our case, we chose to use **Google Drive** as remote storage for managing datasets.

## Usage

Setting up DVC enables the following functionalities:

- **Tracking Data**: DVC tracks changes in datasets without storing them directly in Git.
- **Reproducibility**: Ensures consistent experiments by managing dependencies and outputs.
- **Collaboration**: Simplifies data sharing and synchronization across team members.

## Pull Data from the Remote Storage

Executing the command:

```bash
dvc pull
```

will download the data tracked by DVC from remote storage.
