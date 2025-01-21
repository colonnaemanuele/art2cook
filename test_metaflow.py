from metaflow import FlowSpec, step
from src.art2mus.art2mus_4_train import (
    ARTGRAPH_FOLDER,
    AUDIO_ST,
    FMA_FOLDER,
    IMAGE_AUDIO_JSON,
    IMAGE_ST,
)

from my_dataset import ImageAudioDataset


class Art2MusTrainFlow(FlowSpec):
    dataset = ImageAudioDataset(
        json_file=IMAGE_AUDIO_JSON,
        images_dir=ARTGRAPH_FOLDER,
        img_emb_file=IMAGE_ST,
        audios_dir=FMA_FOLDER,
        audio_emb_file=AUDIO_ST,
    )

    train_data, val_data = dataset.train_val_test_split(val_size=0.2, random_state=0)

    @step
    def start(self):
        self.next(self.train_model)

    @step
    def train_model(self):
        print('Training model...')
        print('dataset is:', self.dataset)
        self.next(self.end)

    @step
    def end(self):
        print("Done training!")


if __name__ == "__main__":
    Art2MusTrainFlow()
