# EasyOCR trainer

use `trainer.ipynb` with yaml config in `config_files` folder

In the config file make sure the following variables are correct:
- experiment_name
- valid_data
- train_data

Make sure to unzip the all_data folder. The structure should be as follows.
trainer
└── all_data
    ├── trian_data
    ├── valid_data
    └── ...

Follow the instructions on [EasyOCR](https://github.com/JaidedAI/EasyOCR) to set up the appropiate packages.


## How to use your custom model

The file easyOCR_customModel.ipynb has been set up to test custom models.

To use your own recognition model, you need the three files from the open-source or web-based approach above. These three files have to share the same name (i.e. `yourmodel.pth`, `yourmodel.yaml`, `yourmodel.py`) that you will then use to call your model with in the EasyOCR API.

We provide [custom_example.zip](https://jaided.ai/easyocr/modelhub/)
as an example. Please download, extract and place `custom_example.py`, `custom_example.yaml` in the `user_network_directory` (default = `~/.EasyOCR/user_network`) and place `custom_example.pth` in model directory (default = `~/.EasyOCR/model`)
Once you place all 3 files in their respective places, you can use `custom_example` by
specifying `recog_network` like this `reader = easyocr.Reader(['en'], recog_network='custom_example')`.
