# EasyOCR trainer

use `trainer.ipynb` with yaml config in `config_files` folder

In the config file make sure the following variables are correct:
- experiment_name
- valid_data
- train_data

Make sure to [download](https://lsumail2-my.sharepoint.com/:u:/g/personal/pherke1_lsu_edu/EQnr-ma5DARNg_CRG6odW9cBkCI_BlejcSrRlQVlW8kCow?e=Fiq9CB) and unzip the all_data folder. The structure should be as follows.<br>
trainer<br>
└── all_data<br>
    &nbsp;&nbsp;&nbsp;&nbsp;
    ├── train_data<br>
    &nbsp;&nbsp;&nbsp;&nbsp;
    ├── valid_data<br>
    &nbsp;&nbsp;&nbsp;&nbsp;
    └── ...

Follow the instructions on [EasyOCR](https://github.com/JaidedAI/EasyOCR) to set up the appropriate packages.

## Saving your Trained Model

Add to the finalSavedModels folder a folder containing your model's log files, final .pth file, and both the .pth files starting with best.


## How to use your custom model

The file easyOCR_customModel.ipynb has been set up to test custom models.

To use your own recognition model, you need the three files from the open-source or web-based approach above. These three files have to share the same name (i.e. `yourmodel.pth`, `yourmodel.yaml`, `yourmodel.py`) that you will then use to call your model with in the EasyOCR API.

We provide [custom_example.zip](https://jaided.ai/easyocr/modelhub/)
as an example. Please download, extract and place `custom_example.py`, `custom_example.yaml` in the `user_network_directory` (default = `~/.EasyOCR/user_network`) and place `custom_example.pth` in model directory (default = `~/.EasyOCR/model`)
Once you place all 3 files in their respective places, you can use `custom_example` by
specifying `recog_network` like this `reader = easyocr.Reader(['en'], recog_network='custom_example')`.
