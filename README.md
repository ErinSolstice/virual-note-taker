# virtual-note-taker
## Installing dependencies

Package was developed using python 3.9.x and the Anaconda distribution.

In a fresh virtual environment run the following commands.
- install pytorch using instrucitons at [pytorch.org](https://pytorch.org/)
- `pip install easyocr`
- `pip install -r requirements.txt`

If you want to do any processing on the whiteboard data.
- `pip install -r requirementsDataProcess.txt`
- install [ImageMagick](https://imagemagick.org/index.php)

If you want to generate your own synthetic data follow the instructions at [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) in a fresh virtual environment.


## Running the virtual notetaker
To run the virtual note taker use either AllTogether.ipynb in jupyter notebooks or run AllTogether.py.

If using AllTogether.py pass the path to the image you wish to perform OCR on as a command line argument.

`python AllTogether.py --path sampleImages/slide.png`


## EasyOCR trainer
Follow the instructions on [EasyOCR](https://github.com/JaidedAI/EasyOCR) to set up the appropriate packages.

Use `trainer.ipynb` with yaml config in `config_files` folder

In the config file make sure the following variables are correct:
- experiment_name (select a new name for each run)
- train_data (path to folder containing all data for training and validation)
- valid_data (path to the validation data from trainer folder)
- select_data (path to the training data from all_data folder)
- saved_model (specify path if you want to do further training of a saved model)

The data used to train the models is included in the folder [download](https://lsumail2-my.sharepoint.com/:f:/g/personal/pherke1_lsu_edu/EskJvnoZlSlIoJm5iVsxUGsBzxwFyjd8pEe__ThRipXxbg?e=q4rohw). The structure of each zipped data folder should be placed as follows in the main repository. The two sets of validation data can be used during training, while the model should never see the test_data while training.

```
trainer  
└── all_data  
    ├── train_data  
    ├── valid_data-1  
    ├── valid_data-2  
    └── test_data
```

To test the accuracy of the model use test_orig.py and pass the config file as a command line option.  
`python test_orig.py --config testing_config_files/test_wb_opts.yaml`

## Saving your Trained Model

Add to the finalSavedModels folder a folder containing your model's log_dataset.txt, log_train.txt, opt.txt, best_accuracy.pth, and best_normED.pth files.

## How to use your custom model

To use your own recognition model, you need the three files from the open-source or web-based approach. These three files have to share the same name (i.e. `yourmodel.pth`, `yourmodel.yaml`, `yourmodel.py`) that you will then use to call your model with in the EasyOCR API.

EasyOCR provides [custom_example.zip](https://jaided.ai/easyocr/modelhub/)
as an example. Please download, extract and place `custom_example.py`, `custom_example.yaml` in the `user_network_directory` (default = `~/.EasyOCR/user_network`) and place `custom_example.pth` in model directory (default = `~/.EasyOCR/model`)
Once you place all 3 files in their respective places, you can use `custom_example` by
specifying `recog_network` like this `reader = easyocr.Reader(['en'], recog_network='custom_example')`.
