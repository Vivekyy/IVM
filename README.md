# IVM

Projects related to the Integrated Value Model.

http://basilwhite.com/ivm/

Here you will find a neural network prediction system that will take in a description for a component and output a list of dependent components it believes will be related to this input.

How to use it:
The simplest way to use this is to add your description to the file titled "description.txt". Then, run use_model.py. This will then print out a list of dependent components it believes to be relevant to your input description.

The model needs to be trained whenever a new dependent component is added in. When a new component is added, the model can be optionally retrained to potentially increase prediction accuracy, but this is not entirely necessary. To retrain the model on an updated dataset, make sure the dataset is saved as 'IntegratedValueModelrawdata.xlsx' and run train.py. Alternatively, you can specify the name of the new dataset file using the --dataset_path parameter. When using a different excel spreadsheet, make sure the spreadsheet is formatted as it is in the raw data spreadsheet found on the IVM website (http://basilwhite.com/ivm/).

You can also modify the training protocol if needed and train multiple different networks, use -h to see the customizable parameters in model.py and in use_model.py.

If you alter the vocabulary size from 500 when training the model, be sure to specify the new vocabulary size in the --vocab_size parameter when running use_model.py.

If you are using models other than models/bow.pt, make sure to specify the name of the model you are using with the --model_path parameter when running use_model.py. The model should be saved into the "models" folder and the argument used for --model_path should be the name of the model without the folder name or the .pt extension (for example, running use_model.py with the model stored at "models/bow.pt" would be accomplished by typing "python use_model.py --model_path bow").