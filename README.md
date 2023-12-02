# -Team_G4_BD4H
big data for healthcare team project


## instalation and setup

use conda to set up the env off of hte enviroment.yml file and then run:

```pip install -r requirements.txt```

if you are trying to source your data and not using the provided raw data place your data in the `data_raw` directory

once data has been sourced and is named the same run the `data_processing.py` file from the root dir.

after the data has been processed you should see files in the `data_processed` dir

you should then run `ranking.py` file to start the model 