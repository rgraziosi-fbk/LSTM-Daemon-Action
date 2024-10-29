# Enhancing the Accuracy of Predictors of Activity Sequences of Business Processes
This project contains supplementary material for the article **"Enhancing the Accuracy of Predictors of Activity Sequences of Business Processes"** by **Muhammad Awais Ali**, **[Marlon Dumas](https://kodu.ut.ee/~dumas/)** and **Fredrik Milani**.

Our project builds upon an existing LSTM-based approach for next activity and case suffix prediction. We have expanded this initial implementation to include our Daemon action approach, alongside various baseline sampling approaches that are detailed in our paper. 

# Datasets:
Datasets can be found in the log folder or on the following link:
- [Logs](logs)

# Reproduce Results:

* To execute this code you need to install anaconda in your system, and create an environment using environment.yml specification provided in the repository.
  ```
  cd GenerativeLSTM
  conda env create -f environment.yml
  conda activate lstm-caise23
  ```

## My running instructions

### Training

- If present, remove the folder lstm/input_files/embedded_matix. It will be generated again when you launch training
- In dg_training.py adjust parameters to your needs. In particular: column_names, timeformat, file_name
- Run `cd lstm`
- Run `python dg_training.py`

Training generates two folders with two different names:

- One contains subfolders, one for each hyperparam space search + a csv with the results for each train
- The other one contains a "parameters" folder + a .h5 model (which is the best model across all trials)

This latter folder must be passed for log generation as 'folder' parameter in the next step.

### Log generation

- In dg_prediction.py choose 'folder', 'model_file' and 'rep' parameters. 'rep' indicates how many logs to generate.
- In model_predictor.py:55 choose the number of traces to generate
- Run `python dg_predictiction.py`

Generated logs will go to the specified 'folder'.

## Running the script

Once created the environment, you can perform training and testing by running the following python scripts dg_training.py  and dg_predictiction.py  through command line as  described below:

*Training LSTM for suffix prediction and remaining time:* To perform this task you need to set the required activity (-a) as 'training' followed by the name of the (-f) event log, and all the following parameters:

* Filename (-f): Log filename.
* Model family (-m): lstm
* Max Eval (-e): Maximum number of evaluations.
* Opt method (-o): Optimization method used. The available options are hpc and bayesian.

```
(lstm_env) C:\sc_lstm>python dg_training.py -f BPI_Challenge_2012.csv -m lstm -e 1 -o bayesian
```
* For training purposes, event log should exist in the input folder
* For training LSTM on different architectures, with in the dg_training.py file please specify the name of the desired architecture to train a model.


*Predictive task:* It is possible to predict case suffix.  To perform this task, you need to set the activity (-a) as ‘pred_sfx’ for case suffix and remaining time prediction. Additionally, it's required to indicate the folder where the predictive model is located (-c), and the name of the .h5 model (-b). Finally, you need to specify the method for selecting the next predicted task (-v) ‘random_choice’ or ‘arg_max’ or any other sampling approach and the number of repetitions of the experiment (-r).

```
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "arg_max" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "topk" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "d_action" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "random_choice" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "nucleus" -r 1


```

* The trained predictive model will be saved in the output folder along with the parameters and test set.
* After running the test set on the trained model, results for different sampling apporaches can be found in the same folder

