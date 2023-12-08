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
  conda activate deep_generator
  ```
## Running the script

Once created the environment, you can perform each one of the tasks, specifying the following parameters in the lstm.py module, or by command line as is described below:

*Training LSTM for suffix prediction and remaining time:* To perform this task you need to set the required activity (-a) as 'training' followed by the name of the (-f) event log, and all the following parameters:

* Filename (-f): Log filename.
* Model family (-m): The available options are lstm, gru, lstm_cx and gru_cx.
* Max Eval (-e): Maximum number of evaluations.
* Opt method (-o): Optimization method used. The available options are hpc and bayesian.

```
(lstm_env) C:\sc_lstm>python dg_training.py -f BPI_Challenge_2012.csv -m lstm -e 1 -o bayesian
```

*Predictive task:* It is possible to predict case suffix.  To perform this task, you need to set the activity (-a) as ‘pred_sfx’ for case suffix and remaining time prediction. Additionally, it's required to indicate the folder where the predictive model is located (-c), and the name of the .h5 model (-b). Finally, you need to specify the method for selecting the next predicted task (-v) ‘random_choice’ or ‘arg_max’ or ` and the number of repetitions of the experiment (-r).

```
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "arg_max" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "topk" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "d_action" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "random_choice" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "nucleus" -r 1


```


## Pretrained model:
Pretrained models can be found in trained model folder or at the following link.
