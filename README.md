## Deriving Boolean structures from distributional vectors

This code corresponds to the model presented in the TACL paper https://aclweb.org/anthology/Q/Q15/Q15-1027.pdf

The code is awfully messy and it desperately calls for a full rewrite. However, because of time constraints the best I can do for now
is making it available as is and hope to clean it up in future iterations.

### Usage

1. Get the semantic spaces (to use the count model from the paper run the fetch_spaces.sh script)

2. Edit the file parameters.lua to set the model hyperparameters and train/test datasets. Edit the config.lua file for other
general configuration (e.g. semantic spaces.)

3. Run the model with `th run_train.lua`. This will generate two output files in out/predict (one for the training set only for
debuggin reasons and another for the testing set) with 5 columns as follows:
    * The input pair
    * The pair label
    * The number of activated units in the intersection between the two boolean representations
    * The number of activated units in the boolean representation of the first unit in the pair
    * The number of activated units in the boolean representation of the second unit in the pair

The ratio between the 3rd and 4th columns yields the BI measure described in the paper. 

