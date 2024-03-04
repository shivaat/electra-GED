# WORK IN PROGRESS


# Grammatical Error Detection using HuggingFace transformers

This repository contains the code to perform token-level grammatical error detection using HuggingFace transformers (ELECTRA in particular) 


## How to Run the System

1. Copy your conll-formatted data files to the path data/.<sup>1</sup>
2. Choose the configuration file from the experiments/ directory or make your own config file with the same fileds as in the files in the config directory.
3. Run `train.py config/{config.json}` for training
4. Run `load_test.py {path_to_the_best_model_saved} -input {path_to_conll_test_file} -out {path_to_output_directory}` to evaluate and write the predictions into a new file in `{path_to_output_directory}`. 

---

[1] The FCE and W&I data are from BEA 2019 Shared Task on Grammatical Error Correction:
https://www.cl.cam.ac.uk/research/nl/bea2019st/

For access to the already prepared conll-fortmatted datasets, please contact us.

---

