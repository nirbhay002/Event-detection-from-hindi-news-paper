### Files Overview

1. utils.py
   - Contains utility functions for data preprocessing and conversion.
   - Key functions include:
     - `transform_label`: Transforms multi-labels into single labels.
     - `transform_probs`: Transforms probabilities to corresponding classes.
     - `get_w_children`: Extracts word children from XML tree.
     - `extract_entities`: Extracts entities from BIO tags.
     - `get_entity_info`: Gets information about entities.
     - `xml_to_conll`: Converts XML to CoNLL format.
     - `xml_to_json`: Converts XML to JSON format.
     - `get_all_json`: Gets all JSON data from XML files in a folder.
     - `process_lang`: Processes language data.

2. Processed data
   - Jupyter Notebook for processing language data.
   - Parses XML files, extracts entity information, and converts them into JSON format.
   - Uses functions from `utils.py` to achieve this.

3. Main_Model.ipynb
   - Jupyter Notebook for training models and generating predictions.
   - Loads training and testing data, preprocesses it, and trains machine learning models (Random Forest Classifier and Conditional Random Fields).
   - Evaluates model performance and generates XML output files.
   - Relies on functions from `utils.py` for data processing and evaluation.

