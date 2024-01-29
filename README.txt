
Activations
----------
The ConceptX library was used to generate activations per layer for BERT models. After installing the required dependencies (as detailed on the library's github), the following command can be used to extract acivations of a specifc layer:

/bin/bash get_activations.sh <ConceptX_SCRIPT_DIR> <PATH_TO_SENTENCE_FILE> <NAME_SENTENCE_FILE> <BERT_MODEL> <MAX_SENTENCE_LENGTH> <MIN_WORD_FREQ> <MAX_WORD_FREQ> <DELETE_FREQ> <TARGET_LAYER>

This script creates a directory for the target layer and a number of files. The required point and vocab numpy files needed for clustering will be created in that folder.


Clustering:
-----------
Scripts 'create_{agglomerative,kmeans,leaders}.py' can be used to create the clustering text files.
The scripts require points.npy and vocab.npy of the activations, in addition to an output path and a number of desired clusters.

Refer to the individual scripts for further details on usage.



Alignment and Coverage
----------
After the clustering text files are produced, alignment and coverage w.r.t human-defined concepts can be computed using the script 'alignment.py'

Example usage: 
python alignment.py --sentence-file <SENTENCES> --label-file <LABELS> --cluster-file <CLUSTERING_RESULT> --threshold <THRESHOLD> --method M2

The script requires a sentence file and and its annotation label file that represent the human concept (e.g. Part of Speech tags).
Threshold is the percentage \theta at which an encoded cluster and a human-defined concept are assument aligned. In our experiments we used 0.95
