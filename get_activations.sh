#!/bin/bash

scriptDir=$1       # path to ConceptX script directory
inputPath=$2       # path to a sentence file
input=$3       # name of the sentence file
model=$4       # path to the model
sentence_length=$5   # maximum sentence length
minfreq=$6  # minimum word frequency
maxfreq=$7  # maximum word frequency
delfreq=$8  # delete frequency
layer=$9   # target layer

working_file=$input.tok.sent_len

mkdir $layer
cd $layer


cp ${inputPath}/$input $input.tok


# Do sentence length filtering and keep sentences max length of {sentence_length}
python ${scriptDir}/sentence_length.py --text-file $input.tok --length ${sentence_length} --output-file $input.tok.sent_len


# Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${working_file} --output-file ${working_file}.words_freq


# Extract layer-wise activations
python -m neurox.data.extraction.transformers_extractor --decompose_layers --filter_layers ${layer} --output_type json ${model} ${working_file} ${working_file}.activations.json


# Create a dataset file with word and sentence indexes
python ${scriptDir}/create_data_single_layer.py --text-file ${working_file} --activation-file ${working_file}.activations-layer${layer}.json --output-prefix ${working_file}


# Filter number of tokens to fit in the memory for clustering. Input file will be from step 4
python ${scriptDir}/frequency_filter_data.py --input-file ${working_file}-dataset.json --frequency-file ${working_file}.words_freq --sentence-file ${working_file}-sentences.json --minimum-frequency ${minfreq} --maximum-frequency ${maxfreq} --delete-frequency ${delfreq} --output-file ${working_file}


# Extract vectors
python -u ${scriptDir}/extract_data.py --input-file ${working_file}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json


cd ..
