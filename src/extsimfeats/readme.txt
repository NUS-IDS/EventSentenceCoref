We provide the extended similarity feature computation as a standalone module.
(Please refer to "ExtendedSimilarities" in the paper)


Requirements: Stanza library should be installed.
Glove embeddings and stopwords list must be obtained from their sources
and the paths appropriately set in ExptSettings file.

We used the glove.6B.200d.txt file from http://nlp.stanford.edu/data/glove.6B.zip 
and the English stopwords list from http://mallet.cs.umass.edu/

To Run:

Use FeatGen with the input directory of JSON files and specify the output directory where the features need to be written.

Example Usage:
python FeatGen.py rawdata processed

where an example rawdata folder for ACL-CASE 2021 data contains 
the files "en-test.json" and  "en-train.json".

We precompute the POS/NER and dependency trees using Stanza and serialize them and later use them for feature computation.


Note: The code provided is for English data.

For Portugues/Spanish please edit the lines suitably in KMDataLoader for the Stanza pipeline.

Since NER library is unavailable for Portuguese at this point, the 
call to markNEREntities should be changed to markNEREntities2 in the normalize function of KMDataLoader when Portuguese documents are being processed.
