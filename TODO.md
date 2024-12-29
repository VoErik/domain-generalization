### Set up data preprocessing pipeline

Take any of the available datasets and return DataLoader object compatible with PyTorch

* take in folder `datasets/DATASET`
* check validity of dataset - clean dataset 
  * this should not be part of the usual pipeline, toggle via flag from config file
  * should be run initially upon first usage of dataset
* add convenience functions (num classes, size of dataset, ...)
* functions to efficiently create train, val, testsplit
  * testset should be ood always to test robustness
  * aka train on all but one domain, test on the remaining

We have imbalanced data
art_painting 2048
cartoon 2344
photo 1670
sketch 3929

art_painting 20.49844860374337
cartoon 23.461115003503153
photo 16.715043539185267
sketch 39.32539285356821

### Set up basic training loop

Create unoptimized training loop for torch resnet18 model
* model init
  * find set of sensible default hyperparams
* training loop
* basic eval (acc)

Later
* set up crossval function 