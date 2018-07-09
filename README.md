# Speech Enhancement using cGANs 
A fully convolutional end-to-end speech enhancement system using conditional Generative Adversarial Nets (cGANs).

_Under Construction_

#### Credits
The keras implementation of cGAN is based on the following repos
* [SEGAN](https://github.com/santi-pdp/segan)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [pix2pix](https://github.com/phillipi/pix2pix)
----
## Pre-requisites
1. Install [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/)
2. The experiments are conducted on a dataset from Valentini et. al.,  and are downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/1942). The following script can be used to download the dataset. *Requires [sox](http://sox.sourceforge.net/) for converting to 16kHz*.
```bash
$ ./download_dataset.sh
```
----
## Running the model
1. **Prepare data for training and testing the various models**. The folder path may be edited if you keep the database in a different folder. This script is to be executed only once and the all the models reads from the same location.
```python
python prepare_data.py
```
2. **Run the model**. This implementation offers several cGAN configurations. Edit the *opts* variable in run_segan.py for choosing the cofiguration. The results will be automatically saved to different folders. The folder name is generated from ``` python files_ops.py ``` and the foldername automatically includes different configuration options.
```python
python run_segan.py
```
3. **The testing is also done together with training**. Set *TEST_SEGAN = Flase* for disabling the testing. 
----
## Misc
* **This code loads all the data into memory for speeding up training**. But if you dont have enough memory, it is possible  to read the mini-batches from the disk using HDF5 read. In *run_segan.py* 
    ```python
     clean_train_data = np.array(fclean['feat_data'])
     noisy_train_data = np.array(fnoisy['feat_data'])
     ```
    change the above lines to 
    ```python
    clean_train_data = fclean['feat_data']
    noisy_train_data = fnoisy['feat_data']
    ```
    **But this can lead to a slow-down of about 20 times (on the test machine)** as the mini-batches are to be read from the     disk over several epochs.
