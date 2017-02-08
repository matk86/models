#!/usr/bin/env python
"""
Split the raw data into training and validation sets
"""
import sys
import os
import shutil
import numpy as np
from PIL import Image


print(sys.argv)

if len(sys.argv) == 1:
    print("usage: generate_train_validation_sets.py [raw data dir] [processed data dir] [percentage of data to be set for validation]")
    sys.exit()

labels = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]


raw_data_dir = sys.argv[1]
processed_data_dir= sys.argv[2]
percentage = float(sys.argv[3])

data_dir = os.path.expanduser(raw_data_dir) # directory containing all the data
validation_dir = os.path.join(os.path.expanduser(processed_data_dir), "validation")
training_dir = os.path.join(os.path.expanduser(processed_data_dir), "train")

print(data_dir, validation_dir, training_dir, percentage)

#nvalidation_samples = 1 
#ntrain_samples = 5

ntotal = 0
num_validation_samples = 0
for l in labels:
    src_dir = os.path.join(data_dir, l)
    files = os.listdir(src_dir)
    nfiles = len(files)
    ntotal += nfiles
    print("{} : {}".format(l, nfiles))
    nvalidation_samples = int(nfiles*percentage/100.) # % of the data
    num_validation_samples += nvalidation_samples
    validation_samples = np.random.choice(files, nvalidation_samples)
    train_dir = os.path.join(training_dir, l)    
    val_dir = os.path.join(validation_dir, l)
    os.makedirs(train_dir)    
    os.makedirs(val_dir)
    for f in files:
        src_file = os.path.join(src_dir, f)
        try:
            im = Image.open(src_file)
            im.verify()
        except:
            print("Issue with : {}".format(src_file))
            continue
        dest_dir = val_dir if f in validation_samples else train_dir
        shutil.copy(src_file, dest_dir)
print("Total samples: {}, validation samples: {}".format(ntotal, num_validation_samples))
