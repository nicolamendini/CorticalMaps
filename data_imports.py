# Helper functions useful to develop Models of Cortical Maps
# Author: Nicola Mendini

import numpy as np
import torch
import os
import struct


# function to import and unpack the norb dataset
def import_norb():
    
    DATA_FOLDER = "./the-small-norb-dataset-v10/"

    PREFIXES = {
        'train': 'smallnorb-5x46789x9x18x6x2x96x96-training-',
        'test': 'smallnorb-5x01235x9x18x6x2x96x96-testing-',
    }
    
    FILE_TYPES = ['info', 'cat', 'dat']

    # helper function to read int from file
    def read_int(f):
        num, = struct.unpack('i', f.read(4))
        return num


    # From https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/ 
    # "The magic number encodes the element type of the matrix"
    # Note: I only copied over the ones I needed for these files.
    map_magic_number_to_data_type = {
        '1e3d4c55': np.uint8,
        '1e3d4c54': np.int32,
    }

    loaded_data = {}

    for dataset, prefix in PREFIXES.items():
        for filetype in FILE_TYPES:
            filename = prefix + filetype + ".mat"
            print('Reading {}'.format(filename))
        
            file_loc = os.path.join(DATA_FOLDER, filename)
            with open( file_loc, 'rb') as f:
                # Read the magic_num, convert it to hexadecimal, and look up the data_type
                raw_magic_num = read_int(f)
                magic_num = format(raw_magic_num, '02x')
                data_type = map_magic_number_to_data_type[magic_num]
                print('dtype', data_type)

                #Read how many dimensions to expect
                ndim = read_int(f)
            
                # Read at least 3 ints, or however many ndim there are
                shape = [
                    read_int(f)
                    for i in range(max(ndim, 3))
                ]   
                # But in case ndims < 3, take at most n_dim elements
                shape = shape[:ndim]
                print('shape', shape)
    
                # Now load the actual data!
                loaded_data[(dataset, filetype)] = np.fromfile(
                    f, 
                    dtype=data_type, 
                    count=np.prod(shape)
                ).reshape(shape)

            

    x_raw = torch.Tensor(loaded_data[('train', 'dat')]) / 255.
    y = torch.Tensor(loaded_data[('train', 'cat')])
    info = torch.Tensor(loaded_data[('train', 'info')])
    x_raw.shape, y.shape, info.shape

    x = torch.zeros(5,5,972,2,96,96)
    count = 0
    instances = [4,6,7,8,9]

    for lab in range(5):
        for ins in range(5):
            count=0
            for lig in range(6):
                for ele in range(9):
                    for azi in range(18):
                        x[lab,ins,count] = x_raw[(y==lab)&(info[:,0]==instances[ins])&(info[:,1]==ele)&(info[:,2]==azi*2)&(info[:,3]==lig)]
                        count += 1
                        
    return x

