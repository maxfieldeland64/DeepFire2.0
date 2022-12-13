import os
import sys
import zipfile
import numpy as np
from pathlib import Path

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr


class ZipGenerator(keras.utils.Sequence):
    """
    The generator designed to provide sampling model to the data.
    It it designed to open a zipfile containing the relevant samples
    (train, validate, test), then provide the x data and class from that sample.
    """

    def __init__(
        self, 
        zippath, 
        batch_size = 16, 
        shuffle = True, 
        binary=True,
        days = None):
        """
        Initialize the generator by requiring the location of the zipfile.

        Parameters:
            :str zippath:       The path of the zipfile, or optionally a list of
                                paths of zipfiles.

            :int batch_size:    The batch size.
            :bool shuffle:      Whether or not samples should be shuffled next
                                epoch.

            :bool binary:       Whether or not the generator is iterating
                                through binary data (True) or mask data (False)

            :list  days:        If desired, a list of relevant days can be
                                passed in, meaning that the generator will
                                only consider samples from that day.

        """
        self.classes = []
        self.binary = binary
        self.shuffle = shuffle
        self.batch_size = batch_size

        if (str(type(zippath)) == "<class 'str'>"):
            self.multizip = False

            self.zf = zipfile.ZipFile(zippath, "r")
            self.filepaths = list(self.zf.namelist())

        elif (str(type(zippath)) == "<class 'list'>"):
            self.multizip = True

            self.zfs = [zipfile.ZipFile(x, "r") for x in zippath]

            self.filepaths = []
            for z in self.zfs:
                self.filepaths.extend(list(z.namelist()))

            # Create pairings of each zipfile with their fire name
            self.pairs = {}
            for z in self.zfs:
                item = z.namelist()[0]
                fire_name = Path(item).parts[1]
                self.pairs[fire_name] = z

        if (not self.binary):
            # A little of file name checking weirdness necessitated here 
            # to make sampling later on much easier.

            self.outdim = ['0','0']
            self.indim = ['0', '0']

            for i in self.filepaths[0:10]:
                if ("outcome" in i):
                    parts = i.split('-')
                    self.outdim[0] = parts[-2]
                    self.outdim[1] = parts[-1].split('.')[0]

                elif ("sample" in i):
                    parts = i.split('-')
                    self.indim[0] = parts[-2]
                    self.indim[1] = parts[-1].split('.')[0]

            self.filepaths = [x for x in self.filepaths if "sample" in x]


        # Handle sorting paths for specific days, or multiple specific days.
        if (str(type(days)) == "<class 'list'>"):
            self.filepaths = [x for x in self.filepaths 
            if Path(x).parts[2] in days]

        elif (str(type(days)) == "<class 'str'>"):
            self.filepaths = [x for x in self.filepaths 
            if Path(x).parts[2] in [days]]

        else:
            pass

        self.on_epoch_end()

    def __len__(self):
        return (int(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates a batch based on the current index position.
        """

        # Get the next subset of paths for this batch.
        idxs = [i for i in range(index*self.batch_size, 
            (index+ 1) * self.batch_size)]

        x,y = self.__data_generation(idxs)

        self.classes.extend(y)
        
        return (x,y)

    def __data_generation(self, idxs):
        """
        Turns filepaths into samples.
        """

        fps = [self.filepaths[i] for i in idxs]

        x = []
        y = []

        # If we're iterating values with binary names, just do normal splits
        if (self.binary):
            for i in fps:

                if (self.multizip):
                    fire_name = Path(i).parts[1]
                    zf = self.pairs[fire_name]

                else:
                    zf = self.zf

                x.append(np.load(zf.open(i)))

                tail = i.split('-')[-1]
                outcome = int(tail.split('.')[0])

                y.append(outcome)

        # If we're iterating samples with masks, do some filepath substitution.
        else:
            for i in fps:

                if (self.multizip):
                    fire_name = Path(i).parts[1]
                    zf = self.pairs[fire_name]

                else:
                    zf = self.zf

                x.append(np.load(zf.open(i)))
                y_path = i.replace("sample", "outcome")

                # Guarantees only end of path (dimensions) are changed.
                y_path = y_path.replace("-{}-{}.npy".format(self.indim[0], 
                    self.indim[1]), "-{}-{}.npy".format(self.outdim[0], 
                    self.outdim[1]))

                y.append(np.load(zf.open(y_path)))


        return np.array(x), np.array(y)

    def on_epoch_end(self):
        """
        Shuffle in place at intiailization and end of each epoch.
        """

        if (self.shuffle):
            np.random.shuffle(self.filepaths)
