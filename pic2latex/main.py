""""
Reads the image file, calls functions in preprocessing module,
loads the model and does predictions, calls functions in post-processing module
for output txt file.
"""


# calls all functions in preprocessing module


def preprocessing_segmentation():
    pass

# loads model
# calls helper function for each label
# return labels and positions


def model_classification():
    pass

# returns latex text


def postprocessing_latex():
    pass


def main():
    print("running as a script")  # parameters below
    preprocessing_segmentation()  # img path, save path (?)
    model_classification()        # segments (as a diff file ?)
    postprocessing_latex()        # labels, positions


# execute code when the file runs as a script, but not when it’s imported as a module
if __name__ == '__main__':
    main()
