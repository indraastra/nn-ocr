# Port coursera-ml nn
- [DONE] Convert .mat data for use with numpy
- [DONE] Sigmoid / relu activation
- [DONE] Forward prop
- [DONE] Backward prop
- [DONE] Runner with metaparameters tweakable from cli - use http://click.pocoo.org/5/
- [DONE] Ensure accuracy is consistent between octave and numpy versions

# Numbers
- [DONE] Generate data for all numbers in all fonts => write to .dat
- [DONE] Randomly divide into training and test sets
- [DONE] Use algorithm from above

# UI
- Live classification from canvas drawing
- Translate and resize canvas image
- [DONE] Provide list of matches is order of probability

# Training/Classification Pipeline
- Trainer should perform metaparameter optimization
    -- Split dataset into training, CV, and test sets
    -- Take cross product of metaparameters, train on each combination
    -- For each combination, record performance on CV set
    -- Return combination that performs best on CV as well as errors
       for each set
    -- Interface should be:
        input: training data (X, y), num_classes
        output: weights
- Classifier should allow for top n matching classes to be returned
    -- Interface should be:
        input: weights, num_classes
        output: top n matching classes

