# Experiment_with_weka

This is used with weka-stable-3.6.6. Before running this code, you have to set up weka on your system.

This code uses just J48 and NaiveBayesSimple options. It can be easily changed to try different options.

TO COMPILE:
>javac -cp weka-stable-3.6.6.jar Experimenter.java

TO RUN:
>java -cp weka-stable-3.6.6.jar;. Experimenter


INPUT REQUIRED:
It will ask for training data file path, it's in the folder.
It will ask for test data file path, it's in the folder.
It will ask for inclusive range for minNumObj, from and to, for J48.
It will ask for inclusive range for numFolds, from and to, for J48.
