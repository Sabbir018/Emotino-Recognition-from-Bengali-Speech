%% Start Here
clc;
clear all;
close all;
disp('Starting Processing...');

%%
disp('Creating Database...');
% Create an audioDatastore object to easily manage this database for training. The datastore allows you to collect necessary files of a file format and read them.

ads = audioDatastore('dataDir/', 'IncludeSubfolders', true,'FileExtensions', '.wav','LabelSource','foldernames');
[trainDatastore, testDatastore]  = splitEachLabel(ads,.80);

%%
disp('Saving Training Set...');
save('trainDatastore.mat');

save('testDatastore.mat');

%%
%Display the datastore and the number of speakers in the train datastore.
%Display the datastore and the number of speakers in the test datastore.
trainDatastoreCount = countEachLabel(trainDatastore); 
testDatastoreCount = countEachLabel(testDatastore); 

%%
% Reading from the train datastore pushes the read pointer so that you can iterate through the database. Reset the train datastore to return the read pointer to the start for the following feature extraction.

[sampleTrain, info] = read(trainDatastore);
%sound(sampleTrain,info.SampleRate);

reset(trainDatastore);

%%
disp('Feature Extraction...');
% Pitch and MFCC features are extracted from each frame using HelperComputePitchAndMFCC which performs the following actions on the data read from each audio file:
%        1. Collect the samples into frames of 30 ms with an overlap of 75%.
%        2. For each frame, use audiopluginexample.SpeechPitchDetector.isVoicedSpeech to decide whether the samples correspond to a voiced speech segment.
%        3. Compute the pitch and 13 MFCCs (with the first MFCC coefficient replaced by log-energy of the audio signal) for the entire file.
%        4. Keep the pitch and MFCC information pertaining to the voiced frames only.
%        5. Get the directory name for the file. This corresponds to the name of the speaker and will be used as a label for training the classifier.

lenDataTrain = length(trainDatastore.Files);
features = cell(lenDataTrain,1);
for i = 1:lenDataTrain
    [dataTrain, infoTrain] = read(trainDatastore);
    features{i} = PitchAndMFCC(dataTrain,infoTrain);
end
features = vertcat(features{:});
features = rmmissing(features);
head(features);   

%%
disp('Normalize the features...');
% Normalize the features by subtracting the mean and dividing the standard deviation of each column.

featureVectors = features{:,2:15};
save('featureVectors.mat','featureVectors');

m = mean(featureVectors);
s = std(featureVectors);
features{:,2:15} = (featureVectors-m)./s;
%head(features);   

%%
disp('Training the Classifier...');
% Train a classifier. Specify all the classifier options and train the classifier.

inputTable     = features;
predictorNames = features.Properties.VariableNames;
predictors     = inputTable(:, predictorNames(2:15));
response       = inputTable.Label;


trainedClassifier = fitcknn(predictors, response, 'Distance', 'euclidean', 'NumNeighbors', 5, 'DistanceWeight', 'squaredinverse', 'ClassNames', unique(response));
save('trainedClassifier.mat','trainedClassifier');

%%