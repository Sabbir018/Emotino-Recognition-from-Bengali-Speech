%% Start here
clear all;
close all;
clc;
disp('Testing with a data...');
%% Open user interface of matlab to get file for testing
disp('Select a file to open...');
[filename, pathname] = uigetfile({'*.*'}, 'Pick a Speech Audio File');
[speech, speechFs] = audioread([pathname,filename]);

%% Extract Features
featurestest = cell(1,1);

info = struct('SampleRate',speechFs,'FileName',[pathname,filename],'Label',[1:1]);

featurestest{1} = PitchAndMFCC(speech,info);
featurestest = vertcat(featurestest{:});
featurestest = rmmissing(featurestest);

featurevectors = featurestest{:,2:15};

m = mean(featurevectors);
s = std(featurevectors);
featurestest{:,2:15} = (featurestest{:,2:15}-m)./s;
%head(featurestest); % Display the first few rows

%% Testing the feature with the model
load('trainedClassifier.mat');
result = HelperTestKNNClassifier(trainedClassifier, featurestest);

%%
%fid = fopen('output.txt','w');
I = result(1,3);

load('a.mat');
if (isequal(I,a))
    %fwrite(fid, 'Happy');
    disp('Happy');
%fclose(fid);
else
    load('b.mat');
    if (isequal(I,b))
        %fwrite(fid, 'Sad');
        disp('Sad');
    %fclose(fid);
    else
        load('c.mat');
        if (isequal(I,c))
            %fwrite(fid, 'Angry');
            disp('Angry');
        %fclose(fid);
        else
            %fwrite(fid, 'Neutral');
            disp('Neutral');
        %fclose(fid);
        end
    end
end

%winopen('output.txt');
%% Clear All
