disp('getting data')
% get the 3 data (training, validation and testing)
train_data = load('New_Train_data.mat');
val_data = load('New_Val_data.mat');
test_data = load('numplateTestingDataset.mat');

trainds = train_data.train_data.numberplate_dataset;
valds = val_data.train_data.numberplate_dataset;
testds = test_data.numberplate_dataset;

%%
%% Training of the data 
disp('begining training')
% TRAINING OPTION/ SETTINGS
%   - sgdm = stochastic gradient descent
%   - Batch size = 10
%   - Initial Learning Rate = 1e-3
%   - Shuffle = shuffle the data to distribute it
%   - Validation Data = prevent overfitting

options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 5e-5, ...
    'MaxEpochs',40,...
    'Shuffle','every-epoch');

% Train yolo v2 detector
[npNet5,info] = trainYOLOv2ObjectDetector(trainds,npNet4,options);
%%
% Save the trained network
disp('saving')
save npNet5 npNet5


%%
disp('runningtraining Validation')

numImages =1800;
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});


% Run detector on each image in the test set and collect results.
for i = 1:numImages
    % Read the image.
    I = imread(trainds.imageFilename{i});
    % Run the detector.
    [bboxes,scores,labels] = detect(npNet5,I);
    % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
    disp(i)
end

% Extract expected bounding box locations from test data.
expectedResults = trainds(:,2);
% Evaluate the object detector using average precision metric.
%%
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

% Plot precision/recall curve
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))


%%
disp('running validatoin Validation')

numImages =900;
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});


% Run detector on each image in the test set and collect results.
for i = 1:numImages
        % Read the image.
    I = imread(valds.imageFilename{i});
        % Run the detector.
    [bboxes,scores,labels] = detect(npNet5,I);
       % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
    disp(i)
end

% Extract expected bounding box locations from test data.
expectedResults = valds(:,2);
% Evaluate the object detector using average precision metric.

[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

% Plot precision/recall curve
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
%%
recall
