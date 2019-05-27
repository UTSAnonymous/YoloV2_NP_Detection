%% Creating the network for training 
disp('creating the network')
% parameters of creating the yolov2 detection r-cnn
imageSize = [227 227 3];
numClasses = 1;

% anchor box size is determine using the estimating_boundingbox script
anchorBoxes = [56 19; 44 29; 79 28;39 14;27 23;46 16;20 15 ; 33 12;28 10 ; 68 23];
%%
% use alexnet feature extraction 
network = resnet50();

featureLayer = 'activation_49_relu';

% create a new lgraph of alexnet feature extraction and yolov2 r-cnn object
% % detection

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,network,featureLayer);
%%
% show lgraph
analyzeNetwork(lgraph_4)

analyzeNetwork(lgraph)

%% get data from the dataset
disp('getting data')
% get the 3 data (training, validation and testing)
train_data = load('New_Train_data.mat');
val_data = load('numplateValDataset.mat');
test_data = load('numplateTestingDataset.mat');

trainds = train_data.train_data.numberplate_dataset;
valds = val_data.numberplate_dataset;
testds = test_data.numberplate_dataset;

% augimdsTrain = augmentedImageDatastore([227 227],train_data);


%%
disp('showing image')
% Add the fullpath to the local vehicle data folder.
% trainds.imageFilename = fullfile(pwd,trainds.imageFilename{1});

% Read one of the images.
I = imread(trainds.imageFilename{50});

% Insert the ROI labels.
I = insertShape(I,'Rectangle',trainds.numplate{50});

% Resize and display image.
% I = imresize(I,3);

imshow(I)


%% Training of the data 
disp('begining training')
% TRAINING OPTION/ SETTINGS
%   - sgdm = stochastic gradient descent
%   - Batch size = 10
%   - Initial Learning Rate = 1e-3
%   - Shuffle = shuffle the data to distribute it
%   - Validation Data = prevent overfitting

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 6, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs',60,...
    'Shuffle','every-epoch');

% Train yolo v2 detector
[npNet1,info] = trainYOLOv2ObjectDetector(trainds,lgraph_5,options)
%%
% Save the trained network
disp('saving')
save npNet1 npNet1 info

%% Testing
% Create a table to hold the bounding boxes, scores, and labels output by
% the detector.     
data = load('npNet1.mat');

%%
detector = npNet1;

% Read a test image.
I = imread(trainds.imageFilename{900});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
imshow(I)

%%
disp('running Validation')

numImages = 1800;
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});


% Run detector on each image in the test set and collect results.
for i = 1:numImages
        % Read the image.
    I = imread(trainds.imageFilename{i});
        % Run the detector.
    [bboxes,scores,labels] = detect(npNet1,I);
       % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
    disp(i)
end

% Extract expected bounding box locations from test data.
expectedResults = trainds(:,2);
% Evaluate the object detector using average precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

% Plot precision/recall curve
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
