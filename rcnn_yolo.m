
%% Creating the network for training 

% parameters of creating the yolov2 detection r-cnn
imageSize = [224 224 3];
numClasses = 1;

% anchor box size is determine using the estimating_boundingbox script
anchorBoxes = [49 36;
               67 23;
               138 50;
               98 34];

% use alexnet feature extraction 
network = resnet50();
featureLayer = 'activation_49_relu';

% create a new lgraph of alexnet feature extraction and yolov2 r-cnn object
% detection
lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,network,featureLayer);

% show lgraph
%analyzeNetwork(lgraph)

%% get data from the dataset

% get the 3 data (training, validation and testing)
train_data = load('numplateTrainingDataset.mat');
val_data = load('numplateValDataset.mat');
test_data = load('numplateTestDataset.mat');

trainds = train_data.numberplate_dataset;
valds = val_data.numberplate_dataset;
testds = test_data.numberplate_dataset;


%% Training of the data 

% TRAINING OPTION/ SETTINGS
%   - sgdm = stochastic gradient descent
%   - Batch size = 10
%   - Initial Learning Rate = 1e-3
%   - Shuffle = shuffle the data to distribute it
%   - Validation Data = prevent overfitting

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs',10,...
    'Shuffle','every-epoch');

% Train yolo v2 detector
[npNet1,info] = trainYOLOv2ObjectDetector(trainds,lgraph,options)

% Save the trained network
save npNet1
