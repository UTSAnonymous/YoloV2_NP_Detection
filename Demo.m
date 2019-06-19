% load('npNet2.mat');
detector = npNet4;

vidPlayer = vision.VideoPlayer;
train_data = load('New_Train_data.mat');
trainds = train_data.train_data.numberplate_dataset;
val_data = load('New_Val_data.mat');
valds = val_data.train_data.numberplate_dataset;


for i = 1:900

% Read a test image.

I = imread(trainds.imageFilename{i});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
try
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
end
vidPlayer(I);
% pause(0.001);
end