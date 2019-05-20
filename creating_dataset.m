
% code for creating the .mat file for training, validation and testing
% dataset

%% Create a matlab .mat file for our dataset 

path_to_data ='D:\UFPR-ALPR\UFPR-ALPR dataset\validation';

imds = imageDatastore( path_to_data,'IncludeSubfolders',true);
imageFilename = imds.Files;


%% read images in the imds
% img = readimage(imds,1);
% 
% I = insertShape(img,'Rectangle',[911 525 73 29]);
% 
% imshow(I)


%% Creating all the bounding box ground truth into an array

row = size(imds.Files, 1);
numplate = cell(row, 1);

for i = 1:row
    
% get the path to the image
str = imds.Files{i};

% change the str from .png to .txt
labels = strrep(str, '.png','.txt');

% read the file
filetext = fileread(labels);

% get the number plate line
expr = '[^\n]*position_plate[^\n]*';
matches = regexp(filetext,expr,'match');

% remove all str in the char
bounding_box_str = erase(matches,'position_plate: ');

% convert char into a 1x4 matrix
bounding_box = str2num(bounding_box_str{1});

numplate{i,1} = bounding_box;
end

size(numplate)


%% create a table and save the data

numberplate_dataset = table(imageFilename, numplate)

save numplateValidationDataset numberplate_dataset

summary(numberplate_dataset)















