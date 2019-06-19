%% to resize Images
%load the data
train_data = load('numplateValDataset.mat');

images = train_data.numberplate_dataset.imageFilename;

x = size(train_data.numberplate_dataset.imageFilename);
%% resize images
disp('press to start overwriting images')
pause

for i = 1: x(1)
    image = imread(train_data.numberplate_dataset.imageFilename{i});
    disp(i)
    image = imresize(image,1/2);
%     image = padarray(image,[420 0],0,'post');
    filename = sprintf('D:/NNFLValImages/image_%02d.png', i);
    imwrite(image,filename,'PNG');
    train_data.numberplate_dataset.imageFilename{i} = filename;
end


%% change bounding boxes

for i = 1:x(1)   
    train_data.numberplate_dataset.numplate{i} = round(train_data.numberplate_dataset.numplate{i}./2);
end

save New_Val_data train_data;
% 

