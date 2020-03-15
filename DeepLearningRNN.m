file = "training";
optimizer = 'adam';
data = readtable(file+".xlsx");
data.classifier = categorical(data.classifier);
 
% f = figure;
% f.Position(3) = 1.5*f.Position(3);
% h = histogram(data.classifier);
% xlabel("Class")
% ylabel("Frequency")
% title("Class Distribution")

cvp = cvpartition(data.classifier,'Holdout',0.30);
%c=cvpartition(data.classifier,'KFold',10);
dataTrain = data(training(cvp),:);  
dataValidation = data(test(cvp),:);

textDataTrain = dataTrain.tweet;
YTrain = dataTrain.classifier;

textDataValidation = dataValidation.tweet;
YValidation = dataValidation.classifier;

% figure
% wordcloud(textDataTrain);
% title("Training Data");


textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);
documentsTrain = erasePunctuation(documentsTrain);

textDataValidation = lower(textDataValidation);
documentsValidation = tokenizedDocument(textDataValidation);
documentsValidation = erasePunctuation(documentsValidation);

encTrain = wordEncoding(documentsTrain);
encValidation = wordEncoding(documentsValidation);
tweetLengths = doclength(documentsTrain);

% figure
% histogram(tweetLengths)
% title("Tweet Lengths")
% xlabel("Length")
% ylabel("Number of tweets")
  
XTrain = doc2sequence(encTrain,documentsTrain,'Length',19);
XValidation = doc2sequence(encValidation,documentsValidation,'Length',19);

inputSize = 1;
embeddingDimension = 100;
numHiddenUnits = encTrain.NumWords;
hiddenSize = 180;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numHiddenUnits)
    lstmLayer(hiddenSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
options = trainingOptions(optimizer, ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);
net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XValidation);
accuracy = sum(YPred==YValidation)/numel(YPred);

reportsNew = readtable("testing.xlsx");
YTestValidation = reportsNew.classifier;
reportsNew = lower(reportsNew.tweet);
documentsNew = tokenizedDocument(reportsNew);
documentsNew = erasePunctuation(documentsNew);
encTest = wordEncoding(documentsNew);
XNew = doc2sequence(encTest,documentsNew,'Length',19);
[labelsNew,score] = classify(net,XNew);

%to be typed in command window 
T=table(reportsNew,string(labelsNew));
testAccuracy = sum(string(labelsNew)==YTestValidation)*100/numel(string(labelsNew));
writetable(T,"testedWith"+file+"_"+optimizer+".xlsx");


