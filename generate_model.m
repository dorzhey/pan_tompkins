function model = generate_model(features,labels)
    X = features;
    y = labels;
    cvp_val = cvpartition(y, 'Holdout', 0.2);
    training_val = cvp_val.training;
    validation = cvp_val.test;
    train_X = X(training_val,:); 
    train_y = y(training_val);
    test_X = X(validation,:); 
    test_y = y(validation);
    % Use the best parameters to train a classifier with all the data
    model1 = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', sqrt(2)*100, 'BoxConstraint', 10);
    model2 = TreeBagger(50, train_X, train_y,'MinLeafSize', 1, 'Method','classification');
    model3 = getNN(train_X, train_y);
    test_pred_y1 = predict(model1, test_X);
    test_pred_y2 = str2double(predict(model2, test_X));
    test_pred_y3 = double(classify(model3, test_X))-1;
    AUCs = zeros(3,1);
    [~,~,~,AUCs(1)] = perfcurve(test_y,test_pred_y1,1);
    [~,~,~,AUCs(2)] = perfcurve(test_y,test_pred_y2,1);
    %[~,~,~,AUCs(3)] = perfcurve(test_y,test_pred_y3,1);

    [~,idx]=max(AUCs(:));
    if idx == 1
        model = fitcsvm(features, labels, 'KernelFunction', 'rbf', 'KernelScale', sqrt(2)*100, 'BoxConstraint', 10);
    elseif idx == 2
        model = TreeBagger(50, features, labels,'MinLeafSize', 1, 'Method','classification');
    else
        model = getNN(features, labels);
    end
    
    
end

function NeuralNet = getNN(X,y)
    numFeatures = size(X,2);
    numClasses = 2;
    y = categorical(y);
     
    layers = [
        featureInputLayer(numFeatures,'Normalization', 'zscore')
        fullyConnectedLayer(256)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(512)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(numClasses)
        sigmoidLayer
        classificationLayer];
    miniBatchSize=128;

    options = trainingOptions('adam', ...
        'MaxEpochs',3, ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'Verbose',false);


    NeuralNet = trainNetwork(X,y,layers,options);
    
end