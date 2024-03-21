%% Load the training data
load('bioinf580_23_train_data.mat')
training_labels = cell2mat(data(:,end));
training_data = data(:,1:end-1);
%% Generate a model
training_features = generate_features(training_data);
model = generate_model(training_features,training_labels);
%% Load the testing (Validation) data
load('bioinf580_23_valid_data.mat')
testing_data = valid_data(randperm(size(valid_data,1)),:);
test_labels = cell2mat(testing_data(:,end));
test_data = testing_data(:,1:end-1);
%% Test the model on the testing data and return F score and AUC
testing_features = generate_features(test_data);

%% changed from here
if isa(model, 'SeriesNetwork')
    test_pred_y = double(classify(model, testing_features))-1;
elseif isa(model, 'TreeBagger')
    test_pred_y = str2double(predict(model, testing_features));
elseif isa(model, 'ClassificationSVM')
    test_pred_y = predict(model, testing_features);
else
    sprintf('There is unexpected error')
end

[~,~,~,AUC] = perfcurve(test_labels,test_pred_y,1);
[precision,recall,f1] = F1Scores(test_labels,test_pred_y);


function [precision,recall,f1Score] = F1Scores(y_true, y_pred)
    %positive class
    pc = 1;
    % precision = TP/(TP+FP)
    TPFP = sum(y_pred==pc);
    if TPFP == 0
        precision = 0;
    else
        precision = sum(y_true==pc & y_pred==pc) / TPFP;
    end
    % recall = TP/(TP+FN)
    TPFN = sum(y_true==pc);
    if TPFN == 0
        recall = 0;
    else
        recall = sum(y_true==pc & y_pred==pc) / TPFN;
    end
    f1Score = 2*(precision*recall)/(precision+recall);
    
    % Handle the case where the denominator is 0
    if isnan(f1Score)
        f1Score = 0;
    end
end
    