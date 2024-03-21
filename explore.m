rng 'default' %random seed
load('bioinf580_23_train_data.mat')

%% data exploration
labels = cell2mat(data(:,end));
ecgs = data(:,1:end-1);
Fs = 300;
%% class imbalance
length(labels(labels==1))/length(labels)
%% Look at data
T = 1/Fs;

subplot(3,2,1)
signal = ecgs{1};
L = length(signal)*T;
t = 0:T:L-T;
plot(t, signal);
title('Normal')

subplot(3,2,2)
signal = ecgs{7};
L = length(signal)*T;
t = 0:T:L-T;
plot(t, signal);
title('FA')

subplot(3,2,3)
signal = ecgs{2};
L = length(signal)*T;
t = 0:T:L-T;
plot(t, signal);


subplot(3,2,4)
signal = ecgs{8};
L = length(signal)*T;
t = 0:T:L-T;
plot(t, signal);

subplot(3,2,5)
signal = ecgs{3};
L = length(signal)*T;
t = 0:T:L-T;
plot(t, signal);

subplot(3,2,6)
signal = ecgs{10};
L = length(signal)*T;
t = 0:T:L-T;
plot(t, signal);

%% Generate Features
training_features = generate_features(ecgs);

X = training_features;
y = labels;

%% Cross Validation
%model = generate_model(training_features,training_labels);
clc;
if exist('results','var') == 1
    disp(results)
end
cvp = cvpartition(y,'KFold',10);
metrics_svm = struct();
metrics_rf = struct();
metrics_nn = struct();

for i = 1:cvp.NumTestSets
    training_cv = cvp.training(i);
    test_cv = cvp.test(i);
    
    train_X_cv = X(training_cv,:); % form the training data
    train_y_cv = y(training_cv); 
    
    test_X_cv = X(test_cv,:); % form the testing data
    test_y_cv = y(test_cv);
    %train
    model1 = fitcsvm(train_X_cv, train_y_cv, 'KernelFunction', 'rbf', 'KernelScale', sqrt(2)*5, 'BoxConstraint', 2);
    model2 = TreeBagger(150, train_X_cv, train_y_cv,'MinLeafSize', 10, 'Method','classification');
    model3 = getNN(train_X_cv, train_y_cv);
    %perfomance on train data
    train_pred_y1 = predict(model1, train_X_cv);
    train_pred_y2 = str2double(predict(model2, train_X_cv));
    train_pred_y3 = double(classify(model3, train_X_cv))-1;
    
    [~,~,~,metrics_svm.train_AUC(i)] = perfcurve(train_y_cv,train_pred_y1,1);
    [~,~,~,metrics_rf.train_AUC(i)] = perfcurve(train_y_cv,train_pred_y2,1);
    [~,~,~,metrics_nn.train_AUC(i)] = perfcurve(train_y_cv,train_pred_y3,1);
    [metrics_svm.train_precision(i),metrics_svm.train_recall(i),metrics_svm.train_f1(i)] = F1Scores(train_y_cv,train_pred_y1);
    [metrics_rf.train_precision(i),metrics_rf.train_recall(i),metrics_rf.train_f1(i)] = F1Scores(train_y_cv,train_pred_y2);
    [metrics_nn.train_precision(i),metrics_nn.train_recall(i),metrics_nn.train_f1(i)] = F1Scores(train_y_cv,train_pred_y3);
    %perfomance on test data
    test_pred_y1 = predict(model1, test_X_cv);
    test_pred_y2 = str2double(predict(model2, test_X_cv));
    test_pred_y3 = double(classify(model3, test_X_cv))-1;
    
    [~,~,~,metrics_svm.test_AUC(i)] = perfcurve(test_y_cv,test_pred_y1,1);
    [~,~,~,metrics_rf.test_AUC(i)] = perfcurve(test_y_cv,test_pred_y2,1);
    [~,~,~,metrics_nn.test_AUC(i)] = perfcurve(test_y_cv,test_pred_y3,1);
    [metrics_svm.test_precision(i),metrics_svm.test_recall(i),metrics_svm.test_f1(i)] = F1Scores(test_y_cv,test_pred_y1);
    [metrics_rf.test_precision(i),metrics_rf.test_recall(i),metrics_rf.test_f1(i)] = F1Scores(test_y_cv,test_pred_y2);
    [metrics_nn.test_precision(i),metrics_nn.test_recall(i),metrics_nn.test_f1(i)] = F1Scores(test_y_cv,test_pred_y3);
end


res_train_svm = [mean(metrics_svm.train_AUC);mean(metrics_svm.train_f1);mean(metrics_svm.train_precision);mean(metrics_svm.train_recall)];
res_test_svm = [mean(metrics_svm.test_AUC);mean(metrics_svm.test_f1);mean(metrics_svm.test_precision);mean(metrics_svm.test_recall)];
res_train_rf = [mean(metrics_rf.train_AUC);mean(metrics_rf.train_f1);mean(metrics_rf.train_precision);mean(metrics_rf.train_recall)];
res_test_rf = [mean(metrics_rf.test_AUC);mean(metrics_rf.test_f1);mean(metrics_rf.test_precision);mean(metrics_rf.test_recall)];
res_train_nn = [mean(metrics_nn.train_AUC);mean(metrics_nn.train_f1);mean(metrics_nn.train_precision);mean(metrics_nn.train_recall)];
res_test_nn = [mean(metrics_nn.test_AUC);mean(metrics_nn.test_f1);mean(metrics_nn.test_precision);mean(metrics_nn.test_recall)];

metrics=["AUC";"F1";"Precision";"Recall"];
results = table(metrics,res_train_svm,res_test_svm,res_train_rf,res_test_rf,res_train_nn, res_test_nn);
disp(results)

%% Grid search for best models
cvp_val = cvpartition(y, 'Holdout', 0.2);
training_val = cvp_val.training;
validation = cvp_val.test;
train_X = X(training_val,:); 
train_y = y(training_val);
val_X = X(validation,:); 
val_y = y(validation);

%% Grid search for Random Forest
nTreesRange = [10, 20, 50, 70, 100]; 
MinLeafSizeRange = [1, 5, 10, 20,50,100]; 

N = length(nTreesRange);
M = length(MinLeafSizeRange);
Result_auc = zeros(N,M);
%grid search
for ntree_idx = 1:N
    for minleaf_idx = 1:M
        model = TreeBagger(nTreesRange(ntree_idx), train_X, train_y,'MinLeafSize', MinLeafSizeRange(minleaf_idx), 'Method','classification');
        val_pred_y = str2double(predict(model, val_X));
        [~,~,~,Result_auc(ntree_idx,minleaf_idx)] = perfcurve(val_y,val_pred_y,1);
    end
end
%get bet parameters
[max_value,idx]=max(Result_auc(:));
[idx_t,idx_l]=ind2sub(size(Result_auc),idx);
ntree_best = nTreesRange(idx_t);
minleaf_best = MinLeafSizeRange(idx_l);

clc
sprintf('Best parameter ntree: %d',ntree_best)
sprintf('Best parameter minleaf: %d',minleaf_best)
sprintf('max AUC: %d',max_value)

% Best parameter ntree: 50
% Best parameter minleaf: 1
% max AUC: 9.6052
%% Grid search for SVM

%Hyper-parameters
kernel_scales = [0.01 0.1 1 5 10 50 100 500 1000];
Cs = [0.01 0.1 1 5 10 50 100 500 1000];

N = length(kernel_scales);
M = length(Cs);
Result_auc = zeros(N,M);
%grid search
for sigma_idx = 1:N
    for C_idx = 1:M
        model = fitcsvm(train_X, train_y, 'KernelFunction','RBF', 'BoxConstraint', Cs(C_idx), 'KernelScale', sqrt(2)*kernel_scales(sigma_idx));
        val_pred_y = predict(model, val_X);
        [~,~,~,Result_auc(sigma_idx,C_idx)] = perfcurve(val_y,val_pred_y,1);
    end
end
%get bet parameters
[max_value,idx]=max(Result_auc(:));
[idx_s,idx_c]=ind2sub(size(Result_auc),idx);
kernel_scale_best = kernel_scales(idx_s);
C_best = Cs(idx_c);

clc
disp(Result_auc)
sprintf('Best parameter C: %d',C_best)
sprintf('Best parameter sigma: %d',kernel_scale_best)

%'Best parameter C: 100'
%'Best parameter sigma: 10'
% max AUC 0.9162
%% Feature selection methods

%% PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(training_features);
PCA_idx = find(cumsum(explained)>95,1);
PCA_train = scoreTrain(:,1:PCA_idx);

%% CART Feature Selection
RandomForest = TreeBagger(200, PCA_train, labels, ...
    'Method','classification', Surrogate="on", OOBPredictorImportance="on");

CART = RandomForest.OOBPermutedPredictorDeltaError;
bar(CART)
title("Standard CART")
ylabel("Predictor Importance Estimates")
xlabel("Predictors")

%% apply CART
CART_idx = find(CART > 1.3);
CART_train = PCA_train(:,CART_idx);


% %%
% ecg_signal = ecgs{100};
% 
% %ecg_signal_norm = normalize(ecg_signal);
% %Apply bandpass filter
% % [b, a] = butter(1, [0.1 60]/(Fs/2)); % 1st order Butterworth filter
% % filtered_ecg = filter(b, a, ecg_signal);
% 
% Wp = [12 30]/(Fs/2); % passband
% Ws = [1 50]/(Fs/2); % stopband
% [n,Wn] = buttord(Wp,Ws,3,20); % order and natural frequency
% [b,a] = butter(n,Wn); % filter coefficients
% filtered_ecg = filtfilt(b, a, ecg_signal); % zero-phase filtering
% bandpass_filter_delay = n / 2;
% 
% differentiation_filter = [1, 2, 0, -2, -1] * (Fs/8);
% differentiated_ecg = conv(filtered_ecg, differentiation_filter, 'same');
% differentiation_delay = (length(differentiation_filter) - 1) / 2;
% 
% squared_ecg = differentiated_ecg .^ 2;
% N = round(0.1 * Fs); % 150 ms window length
% integration_window = ones(1, N) / N;
% integrated_ecg = conv(squared_ecg, integration_window, 'same');
% threshold = mean(integrated_ecg) + 0.1*sqrt(var(integrated_ecg));  % Example threshold
% integration_delay = (N - 1) / 2;
% 
% % Find Peaks (QRS Complexes)
% [~, locs] = findpeaks(integrated_ecg, 'MinPeakHeight', threshold,'MinPeakDistance',0.4*Fs);
% 
% 
% % Adjust QRS locations by finding nearest maxima in original ECG
% adjusted_locs = zeros(size(locs));
% 
% % Parameters
% search_radius = 20; % Define the radius around the detected peak to search for the true maxima
% 
% for i = 1:length(locs)
%     % Define search window around detected QRS location
%     window_start = max(1, locs(i) - search_radius);
%     window_end = min(length(ecg_signal), locs(i) + search_radius);
%     search_window = window_start:window_end;
% 
%     % Find the index of the maximum value in the search window
%     [~, max_idx] = max(ecg_signal(search_window));
% 
%     % Adjust location to the index of the maximum value
%     adjusted_locs(i) = window_start + max_idx - 1;
% end
% 
% % Initialize arrays to store the locations of P, Q, S, and T waves
% P_locs = [];
% Q_locs = [];
% S_locs = [];
% T_locs = [];
% 
% R_locs = adjusted_locs;
% 
% % Define search windows (in seconds)
% window_QS = 0.05; % 50 milliseconds around R peak for Q and S
% window_P = 0.2;   % 200 milliseconds before Q for P
% window_T = 0.2;   % 200 milliseconds after S for T
% 
% for i = 1:length(R_locs)
%     R = R_locs(i);
% 
%     % Search for Q and S
%     start_QS = max(1, R - round(window_QS * Fs));
%     end_QS = min(length(ecg_signal), R + round(window_QS * Fs));
%     [~, Q_idx] = min(ecg_signal(start_QS:R)); % Q wave is a minimum before R
%     [~, S_idx] = min(ecg_signal(R:end_QS));   % S wave is a minimum after R
%     Q_locs = [Q_locs, start_QS + Q_idx - 1];
%     S_locs = [S_locs, R + S_idx - 1];
% 
%     % Search for P wave
%     % Assuming P wave is before Q wave
%     start_P = max(1, Q_locs(end) - round(window_P * Fs));
%     [~, P_idx] = max(ecg_signal(start_P:Q_locs(end))); % P wave is a peak before Q
%     P_locs = [P_locs, start_P + P_idx - 1];
% 
%     % Search for T wave
%     % Assuming T wave is after S wave
%     end_T = min(length(ecg_signal), S_locs(end) + round(window_T * Fs));
%     [~, T_idx] = max(ecg_signal(S_locs(end):end_T)); % T wave is a peak after S
%     T_locs = [T_locs, S_locs(end) + T_idx - 1];
% end
% % Plot the results
% %part = 1000:3000;
% %subplot(5,1,1)
% plot(ecg_signal);
% hold on;
% %plot(intersect(part,adjusted_locs)-part(1), ecg_signal(intersect(part,adjusted_locs)), 'ro');
% % Plot QRS complexes
% plot(Q_locs, ecg_signal(Q_locs), 'x', 'MarkerEdgeColor', 'magenta');
% plot(R_locs, ecg_signal(R_locs), 'v', 'MarkerFaceColor', 'blue');
% plot(S_locs, ecg_signal(S_locs), 'o', 'MarkerFaceColor', 'red');
% 
% % Plot P peaks - replace 'pPeaks' with the actual indices for P peaks
% plot(P_locs, ecg_signal(P_locs), '^', 'MarkerFaceColor', 'green');
% 
% % Plot T peaks - replace 'tPeaks' with the actual indices for T peaks
% plot(T_locs, ecg_signal(T_locs), 's', 'MarkerFaceColor', 'black');
% 
% title('ECG Signal with Detected QRS Peaks');
% legend('ECG Signal', 'Q Peaks', 'R Peaks', 'S Peaks', 'P Peaks', 'T Peaks');

% hold off;
% subplot(5,1,2)
% plot(filtered_ecg(part));
% title('Filtered');
% subplot(5,1,3)
% plot(differentiated_ecg(part));
% title('Differentiated');
% subplot(5,1,4)
% plot(squared_ecg(part));
% title('Squared');
% subplot(5,1,5)
% plot(integrated_ecg(part));
% hold on;
% plot(intersect(part,locs)-part(1), integrated_ecg(intersect(part,locs)), 'ro');
% title('Integrated');
% hold off;

%% Neural Net build
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
    miniBatchSize =64;

    options = trainingOptions('adam', ...
        'MaxEpochs',5, ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'Verbose',false);


    NeuralNet = trainNetwork(X,y,layers,options);
    
end

%%
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

