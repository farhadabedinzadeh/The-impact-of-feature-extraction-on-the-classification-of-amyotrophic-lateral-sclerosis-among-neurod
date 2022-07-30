function [trainedClassifier, validationAccuracy,Sensitivity,specificity] = NaiveBayesian(trainingData)

inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_15;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% Gaussian is replaced with Normal when passing to the fitcnb function
distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};

if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'Kernel', 'Normal', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [1; 2]);
else
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [1; 2]);
end

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 14 columns because this model was trained using 14 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_15;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
% Result
confmat = confusionmat(response,validationPredictions); % where response is the last column in the dataset representing a class
TP = confmat(2, 2);
TN = confmat(1, 1);
FP = confmat(1, 2);
FN = confmat(2, 1);
Accuracy = (TP + TN) / (TP + TN + FP + FN)*100;
Sensitivity = TP / (FN + TP)*100;
specificity = TN / (TN + FP)*100;
z = FP / (FP+TN);
X = [0;Sensitivity;1];
Y = [0;z;1];
AUC = trapz(Y,X);  % This way is used for only binary classification
% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
validationAccuracy=validationAccuracy*100;

fprintf('.......... Naïve Bayesian Result .......... \n')

fprintf('Mean of Validation Accuracy is %.2f', validationAccuracy)
fprintf(' and Mean of Sensitivity is %.2f \n', Sensitivity)

end
