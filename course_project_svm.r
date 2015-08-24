library(caret)
library(doParallel)

# load training data
labeledData = read.csv("pml-training.csv")

# remove unwanted features
labeledData = labeledData[,!(names(labeledData) %in% c("X", "user_name", "cvtd_timestamp"))]

# remove strange values
labeledData[labeledData == "#DIV/0!"] = NA

# store the target feature
classe = labeledData$classe

# remove all near-zero-features
nzv = nearZeroVar(labeledData)
nonZeroVarData = labeledData[,-nzv]

# remove a classe feature (the last column)
nonZeroVarData = nonZeroVarData[,-ncol(nonZeroVarData)]

# convert the data frame into a matrix
labeledMatrix = as.matrix(nonZeroVarData)

# split the training data into train a test set
# 60% train set
# 20% test set
# 20% validation set (4-fold cv)
set.seed(42)
trainIndices = createDataPartition(classe, p = .8, list=FALSE)
trainSet = labeledMatrix[trainIndices,]
testSet = labeledMatrix[-trainIndices,]

trainTarget = classe[trainIndices]
testTarget = classe[-trainIndices]

# center, scale and impute missing values
preProc = preProcess(trainSet, method=c("center", "scale", "knnImpute"))
trainImputed = predict(preProc, trainSet)
testImputed = predict(preProc, testSet)

dim(trainImputed)
dim(testImputed)

# register multiple workers
cluster = makeCluster(detectCores())
registerDoParallel(cluster)

# fit a model on the train set
trainControl = trainControl(method="cv", number=4)
svmGrid = expand.grid(C=c(0.001, 0.01, 0.1, 1.0, 10.0, 100.0))
model = train(x=trainImputed, y=trainTarget, trControl=trainControl, tuneGrid=svmGrid, method="svmRadialCost")

# predict on test set
predictions = predict(model, newdata=testImputed)
xtab = table(predictions, testTarget)
confMatrix = confusionMatrix(xtab)

# print model and confusion matrix
model
confMatrix

# finally predict the outcome for unlabeled data
unlabeledData = read.csv("pml-testing.csv")
unlabeledData = unlabeledData[,!(names(unlabeledData) %in% c("X", "user_name", "cvtd_timestamp", "problem_id"))]
unlabeledData[unlabeledData == "#DIV/0!"] = NA
unlabeledData = unlabeledData[,-nzv]
unlabeledMatrix = as.matrix(unlabeledData)
unlabeledImputed = predict(preProc, newdata=unlabeledMatrix)
unlabeledPredictions = predict(model, newdata=unlabeledImputed)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(unlabeledPredictions)