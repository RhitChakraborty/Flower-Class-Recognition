# Flower-Class-Recognition
# Hackathon Rank 13

To recognize the right flower you will be using 6 different attributes to classify them into the right set of classes(0-7). Using computer vision to do such recognition has reached state-of-the-art. Collecting Image data needs lots of human labor to annotate the images with the labels/bounding-boxes for detection/segmentation based tasks. Hence, some generic attribute which can be collected easily from various Area/Locality/Region were captured for over various species of flowers. Here, we are asked to use classical machine learning classification techniques to come up with a machine learning model that can generalize well on the unseen data provided explanatory attributes about the flower species instead of a picture.      

## Dataset Description:

Train.csv - 12666 rows x 7 columns (includes Class as target column)
Test.csv - 29555 rows x 6 columns

## Attributes Description:

Area_Code - Generic Area code, species were collected from      
Locality_Code - Locality code, species were collected from      
Region_Code - Region code, species were collected from      
Height - Height collected from lab data     
Diameter - Diameter collected from lab data     
Species - Species of the flower     
Class - Target Column (0-7) classes     

## Evaluation:
The submission will be evaluated using the Log Loss metric.


# Solution
Applying Feature Engineering is the key task to attain a good score.Feature Engineering like creating new features and scaling features is done on the basis of Exploratory Analysis. 
Different Classical models are tried on this dataset like knn,Decision trees,ensemble algoriths like Random Forest,ExtraTrees, XGBoost. But Gradient boosting model Catboost is used to fit and predict for the unseen test cases beacus eit gave the best result.
