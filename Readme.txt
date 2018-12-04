Implementation of Supervised Multiclass Classification based on K-nn algorithm in python.

Data Input:
       -- News article dataset file (.mtx format)
       -- Labels file: News Articles are of types: [Business, Politics, Sport, Technology


Code Implementation and steps:

Step1) Accessing Input Files (Dataset, labels)

Step2) Spliting data set into Training and Test set
        -- Training Set : 70#
        -- Test Set : 30%

Step3) calculate Cosine Similarity

Step4) unweighted and weighted K-nn
        -- weighted K-nn:
            - voting for predicting labels is calculated based on weights
            - here inverse distance weighted method used

Step5) Calculate Accuracy
Calculated accuracy of more than 90%


Steps to run code:
Step1: Extract python file KnnClassifier.py and data files newsarticles.mtx and newsarticles.labels in python folder of system.
Step2: Right click on KnnClassifier.py file and choose ‘edit with Idle’ option or you can just open the file directly from python Idle
       If the file is extracted in some other loaction then in main function specify paths in mtx_file and labels variable
Step3: Run
