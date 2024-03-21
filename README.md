**Summary**

The Centers for Disease Control and Prevention (CDC) reports that around 2% of individuals under the age of 65 are affected by atrial fibrillation (FA), whereas the condition is present in about 9% of those aged 65 and above. This work is focused on detecting FA in electrocardiogram (ECG) signals through Machine Learning. A total of 58 features were extracted from each ECG signals, including temporal, morphological, and wavelet features (Appendix A). The wavelets (db6, sym4, and bior3.1) and butter filters were used for feature extraction, primarily involving statistical measures like mean, max, standard deviation, skewness, kurtosis, and entropy [1]. The feature set was selected for its ability to capture different aspects of the ECG signal, crucial for accurate detection of atrial fibrillation [2].

The Random Forest, SVM, and 4-layer Neural Net models were employed for during the process, also KNN and Naïve Bayes were tested. These algorithms were chosen and compared due to their good performance in handling diverse and big datasets and their ability to reduce variance and overfitting, thus enhancing the model's predictive power. The model's validation involved testing Random Forest on the offset validation dataset, yielding AUC score of 99.2% and F-1 score of 99.33%, with high scores in precision and recall. This high level of accuracy demonstrates the model's effectiveness in atrial fibrillation detection.

Visualizations included plots of the signals, which provided insights into the data processing process and performance. Supplementary information regarding the development of a real-time atrial fibrillation detection model gives comprehensive look on honest work done to achieve such high results.

**Literature**

The electrocardiogram (ECG) is a vital tool in diagnosing and managing heart conditions, particularly in differentiating between normal sinus rhythm (NSR) and atrial fibrillation (AF). Essential aspects of this differentiation process include the use of filters, feature extraction methods, and feature selection techniques. In NSR, the ECG typically shows a consistent heart rate with regular P waves preceding each QRS complex, indicating organized atrial depolarization. On the other hand, AF is characterized by rapid and irregular atrial rates, leading to an irregularly irregular rhythm without distinct P waves [3].

**Data processing**

Let's take a look at first 3 signals from each class we have in the dataset.

![](/figures/Picture1.png)

As we can see data is extremely complex and from the first look has no apparent differences between classes. Also, size of signals varies. Moreover, dataset is highly imbalanced with only 13.67% of the data accounts for positive class (FA).

I started with applying filters and wavelets we used at our homework-s and extracting statistical features from them. This gave me the results given below. Here and everywhere after res – results, svm – Support Vector Machine, rf – Random Forest, knn – K Nearest Neighbors, cnb – Naïve Bayes, nn – mentioned Neural Net.

![](/figures/Picture2.png)

Table 1. Results of cross validation on HW feature extraction techniques. Top half performance on test data, second half is on training data.

Here we can see that models learn something, but still models do not generalize no matter how little variations of feature extraction I tried, AUC and F-1 on test data were stuck around 70%.

So, after 2 days of pure coding and reading literature I applied Pan-Tompkin's method on detecting QRS complex [1]. I implemented everything and managed to detect R peaks from P-QRS-T complex in ECG. Below you can see each step of the method and detected R peaks highlighted with red circles.

![](/figures/Picture3.png)

To this point I left only svm, rf and nn in my models as they consistently performed better than knn and cnb. Below you can see results of my 3 models' performance on data without Features generated with R peaks (top half) and with them (bottom half). For more information on features please see Appendix A. Significant increase and my happy tears, my work was rewarded.

![](/figures/Picture4.png)

Now it is a technical work to detect rest of the P-QRS-T complex and brute force best hyperparameters for best model. So, below you can see part of one of ECG signals with all of the P-QRS-T complex detected correctly.

![](/figures/Picture5.png)

Also, here are models' performance on test data after calculating statistics on the on all parts of -QRS-T complex.

![](/figures/Picture6.png)

Now after grid search of the best hyperparameters and incorporated into generate\_model function. There, input data is split in train and test, 3 models are fitted and compared, the best model on AUC (as almost anywhere stated as the most robust among those 4) is retrained on the whole dataset and returned. Evaluation on validation dataset gave almost perfect metrics.



**Important notes**

1. Please be informed that to launch the code, you will need Deep Learning Toolbox.
2. To evaluate NN instead of predict() function, classify() is needed and function double() after, because predict return nx2 array for NNs, and also it returns chars.
3. To evaluate RF after predict(), function str2double() is needed, as TreeBagger() returns strings.

References

[1] R. Sanghavi, F. Chheda, S. Kanchan and S. Kadge, "Detection Of Atrial Fibrillation in Electrocardiogram Signals using Machine Learning," 2021 2nd Global Conference for Advancement in Technology (GCAT), Bangalore, India, 2021, pp. 1-6, doi: 10.1109/GCAT52182.2021.9587664.

[2] Noseworthy, P., Attia, Z., Behnken, E., Giblon, R., Bews, K., Liu, S., ... & Yao, X. (2022). Artificial intelligence-guided screening for atrial fibrillation using electrocardiogram during sinus rhythm: a prospective non-randomised interventional trial. _The Lancet, 400_, 1206-1212. [Artificial intelligence-guided screening for atrial fibrillation using electrocardiogram during sinus rhythm](https://www.semanticscholar.org/paper/c30d178b46fff8f33eb85b5c96a644502c214424)

[3] Han, D., Bashar, S., Lázaro, J., Mohagheghian, F., Peitzsch, A., Nishita, N., ... & Chon, K. (2022). A Real-Time PPG Peak Detection Method for Accurate Determination of Heart Rate during Sinus Rhythm and Cardiac Arrhythmia. _Biosensors, 12_. [A Real-Time PPG Peak Detection Method for Accurate Determination of Heart Rate during Sinus Rhythm and Cardiac Arrhythmia](https://www.semanticscholar.org/paper/9507664e174765bffd2af9f581b18e869084dc12)

[4] Eckardt, L., Sehner, S., Suling, A., Borof, K., Breithardt, G., Crijns, H., ... & Kirchhof, P. (2022). Attaining sinus rhythm mediates improved outcome with early rhythm control therapy of atrial fibrillation: the EAST-AFNET 4 trial. _European Heart Journal, 43_, 4127-4144. [Attaining sinus rhythm mediates improved outcome with early rhythm control therapy of atrial fibrillation: the EAST-AFNET 4 trial](https://www.semanticscholar.org/paper/8d4115f342213bbbf42403e40865d25208c99a5a)

[5] Soulat-Dufour, L., Lang, S., Addetia, K., Ederhy, S., Adavane-Scheublé, S., Chauvet-Droit, M., ... & Cohen, A. (2022). Restoring Sinus Rhythm Reverses Cardiac Remodeling and Reduces Valvular Regurgitation in Patients With Atrial Fibrillation. _Journal of the American College of Cardiology, 79 10_, 951-961. [Restoring Sinus Rhythm Reverses Cardiac Remodeling and Reduces Valvular Regurgitation in Patients With Atrial Fibrillation] (https://pubmed.ncbi.nlm.nih.gov/35272799/#:~:text=With%20Atrial%20Fibrillation-,Restoring%20Sinus%20Rhythm%20Reverses%20Cardiac%20Remodeling%20and%20Reduces%20Valvular%20Regurgitation,J%20Am%20Coll%20Cardiol.)

