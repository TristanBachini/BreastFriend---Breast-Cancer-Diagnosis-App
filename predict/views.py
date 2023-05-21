from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,accuracy_score,f1_score,roc_auc_score
import lime.lime_tabular
import base64
from io import BytesIO
import random


def image_to_base64(image):
    buff = BytesIO()
    image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    img_str = img_str.decode("utf-8")  # convert to str and cut b'' chars
    buff.close()
    return img_str

def Convert(tup, di):
    for a, b in tup:
        a = int(a)
        di.setdefault(a,[]).append(b)
    return di



# Create your views here.
def homepage(request):
    from django.core.cache import cache
    from django.contrib.sessions.backends.db import SessionStore
    cache.clear()
    random.seed(123)
    np.random.seed(123)
    session = SessionStore(session_key=request.session.session_key)
    session.flush() 
    return render(request, 'predict/homepage.html')

def predict_page(request):
    if(request.method == 'POST'):
        datapoint = np.array([request.POST.get('radiusm'),request.POST.get('texturem'),request.POST.get('perimeterm'),request.POST.get('aream'),request.POST.get('smoothnessm'),request.POST.get('compactnessm'),request.POST.get('concavitym'),request.POST.get('concavepointsm'),request.POST.get('symmetrym'),request.POST.get('fractaldimensionm'),
                              request.POST.get('radiusse'),request.POST.get('texturese'),request.POST.get('perimeterse'),request.POST.get('arease'),request.POST.get('smoothnessse'),request.POST.get('compactnessse'),request.POST.get('concavityse'),request.POST.get('concavepointsse'),request.POST.get('symmetryse'),request.POST.get('fractaldimensionse'),
                              request.POST.get('radiusl'),request.POST.get('texturel'),request.POST.get('perimeterl'),request.POST.get('areal'),request.POST.get('smoothnessl'),request.POST.get('compactnessl'),request.POST.get('concavityl'),request.POST.get('concavepointsl'),request.POST.get('symmetryl'),request.POST.get('fractaldimensionl'),])
        dataset = pd.read_csv("predict/data/data.csv")
        dataset = pd.get_dummies(data = dataset, drop_first = True)
        dataset = dataset.drop(columns = 'Unnamed: 32')

        df = pd.DataFrame(dataset)

        X = dataset.iloc[:,1:-1].values
        y = dataset.iloc[:,-1].values

        X_scaled = X

        X_imbalanced = np.vstack((X_scaled[y == 1], X_scaled[y == 0]))
        y_imbalanced = np.hstack((y[y == 1], y[y == 0]))



        from sklearn.utils import resample
        #
        # Create oversampled training data set for minority class
        #
        X_oversampled, y_oversampled = resample(X_imbalanced[y_imbalanced == 1],
                                                y_imbalanced[y_imbalanced == 1],
                                                replace=True,
                                                n_samples=X_imbalanced[y_imbalanced == 0].shape[0],
                                                random_state=123)
        #
        # Append the oversampled minority class to training data and related labels
        #

        X_balanced = np.vstack((X_scaled[y == 0], X_oversampled))
        y_balanced = np.hstack((y[y == 0], y_oversampled))

        x_train,x_test,y_train,y_test = train_test_split(X_balanced,y_balanced, test_size = 0.25, random_state = 45)

        rfc = RandomForestClassifier()
        rfc.fit(x_train,y_train)
        y_pred = rfc.predict(x_test)

        acc = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        auc = roc_auc_score(y_test, y_pred)
        results = pd.DataFrame([['RandomForestClassifier', acc, f1, precision, auc]],
                            columns = ['Model','Accuracy', 'F1', ' Precision', 'AUC'])

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(x_train,
                                                        feature_names=dataset.columns,
                                                        verbose=True, mode='classification')

        ttts = datapoint.astype(float)
        dataset = dataset.drop(['id','diagnosis_M'],axis=1)
        newerdf=pd.DataFrame(ttts.reshape(1,-1), columns=dataset.columns)

        pred = rfc.predict(newerdf)
        accuracy = acc*100
        #pred is the value predicted !
        i = 25
#        x_test = np.append(x_test,datapoint)



        # Calling the explain_instance method by passing in the:
        #    1) ith test vector
        #    2) prediction function used by our prediction model('reg' in this case)
        #    3) the top features which we want to see, denoted by k


        
        
 
        # Number denoting the top features
        k = 10
        # Calling the explain_instance method by passing in the:
        #    1) ith test vector
        #    2) prediction function used by our prediction model('reg' in this case)
        #    3) the top features which we want to see, denoted by k
#        exp_lime = explainer_lime.explain_instance(
#            x_test[i], rfc.predict_proba, num_features=k)

        # Retrieve the scaling parameters

        
        #exp_lime = explainer_lime.explain_instance(
        #    transformed[0], rfc.predict_proba, num_features=k)
        exp_lime = explainer_lime.explain_instance(
            ttts, rfc.predict_proba, num_features=k)
        i = 0
        lime_model = exp_lime.local_exp
        dictionary = {}
        for _,value in lime_model.items():
            dictionary = Convert(value,dictionary)

        columns = list(dataset.columns)

        feat_imp =  {}
        for i in range(0,len(columns)):
            for key in dictionary:
                if i == key:
                    value = round(dictionary[i][0]*100)
                    feat_imp.update({columns[i-1]:value})
    
        results = exp_lime.as_html(labels=None, predict_proba=True, show_predicted_value=True)
       
        plot = exp_lime.as_pyplot_figure()

    


        
        
        
        import io
        from PIL import Image
        import matplotlib.pyplot as plt
        from django.core.cache import cache
        from django.http import HttpResponse
#        plt.rcParams["figure.figsize"] = [100, 50]
#        plt.rcParams["figure.autolayout"] = False
#        Image.MAX_IMAGE_PIXELS = None

        plt.figure(plot)
        plt.xlabel('', fontsize=18)
        plt.ylabel('', fontsize=16)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png',bbox_inches = 'tight')   
        im = Image.open(img_buf)



        image64 = image_to_base64(im)

        

        response = HttpResponse(content_type='image/png')
        response.write(img_buf.getvalue())

        if(pred[0]==0):
            pred = "Negative!"
        else:
            pred = "Positive!"

        data = {'accuracy':accuracy,'results':results,'image64':image64,'pred':pred,'feat_imp':feat_imp,'response':response}
#        data = {'accuracy':accuracy,'results':results,'pred':pred,'feat_imp':feat_imp}
        cache.clear()
        return render(request, 'predict/results.html', data)


    return render(request, 'predict/predict-page.html')
