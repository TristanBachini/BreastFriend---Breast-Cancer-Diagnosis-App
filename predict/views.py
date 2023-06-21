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
import joblib


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


def rename_features(feat_imp):
    for key in list(feat_imp):
        if key == 'radius_mean':
            feat_imp['mean of radius'] = feat_imp['radius_mean']
            del feat_imp['radius_mean']

        elif key == 'texture_mean':
            feat_imp['mean of texture'] = feat_imp['texture_mean']
            del feat_imp['texture_mean']
        
        elif key == 'perimeter_mean':
            feat_imp['mean of perimeter'] = feat_imp['perimeter_mean']
            del feat_imp['perimeter_mean']
        
        elif key == 'area_mean':
            feat_imp['mean of area'] = feat_imp['area_mean']
            del feat_imp['area_mean']
        
        elif key == 'smoothness_mean':
            feat_imp['mean of smoothness'] = feat_imp['smoothness_mean']
            del feat_imp['smoothness_mean']
        
        elif key == 'compactness_mean':
            feat_imp['mean of compactness'] = feat_imp['compactness_mean']
            del feat_imp['compactness_mean']
        
        elif key == 'concavity_mean':
            feat_imp['mean of concavity'] = feat_imp['concavity_mean']
            del feat_imp['concavity_mean']
        
        elif key == 'concave_mean':
            feat_imp['mean of concave'] = feat_imp['concave_mean']
            del feat_imp['concave_mean']
        
        elif key == 'symmetry_mean':
            feat_imp['mean of symmetry'] = feat_imp['symmetry_mean']
            del feat_imp['symmetry_mean']
        
        elif key == 'fractal_dimension_mean':
            feat_imp['mean of fractal dimension'] = feat_imp['fractal_dimension_mean']
            del feat_imp['fractal_dimension_mean']

        elif key == 'radius_se':
            feat_imp['standard error of radius'] = feat_imp['radius_se']
            del feat_imp['radius_se']
        
        elif key == 'texture_se':
            feat_imp['standard error of radius'] = feat_imp['radius_se']
            del feat_imp['radius_se']
        
        elif key == 'perimeter_se':
            feat_imp['standard error of perimeter'] = feat_imp['perimeter_se']
            del feat_imp['perimeter_se']
            
        elif key == 'area_se':
            feat_imp['standard error of area'] = feat_imp['area_se']
            del feat_imp['area_se']
        
        elif key == 'smoothness_se':
            feat_imp['standard error of smoothness'] = feat_imp['smoothness_se']
            del feat_imp['smoothness_se']

        elif key == 'compactness_se':
            feat_imp['standard error of compactness'] = feat_imp['compactness_se']
            del feat_imp['compactness_se']
        
        elif key == 'concavity_se':
            feat_imp['standard error of concavity'] = feat_imp['concavity_se']
            del feat_imp['concavity_se']
        
        elif key == 'concave_se':
            feat_imp['standard error of concave'] = feat_imp['concave_se']
            del feat_imp['concave_se']
        
        elif key == 'symmetry_se':
            feat_imp['standard error of symmetry'] = feat_imp['symmetry_se']
            del feat_imp['symmetry_se']
        
        elif key == 'fractal_dimension_se':
            feat_imp['standard error of fractal dimension'] = feat_imp['fractal_dimension_se']
            del feat_imp['fractal_dimension_se']
        
        elif key == 'radius_worst':
            feat_imp['largest of radius'] = feat_imp['radius_worst']
            del feat_imp['radius_worst']
        
        elif key == 'texture_worst':
            feat_imp['largest of texture'] = feat_imp['texture_worst']
            del feat_imp['texture_worst']
        
        elif key == 'perimeter_worst':
            feat_imp['largest of perimeter'] = feat_imp['perimeter_worst']
            del feat_imp['perimeter_worst']
        
        elif key == 'area_worst':
            feat_imp['largest of area'] = feat_imp['area_worst']
            del feat_imp['area_worst']
        
        elif key == 'smoothness_worst':
            feat_imp['largest of smoothness'] = feat_imp['smoothness_worst']
            del feat_imp['smoothness_worst']
        
        elif key == 'compactness_worst':
            feat_imp['largest of compactness'] = feat_imp['compactness_worst']
            del feat_imp['compactness_worst']
        
        elif key == 'concavity_worst':
            feat_imp['largest of concavity'] = feat_imp['concavity_worst']
            del feat_imp['concavity_worst']
        
        elif key == 'concave_worst':
            feat_imp['largest of concave'] = feat_imp['concave_worst']
            del feat_imp['concave_worst']
        
        elif key == 'symmetry_worst':
            feat_imp['largest of symmetry'] = feat_imp['symmetry_worst']
            del feat_imp['symmetry_worst']
        
        elif key == 'fractal_dimension_worst':
            feat_imp['largest of fractal dimension'] = feat_imp['fractal_dimension_worst']
            del feat_imp['fractal_dimension_worst']
        

    
    return feat_imp



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

        rfc = joblib.load("predict/data/random_forest.joblib")

        import dill
        with open('predict/data/explainer.pkl', 'rb') as f:
            explainer_lime = dill.load(f)

        ttts = datapoint.astype(float)
        dataset = dataset.drop(['id','diagnosis_M'],axis=1)
        newerdf=pd.DataFrame(ttts.reshape(1,-1), columns=dataset.columns)

        pred = rfc.predict(newerdf)
        accuracy = '98.18%'

        exp_lime = explainer_lime.explain_instance(
            ttts, rfc.predict_proba, num_features=10)
        
        # Save the explainer to a file
        
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
            pred = "Benign"
        else:
            pred = "Malignant"

        feat_imp = rename_features(feat_imp)


        data = {'accuracy':accuracy,'results':results,'image64':image64,'pred':pred,'feat_imp':feat_imp,'response':response}

        cache.clear()
        return render(request, 'predict/results.html', data)


    return render(request, 'predict/predict-page.html')


def about_us(request):
    return render(request, 'predict/about-us.html')