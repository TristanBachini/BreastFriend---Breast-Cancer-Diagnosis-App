from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name="homepage"),
    path('predict/', views.predict_page, name="predict-page"),
    path('about-us/', views.about_us, name="about-us"),
]