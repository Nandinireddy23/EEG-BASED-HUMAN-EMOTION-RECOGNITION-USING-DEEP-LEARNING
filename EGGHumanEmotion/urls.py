"""egg URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from . import views as mainviews
from users import views as userviews
from admins import views as admins

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", mainviews.index, name="index"),
    path("AdminLogin/", mainviews.AdminLogin, name="AdminLogin"),
    path("UserLogin/", mainviews.UserLogin, name="UserLogin"),
    path("UserRegister/", mainviews.UserRegister, name="UserRegister"),

    # Adminviews
    path("AdminLoginCheck/", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivaUsers/', admins.ActivaUsers, name='ActivaUsers'),
    path('DeleteUsers/', admins.DeleteUsers, name='DeleteUsers'),
    path('BlockUsers/', admins.BlockUsers, name='BlockUsers'),

    # users side urls
    path('signupfunction/', userviews.signup, name='signupfunction'),
    path('verify-otp/', userviews.verify_otp, name='verify_otp'),
    path('resend-otp/', userviews.resend_otp, name='resend_otp'),
    path('login', userviews.login, name='login'),
    path('forgot-password/', userviews.forgot_password, name='forgot_password'),
    path('reset-password/<str:token>/', userviews.reset_password, name='reset_password'),
    path('profile/', userviews.profile, name='profile'),
    path("StartEmotions/", userviews.StartEmotions, name='StartEmotions'),
    path("UserHome/", userviews.UserHome, name='UserHome'),
    path("Training/", userviews.Training, name="Training"),
    path('deapResults/', userviews.deapResults, name='deapResults'),
]
