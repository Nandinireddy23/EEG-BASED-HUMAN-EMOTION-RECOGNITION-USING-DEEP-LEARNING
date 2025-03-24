from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import check_password
from django.shortcuts import get_object_or_404
from django.shortcuts import render, redirect, HttpResponse
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from django.utils.crypto import get_random_string
from .forms import UserRegistrationForm, ForgotPasswordForm, ResetPasswordForm
from .models import UserRegistration, PasswordResetToken
from django.urls import reverse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from django.contrib.auth.hashers import make_password
from datetime import datetime


def signup(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            otp = get_random_string(6, '0123456789')  # Generate OTP
            request.session['otp'] = otp
            request.session['form_data'] = form.cleaned_data
            send_otp_email(
                form.cleaned_data['email'], otp, form.cleaned_data['loginid'])
            messages.info(
                request, 'An OTP has been sent to your email. Please verify to complete the signup.')
            return redirect('verify_otp')
    else:
        form = UserRegistrationForm()
    return render(request, 'signup.html', {'form': form})


def verify_otp(request):
    if request.method == 'POST':
        otp_entered = request.POST.get('otp')
        otp_generated = request.session.get('otp')
        if otp_entered == otp_generated:
            form_data = request.session.get('form_data')
            if form_data:
                user = UserRegistration.objects.create(
                    loginid=form_data['loginid'],
                    password=make_password(form_data['password']),
                    mobile=form_data['mobile'],
                    email=form_data['email'],
                    city=form_data['city'],
                    state=form_data['state'],
                    status='waiting'
                )
                del request.session['otp']
                del request.session['form_data']
                messages.success(
                    request, 'Your OTP verification is successful. Your account is created.')
                # return redirect('login')
                return render(request, 'some.html')
            else:
                messages.error(request, 'Form data not found.')
        else:
            messages.error(request, 'Invalid OTP. Please try again.')
    return render(request, 'verify_otp.html')


def send_otp_email(email, otp, loginid):
    subject = 'Verification OTP'
    message = f'''<html>
                  <body>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">Hello &nbsp;</span><b><span style="text-transform:uppercase;font-size: 16px;">{loginid}</span><b>,</p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">You are required to enter the following code to signup at "EEG-Based Human Emotion Recognition"</span></P>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">If you did not request this OTP, please ignore this email.</span></p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">Your OTP for account verification is:</span>\n 
                    \n<h1>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{otp}<b></h1></p>
                    
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">Thank you,</span></p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;"><b>Team Alex</span></b></p>
                  </body>
                  </html>'''
    smtp_port = 587  # Change port to 587
    smtp_server = settings.EMAIL_HOST
    sender_email = settings.EMAIL_HOST_USER
    password = settings.EMAIL_HOST_PASSWORD
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = 'Alex Corporation'
    msg['To'] = email
    msg['Subject'] = subject
    # Attach message
    msg.attach(MIMEText(message, 'html', 'utf-8'))
    # Send the message via the SMTP server
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, password)
            server.sendmail(sender_email, email, msg.as_string())
    except Exception as e:
        print("An error occurred:", e)


def resend_otp(request):
    otp = get_random_string(6, '0123456789')  # Generate new OTP
    request.session['otp'] = otp
    form_data = request.session.get('form_data')
    email = form_data.get('email')
    loginid = form_data.get('loginid')  # Retrieve loginid from form_data
    try:
        send_otp_email(email, otp, loginid)  # Pass loginid as an argument
        messages.info(request, 'A new OTP has been sent to your email.')
    except Exception as e:
        print("An error occurred while resending OTP:", e)
        messages.error(request, 'Failed to send OTP. Please try again later.')
    return redirect('verify_otp')


def login(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        password = request.POST.get('password')
        try:
            user = UserRegistration.objects.get(loginid=loginid)
            print("Login ID:", loginid)  # Print login ID to console
            print("Password:", password)  # Print password to console
            print("Status:", user.status)  # Print status to console
            if user.status == 'waiting':
                messages.success(
                    request, 'Your account is waiting for approval. Please confirm by admin')
                return render(request, 'login.html', {})
            elif check_password(password, user.password):
                request.session['id'] = user.id
                request.session['loginid'] = user.loginid
                request.session['email'] = user.email
                if 'last_login' in request.session:
                    last_login = request.session['last_login']
                else:
                    last_login = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                signin_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                request.session['signin_time'] = signin_time
                request.session['last_login'] = last_login
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Invalid password')
        except UserRegistration.DoesNotExist:
            messages.success(request, 'Invalid login id')
    return render(request, 'login.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def forgot_password(request):
    if request.method == 'POST':
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = UserRegistration.objects.get(email=email)
                token = get_random_string(32)

                PasswordResetToken.objects.create(user=user, token=token)
                reset_link = request.build_absolute_uri(
                    reverse('reset_password', kwargs={'token': token})
                )
                send_reset_password_email(email, reset_link, user.loginid)
                messages.info(
                    request, 'A password reset link has been sent to your email.')
                # Redirect to the login page after sending email
                return redirect('login')
            except UserRegistration.DoesNotExist:
                messages.error(
                    request, 'No user found with this email address.')
    else:
        form = ForgotPasswordForm()
    return render(request, 'forgot_password.html', {'form': form})


def reset_password(request, token):
    try:
        password_reset_token = PasswordResetToken.objects.get(token=token)
        print(password_reset_token)
        if request.method == 'POST':
            form = ResetPasswordForm(request.POST)
            if form.is_valid():
                user = password_reset_token.user
                user.password = make_password(form.cleaned_data['password'])
                user.save()
                password_reset_token.delete()
                messages.success(
                    request, 'Your password has been reset successfully.')
                return redirect('login')
        else:
            form = ResetPasswordForm()
        return render(request, 'reset_password.html', {'form': form})
    except PasswordResetToken.DoesNotExist:
        messages.error(request, 'Invalid or expired password reset link.')
        return redirect('login')


def send_reset_password_email(email, reset_link, loginid):
    subject = 'Password Reset'
    message = f'''<html>
                  <body>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">Hello, &nbsp;</span><b><span style="text-transform:uppercase;font-size: 16px;">{loginid}</span><b></p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">If you did not request this OTP, please ignore this email.</span></p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">Please click the following link to reset your password: {reset_link}</span></p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;">Thank you,</span></p>
                    <p><span style="font-family: Arial, sans-serif; font-size: 15px;"><b>Team Alex</span></b></p>
                  </body>
                  </html>'''
    msg = MIMEMultipart()
    msg['From'] = 'Alex Corporation <{}>'.format(settings.EMAIL_HOST_USER)
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'html', 'utf-8'))
    try:
        with smtplib.SMTP(settings.EMAIL_HOST, 587) as server:
            server.starttls()  # Start TLS encryption
            server.login(settings.EMAIL_HOST_USER,
                         settings.EMAIL_HOST_PASSWORD)
            server.sendmail(settings.EMAIL_HOST_USER, email, msg.as_string())
    except Exception as e:
        print("An error occurred:", e)


# @login_required
def profile(request):
    user_id = request.session.get('id')
    signin_time = request.session.get('signin_time')
    last_login = request.session.get('last_login')
    user_profile = get_object_or_404(UserRegistration, id=user_id)
    return render(request, 'users/profile.html',
                  {'user_profile': user_profile, 'signin_time': signin_time, 'last_login': last_login})


# @login_required
def StartEmotions(request):
    from .utility import EmotionRecognitions
    result_list = EmotionRecognitions.StartHumanEmotions()
    if result_list:
        from collections import Counter
        sort = Counter(result_list)
        emotions_counts = sort.most_common(7)
        filters_emo = emotions_counts[0]
        emotion = filters_emo[0]
        count = filters_emo[1]
        print(f'Emotions {emotion} and its count {count}')

    return render(request, 'users/EmotionPage.html', {"Emotions": result_list})


# def Training(request):
#     from .utility.StartTraining import InitializeTraining
#     obj = InitializeTraining()
#     obj.start_process()
#     return render(request, 'users/UserHome.html', {})



# def Training(request):
#     from .utility.StartTraining import InitializeTraining
#     obj = InitializeTraining()
#     obj.start_process()
#     return render(request, 'users/UserHome.html', {})
import os
import json
from django.conf import settings
from django.shortcuts import render
from .utility.StartTraining import InitializeTraining

def Training(request):
    # Path to the JSON file containing the results
    json_file_path = os.path.join(settings.MEDIA_ROOT, 'training_result.json')

    # Check if previous training results exist
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            training_results = json.load(json_file)

        # Check if more training is needed (example: extend to 50 epochs)
        if len(training_results.get("accuracy", [])) < 50:  # Adjust based on your desired total epochs
            obj = InitializeTraining()
            obj.start_process()  # Continue training

            # Reload updated results after additional training
            with open(json_file_path, 'r') as json_file:
                training_results = json.load(json_file)
        
        return render(request, 'users/training_results.html', {'training_results': training_results})

    # If no results exist, start fresh training
    obj = InitializeTraining()
    obj.start_process()

    # Load results after training completes
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            training_results = json.load(json_file)
    else:
        training_results = {}

    return render(request, 'users/training_results.html', {'training_results': training_results})




# def deapResults(request):
#     from .utility import deapModels
#     import pandas as pd
#     history = deapModels.buildDeapModel()
#     df = pd.DataFrame(history)
#     df = df.to_html(index=False)

#     return render(request, 'users/deapresult.html', {'history': df})

from django.shortcuts import render
import os
import json
import pandas as pd
from django.conf import settings
from .utility import deapModels

def deapResults(request):
    json_path = os.path.join(settings.MEDIA_ROOT, 'deap_results.json')
    plot_path_acc = os.path.join(settings.MEDIA_URL, 'accuracy_plot.png')
    plot_path_loss = os.path.join(settings.MEDIA_URL, 'loss_plot.png')

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            history = json.load(f)
        print("Loaded results from JSON.")
    else:
        history = deapModels.buildDeapModel()

    # Convert history dictionary into a list of dictionaries for template rendering
    epochs = len(history["acc"])
    history_list = [
        {
            "epoch": i + 1,
            "train_acc": history["acc"][i],
            "val_acc": history["val_acc"][i],
            "train_loss": history["loss"][i],
            "val_loss": history["val_loss"][i],
        }
        for i in range(epochs)
    ]

    return render(request, 'users/deapresult.html', {
        'history': history_list,  # Pass as a list for iteration
        'plot_acc': plot_path_acc,
        'plot_loss': plot_path_loss
    })


