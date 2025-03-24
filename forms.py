import re
from django import forms
from django.core.validators import RegexValidator, EmailValidator
from .models import UserRegistration

class UserRegistrationForm(forms.ModelForm):
    loginid = forms.CharField(widget=forms.TextInput(attrs={'pattern': '[a-zA-Z]+'}), required=True, max_length=100)
    password = forms.CharField(widget=forms.PasswordInput(), required=True, max_length=100)
    mobile = forms.CharField(widget=forms.TextInput(attrs={'pattern': '[56789][0-9]{9}'}), required=True, max_length=100)
    email = forms.CharField(widget=forms.TextInput(), required=True, max_length=100)
    city = forms.CharField(widget=forms.TextInput(attrs={'autocomplete': 'off', 'pattern': '[A-Za-z ]+', 'title': 'Enter Characters Only '}), required=True, max_length=100)
    state = forms.CharField(widget=forms.TextInput(attrs={'autocomplete': 'off', 'pattern': '[A-Za-z ]+', 'title': 'Enter Characters Only '}), required=True, max_length=100)
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting', max_length=100)

    class Meta():
        model = UserRegistration
        fields = '__all__'

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if not re.match(r'(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}', password):
            raise forms.ValidationError("Password must contain at least one number, one uppercase and lowercase letter, and at least 8 or more characters.")
        return password

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not re.match(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email):
            raise forms.ValidationError("Invalid email format.")
        return email


class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(label='Email')

class ResetPasswordForm(forms.Form):
    password = forms.CharField(label='New Password', widget=forms.PasswordInput)
    confirm_password = forms.CharField(label='Confirm New Password', widget=forms.PasswordInput)

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password:
            if password != confirm_password:
                raise forms.ValidationError("Passwords do not match.")
        return cleaned_data
