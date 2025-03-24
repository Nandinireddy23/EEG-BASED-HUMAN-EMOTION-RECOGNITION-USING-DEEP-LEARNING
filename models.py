from django.db import models


# Create your models here.
class UserRegistration(models.Model):
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'



# models.py

from django.db import models
from django.utils import timezone
from django.contrib.auth import get_user_model

User = get_user_model()

class PasswordResetToken(models.Model):
    user = models.ForeignKey(UserRegistration, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Token for {self.user.loginid}"
