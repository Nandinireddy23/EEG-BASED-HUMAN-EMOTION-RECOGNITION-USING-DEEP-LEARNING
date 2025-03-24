from django.contrib import admin

# Register your models here.
from .models import UserRegistration

admin.site.register(UserRegistration)