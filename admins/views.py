from django.shortcuts import render
from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistration
from users.forms import UserRegistrationForm
from django.db.models import Q
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    query = request.GET.get('q')
    if query:
        data = UserRegistration.objects.filter(
            Q(loginid__icontains=query) |
            Q(mobile__icontains=query) |
            Q(email__icontains=query)
        )
    else:
        data = UserRegistration.objects.all().order_by('-id')
        # Pagination
    paginator = Paginator(data, 5)  # Show 5 items per page
    page_number = request.GET.get('page')
    try:
        data = paginator.page(page_number)
    except PageNotAnInteger:
        data = paginator.page(1)
    except EmptyPage:
        data = paginator.page(paginator.num_pages)
    # Calculate item start
    start_serial = (data.number - 1) * paginator.per_page + 1
    return render(request, 'admins/viewregisterusers.html',
                  {'data': data, 'query': query, 'start_serial': start_serial})


from django.http import HttpResponseRedirect
from django.urls import reverse


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistration.objects.filter(id=id).update(status=status)
        current_page = request.GET.get('page')
        if not current_page:
            current_page = 1

        # Redirect back to the same page with the correct page number
        return HttpResponseRedirect(reverse('RegisterUsersView') + '?page=' + str(current_page))
        # data = UserRegistration.objects.all()
        # return render(request,'admins/viewregisterusers.html',{'data':data})


def DeleteUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistration.objects.filter(id=id).delete()
        current_page = request.GET.get('page')
        if not current_page:
            current_page = 1

        # Redirect back to the same page with the correct page number
        return HttpResponseRedirect(reverse('RegisterUsersView') + '?page=' + str(current_page))
        # data = UserRegistration.objects.all()
        # return render(request,'admins/viewregisterusers.html',{'data':data})


def BlockUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'waiting'
        print("PID = ", id, status)
        UserRegistration.objects.filter(id=id).update(status=status)
        data = UserRegistration.objects.all()
        current_page = request.GET.get('page')
        if not current_page:
            current_page = 1

        # Redirect back to the same page with the correct page number
        return HttpResponseRedirect(reverse('RegisterUsersView') + '?page=' + str(current_page))
        # return render(request, 'admins/viewregisterusers.html', {'data': data})
