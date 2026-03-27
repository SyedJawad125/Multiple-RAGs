from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/user/', include('User_App.users.urls')),
    path('api/images/', include('User_App.images.urls')),

    path('api/jobs/',      include('AI_Resume_Screening_App.jobs.urls')),
    path('api/resumes/',   include('AI_Resume_Screening_App.resumes.urls')),
    path('api/screening/', include('AI_Resume_Screening_App.screening.urls')),
    path('api/rag/',       include('RagStack_App.ragstack.urls')),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
