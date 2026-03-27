from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
# from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/user/', include('AI_Resume_Screening_App.users.urls')),
    path('api/images/', include('AI_Resume_Screening_App.images.urls')),

    path('api/jobs/',      include('AI_Resume_Screening_App.jobs.urls')),
    path('api/resumes/',   include('AI_Resume_Screening_App.resumes.urls')),
    path('api/screening/', include('AI_Resume_Screening_App.screening.urls')),
    

    # path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    # path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    # path("api/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
