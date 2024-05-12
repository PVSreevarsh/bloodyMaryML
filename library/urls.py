from django.urls import path
from .views import SMELabel

urlpatterns = [
    path('sme-label/', SMELabel.as_view(), name='SME-Label'),
]