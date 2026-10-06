from django.urls import path

from webapp import views

urlpatterns = [
    path("", views.index, name="index"),
    path("t/<str:stem>/", views.topic, name="topic"),
]
