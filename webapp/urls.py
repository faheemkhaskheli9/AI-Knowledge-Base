from django.urls import path

from webapp import views

urlpatterns = [
    path("", views.index, name="index"),
    path("t/<str:stem>/", views.topic, name="topic"),
    # Fixed repo-root entry points only; no route takes a file name from the URL.
    path("glossary/", views.page, {"name": "glossary.md"}, name="glossary"),
    path("learning-paths/", views.page, {"name": "learning-paths.md"}, name="learning_paths"),
    path("index/", views.page, {"name": "INDEX.md"}, name="index_md"),
    path("llms.txt", views.llms_txt, name="llms_txt"),
]
