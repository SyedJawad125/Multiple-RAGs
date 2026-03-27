import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")

# ← Explicitly list all apps so core/tasks.py is discovered
app.autodiscover_tasks([
    'AI_Resume_Screening_App.users',
    'AI_Resume_Screening_App.myapp',
    'AI_Resume_Screening_App.notification',
    'AI_Resume_Screening_App.jobs',
    'AI_Resume_Screening_App.resumes',
    'AI_Resume_Screening_App.screening',
    # 'core',           # ← This discovers core/tasks.py
])