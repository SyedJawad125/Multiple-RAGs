import logging

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from utils.permission_enums import *
from utils.reusable_functions import create_response, get_first_error
from utils.helpers import paginate_data
from utils.response_messages import SUCCESSFUL, NOT_FOUND, ID_NOT_PROVIDED

from utils.base_api import BaseView
from utils.decorator import permission_required

from .models import JobDescription, JobSkill, JobAnalysis
from .serializers import (
    JobDescriptionWriteSerializer,
    JobDescriptionListSerializer,
    JobDescriptionDetailSerializer,
    JobAnalysisSerializer,
    JobSkillSerializer,
)
from .filters import JobDescriptionFilter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  Helper — scope queryset to the current user's company
# ─────────────────────────────────────────────────────────
def _get_job_queryset(user):
    """
    Returns a scoped, non-deleted queryset for JobDescription.
    Superusers see everything; everyone else sees only their company.
    Falls back to the full unscoped qs if the user has no company assigned
    (mirrors Company view behaviour where multi-tenancy guard is inactive).
    """
    qs = JobDescription.objects.filter(deleted=False)
    if user.is_superuser:
        return qs
    if user.company_id:
        return qs.filter(company_id=user.company_id)
    # No company assigned — return empty so no data leaks across tenants.
    # Change to `return qs` only if you want to disable multi-tenancy here.
    return qs.none()


# ─────────────────────────────────────────────────────────
#  Main CRUD   →   /api/jobs/v1/job/
# ─────────────────────────────────────────────────────────
@extend_schema(tags=['jobs'])
class JobView(BaseView):
    permission_classes = (IsAuthenticated,)
    serializer_class   = JobDescriptionWriteSerializer
    list_serializer    = JobDescriptionListSerializer
    filterset_class    = JobDescriptionFilter

    # ── POST  ────────────────────────────────────────────
    @extend_schema(summary='Create a job description')
    @permission_required([CREATE_JOB])
    def post(self, request):
        try:
            serializer = JobDescriptionWriteSerializer(
                data=request.data, context={'request': request}
            )
            if serializer.is_valid():
                job = serializer.save(created_by=request.user)
                resp = JobDescriptionDetailSerializer(job, context={'request': request}).data
                return Response(create_response(SUCCESSFUL, resp), status=status.HTTP_201_CREATED)
            return Response(
                create_response(get_first_error(serializer.errors)),
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.exception("JobView.post error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ── GET  ─────────────────────────────────────────────
    @extend_schema(summary='List or retrieve job descriptions')
    @permission_required([READ_JOB])
    def get(self, request):
        try:
            qs = _get_job_queryset(request.user)

            # Apply filters — mirrors Company view: use .qs if valid, else keep qs as-is
            if self.filterset_class:
                filtered = self.filterset_class(request.GET, queryset=qs)
                qs = filtered.qs if filtered.is_valid() else qs

            # Ordering
            order    = request.query_params.get('order', 'desc')
            order_by = request.query_params.get('order_by', 'created_at')
            qs = qs.order_by(f"-{order_by}" if order == 'desc' else order_by)

            job_id = request.query_params.get('id')

            # ── Single record ──
            if job_id:
                instance = qs.filter(id=job_id).first()
                if not instance:
                    return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)
                serializer = JobDescriptionDetailSerializer(instance, context={'request': request})
                return Response(
                    create_response(SUCCESSFUL, serializer.data),
                    status=status.HTTP_200_OK,
                )

            # ── List ──
            data, count = paginate_data(qs, request)
            serialized  = JobDescriptionListSerializer(
                data, many=True, context={'request': request}
            ).data
            return Response(create_response(SUCCESSFUL, serialized, count), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobView.get error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ── PATCH  ───────────────────────────────────────────
    @extend_schema(summary='Partial update a job description')
    @permission_required([UPDATE_JOB])
    def patch(self, request):
        try:
            job_id = request.query_params.get('id')
            if not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            instance = _get_job_queryset(request.user).filter(id=job_id).first()
            if not instance:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            serializer = JobDescriptionWriteSerializer(
                instance, data=request.data, partial=True, context={'request': request}
            )
            if serializer.is_valid():
                job  = serializer.save(updated_by=request.user)
                resp = JobDescriptionDetailSerializer(job, context={'request': request}).data
                return Response(create_response(SUCCESSFUL, resp), status=status.HTTP_200_OK)
            return Response(
                create_response(get_first_error(serializer.errors)),
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.exception("JobView.patch error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ── DELETE  ──────────────────────────────────────────
    @extend_schema(summary='Soft-delete (archive) a job description')
    @permission_required([DELETE_JOB])
    def delete(self, request):
        try:
            job_id = request.query_params.get('id')
            if not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            instance = _get_job_queryset(request.user).filter(id=job_id).first()
            if not instance:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            instance.deleted    = True
            instance.updated_by = request.user
            instance.save()

            resp = JobDescriptionWriteSerializer(instance, context={'request': request}).data
            return Response(create_response(SUCCESSFUL, resp), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobView.delete error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────────────────
#  Lightweight list   →   /api/jobs/v1/job/list/
# ─────────────────────────────────────────────────────────
@extend_schema(tags=['jobs'])
class JobListView(BaseView):
    permission_classes = (IsAuthenticated,)
    serializer_class   = JobDescriptionListSerializer
    filterset_class    = JobDescriptionFilter

    @extend_schema(summary='Lightweight job list for dropdowns / cards')
    @permission_required([SHOW_JOB])
    def get(self, request):
        try:
            qs = _get_job_queryset(request.user)

            if self.filterset_class:
                filtered = self.filterset_class(request.GET, queryset=qs)
                qs = filtered.qs if filtered.is_valid() else qs

            order    = request.query_params.get('order', 'desc')
            order_by = request.query_params.get('order_by', 'created_at')
            qs = qs.order_by(f"-{order_by}" if order == 'desc' else order_by)

            data, count = paginate_data(qs, request)
            serialized  = JobDescriptionListSerializer(
                data, many=True, context={'request': request}
            ).data
            return Response(create_response(SUCCESSFUL, serialized, count), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobListView.get error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────────────────
#  Toggle status   →   /api/jobs/v1/job/toggle/
# ─────────────────────────────────────────────────────────
@extend_schema(tags=['jobs'])
class JobToggleView(BaseView):
    permission_classes = (IsAuthenticated,)

    @extend_schema(summary='Toggle job status')
    @permission_required([UPDATE_JOB])
    def patch(self, request):
        try:
            job_id = request.query_params.get('id')
            if not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            instance = _get_job_queryset(request.user).filter(id=job_id).first()
            if not instance:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            new_status = request.data.get('status')
            if not new_status:
                return Response(
                    create_response('status field is required.'),
                    status=status.HTTP_400_BAD_REQUEST,
                )

            instance.status     = new_status
            instance.updated_by = request.user
            instance.save(update_fields=['status', 'updated_by'])

            resp = JobDescriptionListSerializer(instance, context={'request': request}).data
            return Response(create_response(SUCCESSFUL, resp), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobToggleView.patch error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────────────────
#  AI Analysis   →   /api/jobs/v1/job/analyze/
# ─────────────────────────────────────────────────────────
@extend_schema(tags=['jobs'])
class JobAnalyzeView(BaseView):
    permission_classes = (IsAuthenticated,)

    @extend_schema(summary='Trigger AI analysis for a job description')
    @permission_required([UPDATE_JOB])
    def post(self, request):
        try:
            job_id = request.query_params.get('id')
            if not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            instance = _get_job_queryset(request.user).filter(id=job_id).first()
            if not instance:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            # ── Trigger your AI analysis task here ──
            # e.g. analyze_job_task.delay(str(instance.id))

            analysis = getattr(instance, 'analysis', None)
            resp     = JobAnalysisSerializer(analysis).data if analysis else {}
            return Response(create_response(SUCCESSFUL, resp), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobAnalyzeView.post error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────────────────
#  Stats   →   /api/jobs/v1/job/stats/
# ─────────────────────────────────────────────────────────
@extend_schema(tags=['jobs'])
class JobStatsView(BaseView):
    permission_classes = (IsAuthenticated,)

    @extend_schema(summary='Company job statistics')
    @permission_required([READ_JOB])
    def get(self, request):
        try:
            qs = _get_job_queryset(request.user)

            from django.db.models import Count, Q
            stats = qs.aggregate(
                total         = Count('id'),
                active        = Count('id', filter=Q(status='active')),
                draft         = Count('id', filter=Q(status='draft')),
                closed        = Count('id', filter=Q(status='closed')),
                with_analysis = Count('id', filter=Q(analysis__isnull=False)),
            )
            return Response(create_response(SUCCESSFUL, stats), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobStatsView.get error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ─────────────────────────────────────────────────────────
#  Job Skills   →   /api/jobs/v1/job/skills/
# ─────────────────────────────────────────────────────────
@extend_schema(tags=['jobs'])
class JobSkillView(BaseView):
    permission_classes = (IsAuthenticated,)

    # ── GET all skills for a job ──
    @extend_schema(summary='List skills for a job')
    @permission_required([READ_JOB])
    def get(self, request):
        try:
            job_id = request.query_params.get('job_id')
            if not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            job = _get_job_queryset(request.user).filter(id=job_id).first()
            if not job:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            skills     = job.skills.filter(deleted=False)
            serialized = JobSkillSerializer(skills, many=True).data
            return Response(
                create_response(SUCCESSFUL, serialized, len(serialized)),
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.exception("JobSkillView.get error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ── POST add a skill ──
    @extend_schema(summary='Add a skill to a job')
    @permission_required([UPDATE_JOB])
    def post(self, request):
        try:
            job_id = request.query_params.get('job_id')
            if not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            job = _get_job_queryset(request.user).filter(id=job_id).first()
            if not job:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            serializer = JobSkillSerializer(data=request.data)
            if serializer.is_valid():
                skill = serializer.save(job=job)
                return Response(
                    create_response(SUCCESSFUL, JobSkillSerializer(skill).data),
                    status=status.HTTP_201_CREATED,
                )
            return Response(
                create_response(get_first_error(serializer.errors)),
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.exception("JobSkillView.post error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ── PATCH update a skill ──
    @extend_schema(summary='Update a job skill')
    @permission_required([UPDATE_JOB])
    def patch(self, request):
        try:
            skill_id = request.query_params.get('id')
            job_id   = request.query_params.get('job_id')
            if not skill_id or not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            job = _get_job_queryset(request.user).filter(id=job_id).first()
            if not job:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            skill = job.skills.filter(deleted=False, id=skill_id).first()
            if not skill:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            serializer = JobSkillSerializer(skill, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(create_response(SUCCESSFUL, serializer.data), status=status.HTTP_200_OK)
            return Response(
                create_response(get_first_error(serializer.errors)),
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.exception("JobSkillView.patch error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ── DELETE soft-delete a skill ──
    @extend_schema(summary='Delete a job skill')
    @permission_required([DELETE_JOB])
    def delete(self, request):
        try:
            skill_id = request.query_params.get('id')
            job_id   = request.query_params.get('job_id')
            if not skill_id or not job_id:
                return Response(create_response(ID_NOT_PROVIDED), status=status.HTTP_400_BAD_REQUEST)

            job = _get_job_queryset(request.user).filter(id=job_id).first()
            if not job:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            skill = job.skills.filter(deleted=False, id=skill_id).first()
            if not skill:
                return Response(create_response(NOT_FOUND), status=status.HTTP_404_NOT_FOUND)

            skill.deleted = True
            skill.save(update_fields=['deleted'])
            return Response(create_response(SUCCESSFUL, {'id': str(skill_id)}), status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("JobSkillView.delete error: %s", e)
            return Response(create_response(str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)