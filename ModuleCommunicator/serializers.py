from django.forms import widgets
from rest_framework import serializers
from ModuleCommunicator.models import *


class ResultSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ResultModel
        fields = ('cls_result', 'region_result', 'seg_image', 'seg_image_th', 'result_image_bin', 'result_image_path')
        read_only_fields = ('cls_result', 'region_result', 'seg_image', 'seg_image_th', 'result_image_bin', 'result_image_path')


class ImageSerializer(serializers.HyperlinkedModelSerializer):
    results = ResultSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = ('image', 'modules', 'token', 'uploaded_date', 'updated_date',
                  'image_width', 'image_height', 'patch_size', 'results',
                  'region_threshold',
                  'region_connectivity',
                  'region_noise_filter',
                  'severity_threshold')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'image_width', 'image_height', 'results')
