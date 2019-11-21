from django.forms import widgets
from rest_framework import serializers
from ModuleCommunicator.models import *


class ResultSerializer(serializers.HyperlinkedModelSerializer):
    module_name = serializers.CharField(source='get_module_name', read_only=True)

    class Meta:
        model = ResultModel
        fields = ('module_name', 'module_result')
        read_only_fields = ('module_name', 'module_result')


class ImageSerializer(serializers.HyperlinkedModelSerializer):
    results = ResultSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = ('image', 'modules', 'token', 'uploaded_date', 'updated_date', 'image_width', 'image_height', 'patch_size', 'results',
                  'region_threshold',
                  'connected_component_threshold', 'region_connectivity', 'region_noise_filter', 'severity_threshold')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'image_width', 'image_height', 'results')
