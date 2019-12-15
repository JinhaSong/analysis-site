# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from django.contrib.postgres.fields import JSONField
from django.core.files.base import ContentFile
from rest_framework import exceptions
from AnalysisSite.config import DEBUG
from ModuleCommunicator.tasks import communicator
from ModuleCommunicator.utils import filename
from ModuleManager.models import *
from cv2 import cv2
import os, base64


class ImageModel(models.Model):
    image = models.ImageField(upload_to=filename.uploaded_date)
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    modules = models.TextField(blank=True)
    image_width = models.IntegerField(default=0)
    image_height = models.IntegerField(default=0)
    patch_size = models.IntegerField(default=256)
    region_threshold = models.IntegerField(default=0)
    region_connectivity = models.IntegerField(default=0)
    region_noise_filter = models.IntegerField(default=0)
    severity_threshold = models.IntegerField(default=239)


    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)

        module_set = self.get_module()
        module_result = list()

        for module in module_set.all():
            module_result.append(self.results.create(module=module))

        for result in module_result:
            result.get_result()
        image_size = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../media/' ,str(self.image))).shape

        self.image_height = image_size[0]
        self.image_width = image_size[1]
        super(ImageModel, self).save()

    # Get ModuleModel item from self.modules
    def get_module(self):
        if len(self.modules) == 0:
            return ModuleElementModel.objects.all()

        module_group_list = self.modules.split(',')
        module_set = None

        for module_group in module_group_list:
            try:
                modules_in_group = ModuleGroupModel.objects.get(name=module_group.strip())
            except:
                raise exceptions.ValidationError('Module not found. Please check and send again.')

            if module_set is None:
                module_set = modules_in_group.elements.all()
            else:
                module_set = module_set | modules_in_group.elements.all()

        return module_set.distinct()


class ResultModel(models.Model):
    image = models.ForeignKey(ImageModel, related_name='results', on_delete=models.CASCADE)
    module = models.ForeignKey(ModuleElementModel)
    cls_result = JSONField(null=True)
    seg_image = models.TextField()
    seg_image_th = models.TextField()
    region_result = JSONField(null=True)
    result_image_bin = models.TextField()
    result_image_path = models.ImageField()

    def save(self, *args, **kwargs):
        super(ResultModel, self).save(*args, **kwargs)
        self.set_task()
        super(ResultModel, self).save()

    # Celery Delay
    def set_task(self):
        self.task = None
        image_size = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../media/' ,str(self.image.image))).shape
        # try:
        if DEBUG:
            self.task = communicator(
                self.module.name,
                self.module.url,
                self.image.image.path,
                image_size[1],
                image_size[0],
                self.image.patch_size,
                self.image.region_threshold,
                self.image.region_connectivity,
                self.image.region_noise_filter,
                self.image.severity_threshold
            )
        else:
            self.task = communicator.delay(
                self.module.name,
                self.module.url,
                self.image.image.path,
                image_size[1],
                image_size[0],
                self.image.patch_size,
                self.image.region_threshold,
                self.image.region_connectivity,
                self.image.region_noise_filter,
                self.image.severity_threshold
            )
        # except:
        #     raise exceptions.ValidationError("Module Set Error. Please contact the administrator")

    # Celery Get
    def get_result(self):
        # try:
        if DEBUG:
            task = self.task
        else:
            task = self.task.get()

        if self.module.name == 'crackviewer':
            self.cls_result = task['cls_result']
            self.region_result = task['region_result']
            self.seg_image = task['seg_image']
            self.seg_image_th = task['seg_image_th']
            self.result_image_bin = task['result_image']
        elif self.module.name == 'bin':
            self.cls_result = ''
            self.region_result = task['region_result']
            self.seg_image = ''
            self.seg_image_th = ''
            self.result_image_bin = task['result_image']
        elif self.module.name == 'path':
            self.cls_result = ''
            self.region_result = task['region_result']
            self.seg_image = ''
            self.seg_image_th = ''
            result_path = os.path.join(str(self.image.image).split(".")[0] + "_result" + ".png")
            self.result_image_path = ContentFile(base64.b64decode(task['result_image']), name=result_path)

        # except:
        #     raise exceptions.ValidationError("Module Get Error. Please contact the administrator")
        super(ResultModel, self).save()

    def get_module_name(self):
        return self.module.name
