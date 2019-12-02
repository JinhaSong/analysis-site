# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
from rest_framework import exceptions
import requests


class ModuleElementModel(models.Model):
    name = models.TextField(unique=True)
    url = models.TextField(unique=True)
    content = models.TextField(blank=True)
    status = models.BooleanField(default=True)

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name

    def save(self, *args, **kwargs):
        super(ModuleElementModel, self).save(*args, **kwargs)

        self.group.update_or_create(name=self.name, content=self.content)
        super(ModuleElementModel, self).save()


class ModuleGroupModel(models.Model):
    name = models.TextField(unique=True)
    elements = models.ManyToManyField(ModuleElementModel, related_name='group')
    content = models.TextField(blank=True)

    def __str__(self):
        return self.name

    def __unicode__(self):
        return self.name
