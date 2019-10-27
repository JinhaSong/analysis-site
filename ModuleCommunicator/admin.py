# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin

from .models import ImageModel, ResultModel

admin.site.register(ImageModel)
admin.site.register(ResultModel)

# Register your models here.
