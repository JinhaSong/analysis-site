from __future__ import print_function

import datetime
from AnalysisSite.celerys import app
from ModuleCommunicator import models


@app.task
def delete_old_database(days=0):
    date_now = datetime.date.today()
    date_delta = datetime.timedelta(days)
    date_point = date_now - date_delta

    old_database = models.ImageModel.objects.filter()#uploaded_date__lte=date_point)
    old_database_count = old_database.count()
    for each_database in old_database:  # Delete local image files
        each_database.image.delete()
    old_database.delete()

    print("====================")
    print(" Delete Old Image")
    print(" - Date Point: {0}".format(date_point))
    print("====================")

    return old_database_count
