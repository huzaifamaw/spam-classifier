from django.db import models
import uuid
# Create your models here.
class History(models.Model):
    id = models.UUIDField(default=uuid.uuid4, unique=True,
                          primary_key=True, editable=False)
    text=models.TextField()
    model=models.TextField()
    result=models.TextField()


    def __str__(self):
        return self.text
    
