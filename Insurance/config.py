import pymongo
import pandas as pd
import numpy as np
import os 
import sys
import json
from dataclasses import dataclass

@dataclass
class EnvironmentVariable:
    mongodb_url=os.getenv("MONGO_DB_URL")

env_var=EnvironmentVariable()
mongo_client=pymongo.MongoClient(env_var.mongodb_url)
TARGET_COLUMN= "charges"
print(env_var.mongodb_url)