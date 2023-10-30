from dotenv import load_dotenv
from datetime import datetime
import os

class DataSource:
    def __init__(this):
        load_dotenv()
        this.BODY_MEASUREMENT_PATH = os.getenv("BODY_MEASUREMENT_PATH")
        this.RENDER_RGB_PATH = os.getenv("RENDER_RGB_PATH")
        this.GENDER = os.getenv("GENDER")
        this.SUB_FOLDER = os.getenv("SUBFOLDER")
        this.OUTPUT = os.getenv("OUTPUT")
        this.MODEL_PATH = os.path.join(this.OUTPUT, "model")
        try:
            os.makedirs(this.OUTPUT)
            os.makedirs(this.MODEL_PATH)
        except:
            print("Output folder exists")

        if (not os.path.exists(this.getBodyMeasurementPath())):
            raise Exception(f"body measurement path {this.getBodyMeasurementPath()} does not exist")

        if (not os.path.exists(this.getFrontPath())):
            raise Exception(f"front path {this.getFrontPath()} does not exist")

        if (not os.path.exists(this.getSidePath())):
            raise Exception(f"side path {this.getSidePath()} does not exist")

    def getBodyMeasurementPath(this):
        pathBuilder = os.path.join(this.BODY_MEASUREMENT_PATH, this.GENDER)
        return pathBuilder

    def getFrontPath(this):
        pathBuilder = os.path.join(this.RENDER_RGB_PATH, this.GENDER)
        pathBuilder = os.path.join(pathBuilder, "front")
        pathBuilder = os.path.join(pathBuilder, this.SUB_FOLDER)
        return pathBuilder

    def getSidePath(this):
        pathBuilder = os.path.join(this.RENDER_RGB_PATH, this.GENDER)
        pathBuilder = os.path.join(pathBuilder, "side")
        pathBuilder = os.path.join(pathBuilder, this.SUB_FOLDER)
        return pathBuilder

    def getTrainH5Path(this):
        return os.path.join(this.OUTPUT, f"train_{this.GENDER}_{this.SUB_FOLDER}.h5")

    def getValidateH5Path(this):
        return os.path.join(this.OUTPUT, f"validate_{this.GENDER}_{this.SUB_FOLDER}.h5")

    def getTestH5Path(this):
        return os.path.join(this.OUTPUT, f"test_{this.GENDER}_{this.SUB_FOLDER}.h5")

    def getMeasurementH5Path(this):
        return os.path.join(this.OUTPUT, f"{this.GENDER}.h5")

    def getModelPath(this, timestamp, epoch_number):
        filename = f"model_{timestamp}_{epoch_number}.ckpt"
        return os.path.join(this.MODEL_PATH, filename)