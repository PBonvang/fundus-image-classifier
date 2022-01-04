import enum

class SAMPLE_TYPE(enum.Enum):
    NOT_FUNDUS = 0
    FUNDUS = 1

    def to_sample_type(image_name: str):
        if image_name[0] == '1':
            return SAMPLE_TYPE.FUNDUS

        return SAMPLE_TYPE.NOT_FUNDUS