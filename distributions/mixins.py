class ComponentModel(object):

    def group_create(self, values=[]):
        group = self.Group()
        self.group_init(group)
        for value in values:
            self.group_add_value(group, value)
        return group


class Serializable(object):

    @classmethod
    def model_load(cls, raw):
        model = cls()
        model.load(raw)
        return model

    @staticmethod
    def model_dump(model):
        return model.dump()

    @classmethod
    def group_load(cls, raw):
        group = cls.Group()
        group.load(raw)
        return group

    @staticmethod
    def group_dump(group):
        return group.dump()
