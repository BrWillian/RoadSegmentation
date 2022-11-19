import os


class Utils(object):

    @staticmethod
    def get_imgs_from_directory(path) -> list:
        data = []
        extensions = ('.jpg', '.jpeg', '.png')

        for root, path, files in os.walk(path):
            for filename in files:
                if filename.endswith(extensions):
                    data.append(filename)

        return data