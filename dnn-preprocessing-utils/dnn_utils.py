import sys
import os
import cv2 as cv

def add_argument(zoo, parser, name, help, required=False, default=None, type=None, action=None, nargs=None):
    """
    Add argument to parser, optionally reading default values from a model zoo file.
    """
    if len(sys.argv) <= 1:
        return

    modelName = sys.argv[1]

    if os.path.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        node = fs.getNode(modelName)
        if not node.empty():
            value = node.getNode(name)
            if not value.empty():
                if value.isReal():
                    default = value.real()
                elif value.isString():
                    default = value.string()
                elif value.isInt():
                    default = int(value.real())
                elif value.isSeq():
                    default = []
                    for i in range(value.size()):
                        v = value.at(i)
                        if v.isInt():
                            default.append(int(v.real()))
                        elif v.isReal():
                            default.append(v.real())
                        else:
                            print('Unexpected value format')
                            exit(0)
                else:
                    print('Unexpected field format')
                    exit(0)
                required = False

    if action == 'store_true':
        default = 1 if default == 'true' else (0 if default == 'false' else default)
        assert(default is None or default == 0 or default == 1)
        parser.add_argument('--' + name, required=required, help=help, default=bool(default),
                            action=action)
    else:
        parser.add_argument('--' + name, required=required, help=help, default=default,
                            action=action, nargs=nargs, type=type)

def add_preproc_args(zoo, parser, sample):
    """
    Add preprocessing arguments for a given model sample.
    """
    aliases = []
    if os.path.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        root = fs.root()
        for name in root.keys():
            model = root.getNode(name)
            if model.getNode('sample').string() == sample:
                aliases.append(name)

    parser.add_argument('alias', nargs='?', choices=aliases,
                        help='Alias name of model to extract preprocessing parameters from models.yml file.')
    
    add_argument(zoo, parser, 'model', required=True,
                 help='Path to a binary file of trained model weights (.caffemodel, .pb, .t7, .weights, .bin)')
    add_argument(zoo, parser, 'config',
                 help='Path to network configuration file (.prototxt, .pbtxt, .cfg, .xml)')
    add_argument(zoo, parser, 'mean', nargs='+', type=float, default=[0, 0, 0],
                 help='Preprocess input image by subtracting mean values (BGR order).')
    add_argument(zoo, parser, 'scale', type=float, default=1.0,
                 help='Preprocess input image by multiplying by scale factor.')
    add_argument(zoo, parser, 'width', type=int, help='Resize input image to specific width.')
    add_argument(zoo, parser, 'height', type=int, help='Resize input image to specific height.')
    add_argument(zoo, parser, 'rgb', action='store_true',
                 help='Indicate model uses RGB input instead of BGR.')
    add_argument(zoo, parser, 'classes',
                 help='Optional path to text file with class names for object detection.')

def findFile(filename):
    """
    Resolve file path, searching in OpenCV test/data directories if needed.
    """
    if filename:
        if os.path.exists(filename):
            return filename

        fpath = cv.samples.findFile(filename, False)
        if fpath:
            return fpath

        samplesDataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'data', 'dnn')
        if os.path.exists(os.path.join(samplesDataDir, filename)):
            return os.path.join(samplesDataDir, filename)

        for path in ['OPENCV_DNN_TEST_DATA_PATH', 'OPENCV_TEST_DATA_PATH']:
            try:
                extraPath = os.environ[path]
                absPath = os.path.join(extraPath, 'dnn', filename)
                if os.path.exists(absPath):
                    return absPath
            except KeyError:
                pass

        print(f'File {filename} not found! Please specify a valid path.')
        exit(0)
